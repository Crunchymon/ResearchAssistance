import json
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

from schemas.chunk_schema import Chunk
from schemas.document_schema import Document
from schemas.fact_schema import Fact
from utils.config import FAST_LLM_MODEL
from utils.observability import get_original_query, logged_chat_completion


def _safe_json_loads(payload: str):
    if not payload:
        return {}

    try:
        return json.loads(payload)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", payload)
    if not match:
        return {}

    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(repaired)
        except Exception:
            return {}


def _safe_cluster_id(value) -> int:
    return int(value) if value is not None else 0


def _chunks_by_cluster(chunks: List[Chunk], documents: List[Document]) -> Dict[int, List[Chunk]]:
    doc_cluster_map = {doc.id: _safe_cluster_id(doc.cluster_id) for doc in documents}
    grouped: Dict[int, List[Chunk]] = {}
    for chunk in chunks:
        if not chunk.embedding:
            continue
        cluster_id = _safe_cluster_id(chunk.cluster_id)
        if cluster_id == 0 and chunk.doc_id in doc_cluster_map:
            cluster_id = doc_cluster_map.get(chunk.doc_id, 0)
        grouped.setdefault(cluster_id, []).append(chunk)
    return grouped


def _select_top_k_near_centroid(cluster_chunks: List[Chunk], top_k: int) -> List[Chunk]:
    if not cluster_chunks:
        return []

    embeddings = np.array([chunk.embedding for chunk in cluster_chunks if chunk.embedding])
    if len(embeddings) == 0:
        return []

    centroid = np.mean(embeddings, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0:
        return cluster_chunks[:top_k]

    scored: List[Tuple[Chunk, float]] = []
    for chunk in cluster_chunks:
        vec = np.array(chunk.embedding)
        denom = np.linalg.norm(vec) * centroid_norm
        sim = float(np.dot(vec, centroid) / denom) if denom != 0 else -1.0
        scored.append((chunk, sim))

    top_chunks = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
    return [chunk for chunk, _ in top_chunks]


def _generate_cluster_heading(cluster_id: int, chunks: List[Chunk], client) -> str:
    context = "\n\n".join([f"- {chunk.title}: {chunk.text[:280]}" for chunk in chunks[:3]])
    prompt = f"""Generate a concise section heading (4-8 words) for the cluster below.

Rules:
- Focus on the shared theme only.
- Do not include punctuation at the end.
- Return plain text only.

Cluster ID: {cluster_id}
Cluster Excerpts:
{context}
"""

    try:
        response = logged_chat_completion(
            client,
            node_name="extract_facts.generate_cluster_heading",
            model=FAST_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You create short research section titles."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        heading = response.choices[0].message.content.strip()
        heading = " ".join(heading.split())
        print(f"Generated heading for Cluster {cluster_id}: '{heading}'")
        return heading[:90] if heading else f"Cluster {cluster_id} Findings"
    except Exception:
        print(f"Error generating heading for Cluster {cluster_id}")
        return f"Cluster {cluster_id} Findings"


def _extract_cluster_facts(
    cluster_heading: str,
    chunks: List[Chunk],
    client,
) -> List[Fact]:
    original_query = get_original_query()
    context = "\n\n".join([
        f"Source: {c.url}\nContent: {c.text[:900]}"
        for c in chunks
    ])

    prompt = f"""You are a strict, highly analytical research assistant. Your task is to extract high-density, factual bullet points from the provided sources that directly answer the core research query.

Original Research Query: {original_query or "Not provided"}
Section Heading: {cluster_heading}

RULES FOR EXTRACTION:
1. Relevance Gate: The extracted facts MUST explicitly tie back to the Original Research Query. If a source chunk discusses generalities but lacks the core entities of the original query, DO NOT extract facts from it.
2. High-Density Data Only: Extract concrete findings, metrics, specific outcomes, or defined limitations. 
3. Ban "Claim" Words: Do NOT extract sentences that just announce a topic (e.g., "A study was conducted", "Experts are concerned", "Research suggests"). State the actual finding directly.
4. Max 2 sentences per fact.
5. Include the exact Source URL for every fact.

OUTPUT SCHEMA:
You must output ONLY valid JSON. Evaluate the text logically before extracting.
{{
  "contains_core_entities": true, // Boolean: Does the text explicitly mention the core subjects of the original query?
  "has_concrete_data": true, // Boolean: Does the text contain actual findings, metrics, or concrete facts?
  "reasoning": "Briefly explain why the text qualifies or fails.",
  "facts": [
    {{
      "fact": "...", 
      "source": "https://..."
    }}
  ] // Leave this array completely EMPTY if either boolean above is false.
}}

Sources:
{context}
"""

    response_content = ""
    try:
        response = logged_chat_completion(
            client,
            node_name="extract_facts.extract_cluster_facts.primary",
            model=FAST_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a research assistant extracting factual points. Respond only with valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
            max_completion_tokens=500,
        )
        response_content = response.choices[0].message.content
    except Exception:
        # Fallback: disable strict JSON mode and parse best-effort JSON block.
        try:
            response = logged_chat_completion(
                client,
                node_name="extract_facts.extract_cluster_facts.fallback",
                model=FAST_LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Return compact JSON only with key 'facts' containing fact/source pairs.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_completion_tokens=420,
            )
            response_content = response.choices[0].message.content
        except Exception:
            return []

    facts: List[Fact] = []
    data = _safe_json_loads(response_content)

    try:
        facts_list = data.get("facts", []) if isinstance(data, dict) else []
        for item in facts_list:
            fact_text = item.get("fact") if isinstance(item, dict) else None
            source = item.get("source") if isinstance(item, dict) else None
            if fact_text and source:
                facts.append(Fact(sub_query=cluster_heading, fact=fact_text, source=source))
    except Exception as e:
        print(f"Error parsing facts for section '{cluster_heading}': {e}")
    
    return facts


def extract_facts(
    chunks: List[Chunk],
    documents: List[Document],
    client,
    top_k: int = 3,
) -> List[Fact]:
    all_facts: List[Fact] = []
    grouped_chunks = _chunks_by_cluster(chunks, documents)

    top_chunks_by_cluster: Dict[int, List[Chunk]] = {}
    for cluster_id, cluster_chunks in grouped_chunks.items():
        top_chunks_by_cluster[cluster_id] = _select_top_k_near_centroid(cluster_chunks, top_k=top_k)

    headings: Dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(6, len(top_chunks_by_cluster)))) as executor:
        futures = {
            executor.submit(_generate_cluster_heading, cluster_id, cluster_chunks, client): cluster_id
            for cluster_id, cluster_chunks in top_chunks_by_cluster.items()
            if cluster_chunks
        }
        for future, cluster_id in futures.items():
            headings[cluster_id] = future.result()

    for cluster_id, selected_chunks in top_chunks_by_cluster.items():
        if not selected_chunks:
            continue
        heading = headings.get(cluster_id, f"Cluster {cluster_id} Findings")
        cluster_facts = _extract_cluster_facts(
            heading,
            selected_chunks,
            client,
        )
        for fact in cluster_facts:
            fact.cluster_id = cluster_id
        all_facts.extend(cluster_facts)

    return all_facts
