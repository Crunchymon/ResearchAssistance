import json
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

from schemas.chunk_schema import Chunk
from schemas.document_schema import Document
from schemas.fact_schema import Fact
from utils.config import FAST_LLM_MODEL


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
        response = client.chat.completions.create(
            model=FAST_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You create short research section titles."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        heading = response.choices[0].message.content.strip()
        heading = " ".join(heading.split())
        return heading[:90] if heading else f"Cluster {cluster_id} Findings"
    except Exception:
        return f"Cluster {cluster_id} Findings"


def _extract_cluster_facts(cluster_heading: str, chunks: List[Chunk], client) -> List[Fact]:
    context = "\n\n".join([
        f"Source: {c.url}\nContent: {c.text[:900]}"
        for c in chunks
    ])

    prompt = f"""Extract factual bullet points from the sources below.

Rules:
- Keep each item factual and specific.
- Max 2 sentences per item.
- Include the exact source URL for each item.
- Output only JSON in this shape:
{{
  "facts": [
    {{"fact": "...", "source": "https://..."}}
  ]
}}

Section heading: {cluster_heading}

Sources:
{context}
"""

    response_content = ""
    try:
        response = client.chat.completions.create(
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
            response = client.chat.completions.create(
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


def extract_facts(chunks: List[Chunk], documents: List[Document], client, top_k: int = 3) -> List[Fact]:
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
        cluster_facts = _extract_cluster_facts(heading, selected_chunks, client)
        for fact in cluster_facts:
            fact.cluster_id = cluster_id
        all_facts.extend(cluster_facts)

    return all_facts
