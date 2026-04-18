from typing import List, Dict, Optional
from schemas.fact_schema import Fact
from utils.config import HEAVY_LLM_MODEL

def generate_report(
    grouped_facts: Dict[str, List[Fact]],
    client,
    original_query: str = "",
    sub_queries: Optional[List[str]] = None,
) -> str:
    if not grouped_facts:
        return "No sufficient data found to generate a report."

    sub_queries = sub_queries or []

    context = ""
    for section_key, facts in grouped_facts.items():
        context += f"Candidate Section: {section_key}\n"
        for f in facts:
            cluster_hint = f" (Cluster {f.cluster_id})" if f.cluster_id is not None else ""
            context += f"- {f.fact}{cluster_hint} [{f.source}]\n"
        context += "\n"

    sub_query_block = "\n".join([f"- {sq}" for sq in sub_queries]) if sub_queries else "- Use best available section mapping"

    prompt = f"""Write a comprehensive, logically ordered research report based ONLY on the factual evidence below.

Rules:
- Use only the provided facts.
- Keep the citations [Source URL] intact in the text.
- No hallucinations or outside knowledge.
- Anchor the report to the Original Query and Sub-queries.
- Use the Sub-queries as the main section structure, in the same order.
- Integrate cluster-level evidence INTO these sub-query sections; do NOT create standalone cluster-summary sections.
- If a fact appears to fit multiple sections, place it where it best advances logical flow and avoid repetition.
- Use this strict report structure:
  1) Executive Summary
  2) Sub-query Sections (in listed order)
  3) Cross-Section Synthesis
  4) Evidence Gaps and Uncertainties
- In each sub-query section, move from foundational context -> current evidence -> implications.
- Keep prose cohesive, concise, and professional.

Original Query:
{original_query or "Not provided"}

Sub-queries (ordered):
{sub_query_block}

Factual Points:
{context}
"""
    
    response = client.chat.completions.create(
        model=HEAVY_LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a senior research writer. Write professional, cited reports based strictly on provided facts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content
