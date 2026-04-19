from typing import List, Dict, Optional
from schemas.fact_schema import Fact
from utils.config import HEAVY_LLM_MODEL
from utils.observability import logged_chat_completion

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

    prompt = f"""You are an expert research analyst. Your task is to synthesize the provided factual evidence into a cohesive, engaging research report that directly answers the Original Query.

RULES FOR SYNTHESIS:
1. Zero Hallucination: You must rely entirely on the provided facts. Do not inject outside knowledge, assumptions, or unverified claims.
2. Structural Anchor: You MUST structure the body of the report around the provided Sub-queries. Create a distinct section for each sub-query to ensure every dimension of the research is covered.
3. Dynamic Headings: Do NOT use the raw sub-queries as your section headers. Generate a dynamic, engaging overarching Title for the report, and write descriptive, professional headings for each section that capture the specific narrative of that data (e.g., transform '"TikTok algorithm impact"' into 'The Neurological Toll of Infinite Scrolling').
4. Narrative Weaving: Do not simply list bullet points. Weave the facts into cohesive paragraphs. Move naturally from context, to evidence, to implications within each section.
5. Citation Integrity: Every claim must be supported by its corresponding source. Embed the citations smoothly into the prose using the provided [Source URL].
6. Deduplication: If a fact appears to fit multiple sections, place it where it best advances the logical flow and avoid repetition.

REPORT FLOW:
- [Dynamic Report Title]
- A brief, organic introduction synthesizing the core findings.
- [Dynamic Heading 1] (Based on Sub-query 1)
- [Dynamic Heading 2] (Based on Sub-query 2)
- [Dynamic Heading 3] (Based on Sub-query 3)
- [Dynamic Heading 4] (Based on Sub-query 4)
- A brief, organic conclusion highlighting evidence gaps or final implications.

INPUT DATA:
Original Query: {original_query}

Sub-queries (Use these to anchor your sections):
{sub_query_block}

Verified Factual Points:
{context}
"""
    
    response = logged_chat_completion(
        client,
        node_name="generate_report",
        model=HEAVY_LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a senior research writer. Write professional, cited reports based strictly on provided facts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    
    return response.choices[0].message.content
