import json
from typing import List
from utils.config import HEAVY_LLM_MODEL
from utils.observability import logged_chat_completion

def decompose_query(query: str, client) -> List[str]:
    prompt = f"""
    
You are an elite Research Strategist and Systems Analyst. Your task is to decompose the following research query into exactly four distinct, highly technical web search queries to ensure rigorous, mechanism-driven coverage of the topic.

To ensure the original query's exact intent is strictly preserved:
1. Identify the "Core Entities & Constraints" (e.g., the primary subject PLUS specific demographics, locations, or conditions). 
2. These exact entities and constraints MUST be explicitly included in every single sub-query to prevent topic drift. Do NOT drift into broad societal generalities; keep the focus highly analytical and specific to the entities.

Each sub-query must be optimized for a search engine and target a specific, elite dimension of the topic:
1. Mechanism & Architecture — Focus on how the core system, technology, or policy actually functions under the hood (e.g., underlying algorithms, structural processes, technical design, core mechanics).
2. Systemic & Behavioral Costs — Focus on first- and second-order externalities, measurable outcomes, and hidden friction (e.g., cognitive loads, economic trade-offs, unintended behavioral shifts, structural vulnerabilities).
3. Comparative Differentiation — Focus on what makes this specific entity unique compared to alternatives, legacy systems, or direct competitors (e.g., structural advantages, paradigm shifts, specific distinctions).
4. Policy, Governance & Intervention — Focus on systemic critiques, governance models, mitigation strategies, and design implications (e.g., regulatory frameworks, transparency mandates, normative debates, safety interventions).

Rules for Search Syntax:
- Write the queries as direct, keyword-rich search terms.
- Use quotation marks for "exact phrases" to improve precision.
- DO NOT use literal boolean operators like "AND" or "OR". Use natural, space-separated keyword clusters.
- Do NOT write conversational questions.
- Do not include the dimension labels in the output strings.
- Output ONLY a valid JSON object containing a "queries" key whose value is a list of strings. Do not include markdown blocks, explanations, or conversational filler.
- Example output format: {{"queries": ["search query 1", "search query 2", "search query 3", "search query 4"]}}

Research Query: {query}
"""
    
    response = logged_chat_completion(
        client,
        node_name="decomposition",
        model=HEAVY_LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a research assistant that decomposes complex queries into searchable sub-queries. Respond only with a JSON object containing a 'queries' array."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    
    try:
        content = response.choices[0].message.content
        # The model might return {"queries": [...]} or just [...]
        data = json.loads(content)
        if isinstance(data, list):
            return data[:4]
        elif isinstance(data, dict):
            # Try to find a list in the dict
            for key in data:
                if isinstance(data[key], list):
                    return data[key][:4]
        return []
    except Exception as e:
        print(f"Error decomposing query: {e}")
        return []
