import json
from typing import List
from utils.config import HEAVY_LLM_MODEL

def decompose_query(query: str, client) -> List[str]:
    prompt = f"""You are a research strategist. Decompose the following research query into exactly four distinct, highly targeted web search queries to ensure comprehensive coverage.

Each sub-query must be optimized for a search engine and target a specific dimension of the topic:
1. Mechanism — Focus on how it works, underlying processes, or core technologies.
2. Impact — Focus on effects, consequences, outcomes, or real-world applications.
3. Evidence — Focus on data, case studies, key findings, or scientific research.
4. Contradictions — Focus on opposing views, limitations, current debates, or alternative theories.

Rules:
- Write the queries as direct, keyword-rich search terms (e.g., "climate change economic impact data"), NOT conversational questions.
- Do not include the dimension labels (e.g., "Mechanism:") in the output strings.
- You must output exactly 4 strings.
- Output ONLY a valid JSON object containing a "queries" key whose value is a list of strings. Do not include markdown blocks, explanations, or conversational filler.
- Example output format: {{"queries": ["search query 1", "search query 2", "search query 3", "search query 4"]}}

Research Query: {query}
"""
    
    response = client.chat.completions.create(
        model=HEAVY_LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a research assistant that decomposes complex queries into searchable sub-queries. Respond only with a JSON object containing a 'queries' array."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
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
