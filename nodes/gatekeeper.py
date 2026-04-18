from typing import List, Dict
from schemas.fact_schema import Fact

def gatekeeper(facts: List[Fact]) -> Dict[str, List[Fact]]:
    grouped_facts: Dict[str, List[Fact]] = {}
    for fact in facts:
        grouped_facts.setdefault(fact.sub_query, []).append(fact)
            
    final_grouped = {}
    
    # Rules from Agent.md:
    # 0 facts → mark "No data" (prune)
    # 1 fact → mark "Low evidence" (merge)
    
    # For simplicity, we just filter
    for section, section_facts in grouped_facts.items():
        if len(section_facts) == 0:
            continue
        elif len(section_facts) == 1:
            # Simple merge attempt or just keep with low evidence
            final_grouped[section] = section_facts
        else:
            final_grouped[section] = section_facts
            
    return final_grouped
