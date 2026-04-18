from typing import List, Dict
from schemas.document_schema import Document

def get_cluster_distribution(documents: List[Document]) -> Dict[int, int]:
    dist = {}
    for doc in documents:
        if doc.cluster_id is not None:
            dist[doc.cluster_id] = dist.get(doc.cluster_id, 0) + 1
    return dist
