import numpy as np
from typing import List
from schemas.document_schema import Document

def calculate_relevance(documents: List[Document], query_embedding: List[float]) -> List[Document]:
    query_vec = np.array(query_embedding)
    
    for doc in documents:
        if doc.embedding:
            doc_vec = np.array(doc.embedding)
            # Cosine similarity
            similarity = np.dot(doc_vec, query_vec) / (np.linalg.norm(doc_vec) * np.linalg.norm(query_vec))
            doc.relevance_score = float(similarity)
        else:
            doc.relevance_score = 0.0
            
    return documents
