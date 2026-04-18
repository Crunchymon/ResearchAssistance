import numpy as np
from typing import List
from schemas.chunk_schema import Chunk
from schemas.document_schema import Document

def aggregate(chunks: List[Chunk], documents: List[Document]) -> List[Document]:
    doc_id_to_embeddings = {}
    for chunk in chunks:
        if chunk.embedding:
            if chunk.doc_id not in doc_id_to_embeddings:
                doc_id_to_embeddings[chunk.doc_id] = []
            doc_id_to_embeddings[chunk.doc_id].append(chunk.embedding)
            
    for doc in documents:
        if doc.id in doc_id_to_embeddings:
            embeddings = doc_id_to_embeddings[doc.id]
            doc.embedding = np.mean(embeddings, axis=0).tolist()
            
    return documents
