from typing import List
from schemas.document_schema import Document
from schemas.chunk_schema import Chunk

def check_density(documents: List[Document], chunks: List[Chunk]) -> List[Document]:
    doc_id_to_chunk_count = {}
    for chunk in chunks:
        doc_id_to_chunk_count[chunk.doc_id] = doc_id_to_chunk_count.get(chunk.doc_id, 0) + 1
    
    # Simple density score: number of chunks
    # Could be more complex (e.g., avg words per chunk)
    return doc_id_to_chunk_count
