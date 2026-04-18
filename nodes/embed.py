from typing import List
from schemas.chunk_schema import Chunk

def embed_chunks(chunks: List[Chunk], model) -> List[Chunk]:
    if not chunks:
        return []
    
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)
    
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding.tolist()
        
    return chunks
