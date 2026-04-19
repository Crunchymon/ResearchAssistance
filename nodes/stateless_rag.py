import numpy as np
from typing import List
from schemas.chunk_schema import Chunk
from utils.config import FAST_LLM_MODEL
from utils.observability import logged_chat_completion

def stateless_rag_answer(query: str, chunks: List[Chunk], client, embedding_model) -> str:
    # 1. Embed query
    query_vec = embedding_model.encode(query, normalize_embeddings=True)
    
    # 2. Retrieve top-K chunks
    similarities = []
    valid_chunks = [c for c in chunks if c.embedding]
    for chunk in valid_chunks:
        chunk_vec = np.array(chunk.embedding)
        sim = np.dot(chunk_vec, query_vec) / (np.linalg.norm(chunk_vec) * np.linalg.norm(query_vec))
        similarities.append((chunk, sim))
        
    top_chunks = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
    
    context = "\n\n".join([f"Source: {c.url}\nContent: {c.text}" for c, _ in top_chunks])
    
    # 3. Generate answer
    prompt = f"""Answer the user's question using ONLY the provided context chunks.

Question: {query}

Context:
{context}

Rules:
- Answer strictly using the context.
- Cite the source URL for each claim you make.
- If the answer isn't in the context, say you don't know.
"""

    response = logged_chat_completion(
        client,
        node_name="stateless_rag",
        model=FAST_LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful research assistant. Answer only from the context provided. Always cite sources."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    
    return response.choices[0].message.content
