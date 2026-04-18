import trafilatura
from bs4 import BeautifulSoup
from typing import List
from schemas.document_schema import Document
from schemas.chunk_schema import Chunk
import uuid

def clean_and_chunk(documents: List[Document]) -> List[Chunk]:
    all_chunks = []
    
    for doc in documents:
        # Extract text from raw_html if available, otherwise just use empty string or title
        text = ""
        if doc.raw_html:
            # Trafilatura is better for extraction
            text = trafilatura.extract(doc.raw_html, include_links=False, include_images=False, include_tables=False)
            if not text:
                # Fallback to BeautifulSoup if trafilatura fails
                soup = BeautifulSoup(doc.raw_html, 'html.parser')
                text = soup.get_text(separator=' ', strip=True)
        
        if not text:
            continue
            
        # Basic chunking: split into paragraphs first
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        current_chunks = []
        temp_chunk = ""
        
        for p in paragraphs:
            words = p.split()
            # Rules from Agent.md: 100-200 words (~150-250 tokens)
            # split > 250 words
            # merge < 80 words
            
            if len(words) > 250:
                # Split large paragraph
                for i in range(0, len(words), 200):
                    sub_chunk = " ".join(words[i:i+200])
                    current_chunks.append(sub_chunk)
            elif len(words) < 80:
                # Merge small paragraph
                if temp_chunk:
                    temp_chunk += " " + p
                else:
                    temp_chunk = p
                
                if len(temp_chunk.split()) >= 100:
                    current_chunks.append(temp_chunk)
                    temp_chunk = ""
            else:
                if temp_chunk:
                    current_chunks.append(temp_chunk)
                    temp_chunk = ""
                current_chunks.append(p)
        
        if temp_chunk:
            current_chunks.append(temp_chunk)
            
        for chunk_text in current_chunks:
            if len(chunk_text.split()) >= 30: # Minimum size for a chunk
                all_chunks.append(Chunk(
                    id=str(uuid.uuid4()),
                    doc_id=doc.id,
                    text=chunk_text,
                    url=doc.url,
                    title=doc.title
                ))
                
    return all_chunks
