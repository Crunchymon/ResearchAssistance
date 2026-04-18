from pydantic import BaseModel, Field
from typing import List, Optional

class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    url: str
    title: str
    embedding: Optional[List[float]] = None
    cluster_id: Optional[int] = None
