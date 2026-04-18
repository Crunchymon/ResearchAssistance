from pydantic import BaseModel, Field
from typing import List, Optional

class Document(BaseModel):
    id: str
    url: str
    title: str
    raw_html: Optional[str] = None
    embedding: Optional[List[float]] = None
    cluster_id: Optional[int] = None
    relevance_score: Optional[float] = None
