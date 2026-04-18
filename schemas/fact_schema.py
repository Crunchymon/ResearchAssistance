from pydantic import BaseModel, Field
from typing import Optional

class Fact(BaseModel):
    sub_query: str
    fact: str
    source: str
    cluster_id: Optional[int] = None
