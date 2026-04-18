from pydantic import BaseModel, Field
from typing import List

class QueryInput(BaseModel):
    query: str = Field(..., min_length=1, description="The high-level research question.")

class QueryState(BaseModel):
    query: str
    sub_queries: List[str] = Field(default_factory=list)
