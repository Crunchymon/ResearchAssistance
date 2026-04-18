from typing import Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl

from schemas.chunk_schema import Chunk
from schemas.document_schema import Document
from schemas.fact_schema import Fact


class DecompositionInput(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)


class DecompositionOutput(BaseModel):
    sub_queries: List[str] = Field(..., min_length=1, max_length=4)


class SearchInput(BaseModel):
    sub_queries: List[str] = Field(..., min_length=1)


class SearchOutput(BaseModel):
    documents: List[Document] = Field(default_factory=list)


class CleanChunkInput(BaseModel):
    documents: List[Document] = Field(default_factory=list)


class CleanChunkOutput(BaseModel):
    chunks: List[Chunk] = Field(default_factory=list)


class EmbedInput(BaseModel):
    chunks: List[Chunk] = Field(default_factory=list)


class EmbedOutput(BaseModel):
    chunks: List[Chunk] = Field(default_factory=list)


class AggregateInput(BaseModel):
    chunks: List[Chunk] = Field(default_factory=list)
    documents: List[Document] = Field(default_factory=list)


class AggregateOutput(BaseModel):
    documents: List[Document] = Field(default_factory=list)


class ClusterInput(BaseModel):
    documents: List[Document] = Field(default_factory=list)


class ClusterOutput(BaseModel):
    documents: List[Document] = Field(default_factory=list)


class RelevanceInput(BaseModel):
    documents: List[Document] = Field(default_factory=list)
    query_embedding: List[float] = Field(default_factory=list)


class RelevanceOutput(BaseModel):
    documents: List[Document] = Field(default_factory=list)


class FactExtractionInput(BaseModel):
    chunks: List[Chunk] = Field(..., min_length=1)
    documents: List[Document] = Field(..., min_length=1)


class FactExtractionOutput(BaseModel):
    facts: List[Fact] = Field(default_factory=list)


class GatekeeperInput(BaseModel):
    facts: List[Fact] = Field(default_factory=list)


class GatekeeperOutput(BaseModel):
    grouped_facts: Dict[str, List[Fact]] = Field(default_factory=dict)


class ReportInput(BaseModel):
    grouped_facts: Dict[str, List[Fact]] = Field(default_factory=dict)


class ReportOutput(BaseModel):
    report: str = ""


class StatelessRagInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    chunks: List[Chunk] = Field(default_factory=list)


class StatelessRagOutput(BaseModel):
    answer: str = ""
    sources: List[HttpUrl] = Field(default_factory=list)


class SourceItem(BaseModel):
    url: HttpUrl
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


class FactItem(BaseModel):
    sub_query: str = Field(..., min_length=1)
    fact: str = Field(..., min_length=1)
    source: HttpUrl


class DecompositionStructuredOutput(BaseModel):
    sub_queries: List[str] = Field(..., min_length=1, max_length=4)


class FactExtractionStructuredOutput(BaseModel):
    facts: List[FactItem] = Field(default_factory=list)


class ReportStructuredOutput(BaseModel):
    report: str = Field(..., min_length=1)


class RagStructuredOutput(BaseModel):
    answer: str = Field(..., min_length=1)
    sources: List[HttpUrl] = Field(default_factory=list)
