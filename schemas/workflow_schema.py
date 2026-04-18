from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from schemas.chunk_schema import Chunk
from schemas.document_schema import Document
from schemas.fact_schema import Fact


class WorkflowPhase(str, Enum):
    INPUT = "input"
    DECOMPOSITION = "decomposition"
    QUERY_TUNING_PAUSE = "query_tuning_pause"
    SEARCH = "search"
    CLEAN_CHUNK = "clean_chunk"
    EMBED = "embed"
    AGGREGATE = "aggregate"
    CLUSTER = "cluster"
    EVALUATE = "evaluate"
    SOURCE_CURATION_PAUSE = "source_curation_pause"
    EXTRACT_FACTS = "extract_facts"
    GATEKEEPER = "gatekeeper"
    GENERATE_REPORT = "generate_report"
    FINAL_REPORT = "final_report"
    CHAT = "chat"
    COMPLETE = "complete"
    ERROR = "error"


class WorkflowState(BaseModel):
    phase: WorkflowPhase = WorkflowPhase.INPUT
    original_query: str = ""
    sub_queries: List[str] = Field(default_factory=list)

    documents: List[Document] = Field(default_factory=list)
    chunks: List[Chunk] = Field(default_factory=list)
    approved_doc_ids: List[str] = Field(default_factory=list)
    approved_chunks: List[Chunk] = Field(default_factory=list)

    facts: List[Fact] = Field(default_factory=list)
    grouped_facts: Dict[str, List[Fact]] = Field(default_factory=dict)
    report: str = ""

    chat_query: str = ""
    chat_answer: str = ""

    errors: List[str] = Field(default_factory=list)
