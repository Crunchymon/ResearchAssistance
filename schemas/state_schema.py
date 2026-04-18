from pydantic import BaseModel, Field
from enum import Enum

class AppPhase(Enum):
    INPUT = "input"
    QUERY_TUNING = "query_tuning"
    SOURCE_CURATION = "source_curation"
    FINAL_REPORT = "final_report"

class AppState(BaseModel):
    phase: AppPhase = AppPhase.INPUT
