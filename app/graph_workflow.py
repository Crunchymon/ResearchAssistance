from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from evals import relevance_scorer
from nodes import (
    aggregate,
    clean_chunk,
    cluster,
    decomposition,
    embed,
    extract_facts,
    gatekeeper,
    generate_report,
)
from schemas.node_io_schema import (
    AggregateInput,
    AggregateOutput,
    CleanChunkInput,
    CleanChunkOutput,
    ClusterInput,
    ClusterOutput,
    DecompositionInput,
    DecompositionOutput,
    FactExtractionInput,
    FactExtractionOutput,
    GatekeeperInput,
    GatekeeperOutput,
    RelevanceInput,
    RelevanceOutput,
    ReportInput,
    ReportOutput,
    SearchInput,
    SearchOutput,
)
from schemas.workflow_schema import WorkflowPhase, WorkflowState
from utils.guardrails import GuardrailRunner
from utils import observability as obs
from nodes import search


def build_graph(llm_client, embedding_model):
    """Build LangGraph skeleton for the research pipeline."""

    def decomposition_node(state: WorkflowState) -> WorkflowState:
        valid = GuardrailRunner.validate_input({"query": state.original_query}, DecompositionInput)
        sub_queries = decomposition.decompose_query(valid.query, llm_client)
        parsed = GuardrailRunner.validate_output({"sub_queries": sub_queries}, DecompositionOutput)
        state.sub_queries = parsed.sub_queries
        state.phase = WorkflowPhase.QUERY_TUNING_PAUSE
        return state

    def search_node(state: WorkflowState) -> WorkflowState:
        valid = GuardrailRunner.validate_input({"sub_queries": state.sub_queries}, SearchInput)
        documents = search.search(valid.sub_queries)
        parsed = GuardrailRunner.validate_output({"documents": documents}, SearchOutput)
        state.documents = parsed.documents
        state.phase = WorkflowPhase.CLEAN_CHUNK
        return state

    def clean_chunk_node(state: WorkflowState) -> WorkflowState:
        valid = GuardrailRunner.validate_input({"documents": state.documents}, CleanChunkInput)
        chunks = clean_chunk.clean_and_chunk(valid.documents)
        parsed = GuardrailRunner.validate_output({"chunks": chunks}, CleanChunkOutput)
        state.chunks = parsed.chunks
        state.phase = WorkflowPhase.EMBED
        return state

    def embed_node(state: WorkflowState) -> WorkflowState:
        state.chunks = embed.embed_chunks(state.chunks, embedding_model)
        state.phase = WorkflowPhase.AGGREGATE
        return state

    def aggregate_node(state: WorkflowState) -> WorkflowState:
        valid = GuardrailRunner.validate_input(
            {"chunks": state.chunks, "documents": state.documents},
            AggregateInput,
        )
        documents = aggregate.aggregate(valid.chunks, valid.documents)
        parsed = GuardrailRunner.validate_output({"documents": documents}, AggregateOutput)
        state.documents = parsed.documents
        state.phase = WorkflowPhase.CLUSTER
        return state

    def cluster_node(state: WorkflowState) -> WorkflowState:
        valid = GuardrailRunner.validate_input({"documents": state.documents}, ClusterInput)
        documents, _, _ = cluster.cluster_documents(valid.documents)
        parsed = GuardrailRunner.validate_output({"documents": documents}, ClusterOutput)
        state.documents = parsed.documents
        state.phase = WorkflowPhase.EVALUATE
        return state

    def evaluate_node(state: WorkflowState) -> WorkflowState:
        query_embedding = embedding_model.encode(state.original_query)
        valid = GuardrailRunner.validate_input(
            {"documents": state.documents, "query_embedding": query_embedding},
            RelevanceInput,
        )
        documents = relevance_scorer.calculate_relevance(valid.documents, valid.query_embedding)
        parsed = GuardrailRunner.validate_output({"documents": documents}, RelevanceOutput)
        state.documents = parsed.documents
        state.phase = WorkflowPhase.SOURCE_CURATION_PAUSE
        return state

    def extract_facts_node(state: WorkflowState) -> WorkflowState:
        obs.set_original_query(state.original_query)
        valid = GuardrailRunner.validate_input(
            {"chunks": state.approved_chunks, "documents": state.documents},
            FactExtractionInput,
        )
        facts = extract_facts.extract_facts(
            valid.chunks,
            valid.documents,
            llm_client,
            top_k=3,
        )
        parsed = GuardrailRunner.validate_output({"facts": facts}, FactExtractionOutput)
        state.facts = parsed.facts
        state.phase = WorkflowPhase.GATEKEEPER
        return state

    def gatekeeper_node(state: WorkflowState) -> WorkflowState:
        valid = GuardrailRunner.validate_input(
            {"facts": state.facts},
            GatekeeperInput,
        )
        grouped_facts = gatekeeper.gatekeeper(valid.facts)
        parsed = GuardrailRunner.validate_output({"grouped_facts": grouped_facts}, GatekeeperOutput)
        state.grouped_facts = parsed.grouped_facts
        state.phase = WorkflowPhase.GENERATE_REPORT
        return state

    def generate_report_node(state: WorkflowState) -> WorkflowState:
        valid = GuardrailRunner.validate_input({"grouped_facts": state.grouped_facts}, ReportInput)
        report = generate_report.generate_report(
            valid.grouped_facts,
            llm_client,
            original_query=state.original_query,
            sub_queries=state.sub_queries,
        )
        parsed = GuardrailRunner.validate_output({"report": report}, ReportOutput)
        state.report = parsed.report
        state.phase = WorkflowPhase.FINAL_REPORT
        return state

    graph = StateGraph(WorkflowState)

    graph.add_node("decompose", decomposition_node)
    graph.add_node("search", search_node)
    graph.add_node("clean_chunk", clean_chunk_node)
    graph.add_node("embed", embed_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("cluster", cluster_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("extract_facts", extract_facts_node)
    graph.add_node("gatekeeper", gatekeeper_node)
    graph.add_node("generate_report", generate_report_node)

    graph.add_edge(START, "decompose")
    graph.add_edge("decompose", "search")
    graph.add_edge("search", "clean_chunk")
    graph.add_edge("clean_chunk", "embed")
    graph.add_edge("embed", "aggregate")
    graph.add_edge("aggregate", "cluster")
    graph.add_edge("cluster", "evaluate")
    graph.add_edge("evaluate", "extract_facts")
    graph.add_edge("extract_facts", "gatekeeper")
    graph.add_edge("gatekeeper", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()
