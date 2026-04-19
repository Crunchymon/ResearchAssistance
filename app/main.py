import streamlit as st
import sys
import os
import uuid
import importlib
from io import BytesIO

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import state_machine as sm
import ui_components as ui
from schemas.document_schema import Document
from schemas.state_schema import AppPhase
from nodes import decomposition, search, clean_chunk, embed, aggregate, cluster, extract_facts, gatekeeper, generate_report, stateless_rag
from evals import relevance_scorer
from utils import llm_clients
from utils import observability as obs

st.set_page_config(page_title="Research Assistant", layout="wide")

sm.init_state()
ui.inject_theme()

@st.cache_resource
def load_clients():
    return llm_clients.get_groq_client(), llm_clients.get_embedding_model()

groq_client, embedding_model = load_clients()

ui.render_icon_title("manage_search", "Research Assistant", level=1)

phase = st.session_state.app_state.phase
obs.set_run_context(st.session_state.get("run_id"), phase=phase.value)

if "user_resource_urls" not in st.session_state:
    st.session_state.user_resource_urls = set()

if "visited_phase_values" not in st.session_state:
    st.session_state.visited_phase_values = {phase.value}
st.session_state.visited_phase_values.add(phase.value)

if "run_id" not in st.session_state:
    st.session_state.run_id = None
obs.set_original_query(st.session_state.get("query_data", {}).get("original", ""))


def _normalize_urls(raw_urls: str):
    urls = []
    for token in (raw_urls or "").replace(",", "\n").splitlines():
        value = token.strip()
        if not value:
            continue
        if value.startswith("http://") or value.startswith("https://"):
            urls.append(value)
    return list(dict.fromkeys(urls))


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    parser_cls = None
    for module_name in ["pypdf", "PyPDF2"]:
        try:
            module = importlib.import_module(module_name)
            parser_cls = getattr(module, "PdfReader", None)
            if parser_cls:
                break
        except Exception:
            continue

    if parser_cls is None:
        return ""

    reader = parser_cls(BytesIO(pdf_bytes))
    page_text = []
    for page in reader.pages:
        content = page.extract_text() or ""
        if content.strip():
            page_text.append(content)
    return "\n".join(page_text)


def _ingest_new_documents(new_documents):
    if not new_documents:
        return

    obs.set_run_context(st.session_state.get("run_id"), phase=AppPhase.SOURCE_CURATION.value)
    with obs.node_span(
        "manual_ingest.clean_chunk",
        {"new_documents": len(new_documents)},
    ):
        new_chunks = clean_chunk.clean_and_chunk(new_documents)
    obs.log_node_output("manual_ingest.clean_chunk", {"new_chunks": len(new_chunks)})

    if not new_chunks:
        return

    with obs.node_span("manual_ingest.embed", {"chunks": len(new_chunks)}):
        new_chunks = embed.embed_chunks(new_chunks, embedding_model)
    obs.log_node_output("manual_ingest.embed", {"embedded_chunks": len(new_chunks)})

    all_documents = st.session_state.documents + new_documents
    all_chunks = st.session_state.chunks + new_chunks
    with obs.node_span(
        "manual_ingest.aggregate",
        {"chunks": len(all_chunks), "documents": len(all_documents)},
    ):
        all_documents = aggregate.aggregate(all_chunks, all_documents)
    obs.log_node_output("manual_ingest.aggregate", {"documents": len(all_documents)})

    main_query_embedding = embedding_model.encode(st.session_state.query_data["original"])
    with obs.node_span(
        "manual_ingest.relevance",
        {"documents": len(all_documents)},
    ):
        all_documents = relevance_scorer.calculate_relevance(all_documents, main_query_embedding)
    obs.log_node_output("manual_ingest.relevance", {"documents": len(all_documents)})

    st.session_state.documents = all_documents
    st.session_state.chunks = all_chunks


def _active_chunks_from_selected_ids(selected_doc_ids):
    selected_set = set(selected_doc_ids)
    return [
        chunk
        for chunk in st.session_state.chunks
        if chunk.doc_id in selected_set and chunk.embedding
    ]


def _reset_from_phase(target_phase: AppPhase):
    if target_phase == AppPhase.INPUT:
        st.session_state.query_data = {"original": "", "sub_queries": []}
        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.approved_doc_ids = []
        st.session_state.approved_chunks = []
        st.session_state.report = ""
        st.session_state.facts = []
        st.session_state.chat_history = []
        st.session_state.user_resource_urls = set()
        st.session_state.pop("selected_cluster_k", None)
        st.session_state.pop("last_active_chunk_count", None)
        st.session_state.run_id = None
        obs.set_original_query("")
    elif target_phase == AppPhase.QUERY_TUNING:
        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.approved_doc_ids = []
        st.session_state.approved_chunks = []
        st.session_state.report = ""
        st.session_state.facts = []
        st.session_state.chat_history = []
        st.session_state.user_resource_urls = set()
        st.session_state.pop("selected_cluster_k", None)
        st.session_state.pop("last_active_chunk_count", None)
        st.session_state.run_id = None
        obs.set_original_query("")
    elif target_phase == AppPhase.SOURCE_CURATION:
        st.session_state.approved_doc_ids = []
        st.session_state.approved_chunks = []
        st.session_state.report = ""
        st.session_state.facts = []
        st.session_state.chat_history = []


def _navigate_to_phase(target_phase: AppPhase):
    if st.session_state.app_state.phase == target_phase:
        return
    sm.set_phase(target_phase)
    st.rerun()

phase_order = [AppPhase.INPUT, AppPhase.QUERY_TUNING, AppPhase.SOURCE_CURATION, AppPhase.FINAL_REPORT]
phase_names = ["Input", "Query Tuning", "Source Curation", "Report"]
current_idx = phase_order.index(phase) if phase in phase_order else 0
visited_indices = [
    idx for idx, phase_item in enumerate(phase_order)
    if phase_item.value in st.session_state.visited_phase_values
]
max_visited_idx = max(visited_indices) if visited_indices else 0

cols = st.columns(4)
for i, name in enumerate(phase_names):
    with cols[i]:
        can_navigate = i <= max_visited_idx
        button_type = "primary" if i < current_idx else "secondary"
        if st.button(name, key=f"nav_{name}", use_container_width=True, type=button_type, disabled=not can_navigate):
            _navigate_to_phase(phase_order[i])
st.divider()

phase = st.session_state.app_state.phase

if phase == AppPhase.INPUT:
    query = st.text_input("Enter your research question:", placeholder="e.g., What are the latest developments in fusion energy?")
    if st.button("Start Research"):
        if query:
            run_id = obs.new_run_id()
            st.session_state.run_id = run_id
            obs.set_run_context(run_id, phase=AppPhase.INPUT.value)
            obs.set_original_query(query)
            obs.log_event("run_started", {"query": query})
            st.session_state.query_data["original"] = query
            with st.spinner("Decomposing query..."):
                with obs.node_span("decomposition", {"query": query}):
                    sub_queries = decomposition.decompose_query(query, groq_client)
                st.session_state.query_data["sub_queries"] = sub_queries
                obs.log_node_output("decomposition", {"sub_queries": sub_queries, "sub_query_count": len(sub_queries)})
            sm.set_phase(AppPhase.QUERY_TUNING)
            st.rerun()

elif phase == AppPhase.QUERY_TUNING:
    updated_queries = ui.render_query_tuning(st.session_state.query_data["sub_queries"])
    if st.button("Confirm Sub-queries"):
        obs.set_run_context(st.session_state.get("run_id"), phase=AppPhase.QUERY_TUNING.value)
        st.session_state.query_data["sub_queries"] = updated_queries
        with st.spinner("Searching and processing sources..."):
            # Search
            with obs.node_span("search", {"sub_queries": updated_queries, "sub_query_count": len(updated_queries)}):
                docs = search.search(updated_queries)
            obs.log_node_output("search", {"documents": len(docs), "document_ids": [d.id for d in docs]})
            # Clean & Chunk
            with obs.node_span("clean_chunk", {"documents": len(docs)}):
                chunks = clean_chunk.clean_and_chunk(docs)
            obs.log_node_output("clean_chunk", {"chunks": len(chunks)})
            # Embed
            with obs.node_span("embed", {"chunks": len(chunks)}):
                chunks = embed.embed_chunks(chunks, embedding_model)
            obs.log_node_output("embed", {"embedded_chunks": len(chunks)})
            # Aggregate
            with obs.node_span("aggregate", {"chunks": len(chunks), "documents": len(docs)}):
                docs = aggregate.aggregate(chunks, docs)
            obs.log_node_output("aggregate", {"documents": len(docs)})
            # Relevance Eval
            with obs.node_span("relevance_scorer", {"documents": len(docs)}):
                main_query_embedding = embedding_model.encode(st.session_state.query_data["original"])
                docs = relevance_scorer.calculate_relevance(docs, main_query_embedding)
            obs.log_node_output(
                "relevance_scorer",
                {
                    "documents": len(docs),
                    "top_relevance_scores": sorted(
                        [float(d.relevance_score or 0.0) for d in docs],
                        reverse=True,
                    )[:5],
                },
            )
            
            st.session_state.documents = docs
            st.session_state.chunks = chunks
            
        sm.set_phase(AppPhase.SOURCE_CURATION)
        st.rerun()

elif phase == AppPhase.SOURCE_CURATION:
    obs.set_run_context(st.session_state.get("run_id"), phase=AppPhase.SOURCE_CURATION.value)
    ui.render_icon_title("analytics", "Source Curation", level=2)
    left_col, right_col = st.columns([0.56, 0.44], gap="medium")

    if "pdf_uploader_nonce" not in st.session_state:
        st.session_state.pdf_uploader_nonce = 0

    with right_col:
        with st.expander("Add Your Own Resources", expanded=True):
            with st.form("manual_url_form", clear_on_submit=True):
                raw_urls = st.text_area(
                    "Paste URLs (one per line)",
                    placeholder="https://example.com/report\nhttps://example.org/article",
                    key="manual_urls_input",
                )
                add_urls_clicked = st.form_submit_button("Add URLs", use_container_width=True)

            if add_urls_clicked:
                candidate_urls = _normalize_urls(raw_urls)
                existing_urls = {doc.url for doc in st.session_state.documents}
                fetch_urls = [u for u in candidate_urls if u not in existing_urls]
                if fetch_urls:
                    with st.spinner("Fetching URL content..."):
                        new_docs = search.fetch_manual_documents(fetch_urls)
                    _ingest_new_documents(new_docs)
                    st.session_state.user_resource_urls.update({doc.url for doc in new_docs if doc.url})
                    st.rerun()

            uploaded_pdfs = st.file_uploader(
                "Upload PDFs",
                type=["pdf"],
                accept_multiple_files=True,
                key=f"manual_pdf_upload_{st.session_state.pdf_uploader_nonce}",
            )
            if st.button("Add PDFs", use_container_width=True):
                new_docs = []
                existing_urls = {doc.url for doc in st.session_state.documents}
                for uploaded in uploaded_pdfs or []:
                    virtual_url = f"uploaded://{uploaded.name}"
                    if virtual_url in existing_urls:
                        continue
                    try:
                        extracted = _extract_pdf_text(uploaded.getvalue())
                    except Exception:
                        extracted = ""
                    if not extracted.strip():
                        continue
                    new_docs.append(
                        Document(
                            id=str(uuid.uuid4()),
                            url=virtual_url,
                            title=uploaded.name,
                            raw_html=extracted,
                        )
                    )
                _ingest_new_documents(new_docs)
                st.session_state.user_resource_urls.update({doc.url for doc in new_docs if doc.url})
                st.session_state.pdf_uploader_nonce += 1
                st.rerun()

            selected_doc_ids = ui.render_source_links(
                st.session_state.documents,
                container_height=470,
                user_resource_urls=st.session_state.user_resource_urls,
            )

    active_chunks = _active_chunks_from_selected_ids(selected_doc_ids)
    _, _, cluster_meta = cluster.cluster_chunks(active_chunks)
    cluster_min = cluster_meta["min_k"]
    cluster_max = cluster_meta["max_k"]
    auto_k = cluster_meta["auto_k"]
    current_chunk_count = cluster_meta["n_chunks"]

    if st.session_state.get("last_active_chunk_count") != current_chunk_count:
        st.session_state.selected_cluster_k = auto_k
        st.session_state.last_active_chunk_count = current_chunk_count

    if "selected_cluster_k" not in st.session_state:
        st.session_state.selected_cluster_k = auto_k
    if st.session_state.selected_cluster_k < cluster_min or st.session_state.selected_cluster_k > cluster_max:
        st.session_state.selected_cluster_k = auto_k

    with left_col:
        if cluster_max > 1:
            selected_k = st.slider(
                "Number of Clusters",
                min_value=cluster_min,
                max_value=cluster_max,
                value=st.session_state.selected_cluster_k,
                help="Starts at auto-selected k using silhouette/elbow logic.",
                key="selected_cluster_k",
            )
        else:
            selected_k = 1
            st.info("Need at least 2 selected resources for clustering.")

        ui.render_icon_title("analytics", "Cluster PCA", level=3)
        pca_panel = st.container(height=650, border=True)
        with pca_panel:
            clustered_chunks, pca_fig, cluster_meta = cluster.cluster_chunks(active_chunks, requested_k=selected_k)
            if pca_fig:
                st.plotly_chart(pca_fig, use_container_width=True)
                st.caption(
                    f"Auto-k: {cluster_meta['auto_k']} | Selected-k: {cluster_meta['selected_k']} | Active chunks: {cluster_meta['n_chunks']}"
                )
            else:
                st.info("Select at least 2 resources with embeddings to view PCA clusters.")

        generate_clicked = st.button("Generate Report", use_container_width=True)

    if generate_clicked:
        # Filter docs and chunks
        approved_chunks = [c for c in st.session_state.chunks if c.doc_id in selected_doc_ids]
        with obs.node_span(
            "cluster_selected_chunks",
            {
                "approved_doc_ids": selected_doc_ids,
                "approved_chunk_count": len(approved_chunks),
                "selected_k": selected_k,
            },
        ):
            clustered_approved_chunks, _, cluster_metrics = cluster.cluster_chunks(approved_chunks, requested_k=selected_k)
        obs.log_node_output("cluster_selected_chunks", {"cluster_metrics": cluster_metrics})
        st.session_state.approved_doc_ids = selected_doc_ids
        st.session_state.approved_chunks = clustered_approved_chunks
        st.session_state.chat_history = []
        
        with st.spinner("Extracting facts and writing report..."):
            # Extract Facts
            selected_docs = [doc for doc in st.session_state.documents if doc.id in selected_doc_ids]
            with obs.node_span(
                "extract_facts",
                {
                    "approved_chunks": len(clustered_approved_chunks),
                    "selected_docs": len(selected_docs),
                    "top_k": 3,
                },
            ):
                facts = extract_facts.extract_facts(
                    clustered_approved_chunks,
                    selected_docs,
                    groq_client,
                    top_k=3,
                )
            obs.log_node_output("extract_facts", {"facts_count": len(facts), "facts": [f.model_dump() for f in facts]})
            # Gatekeeper
            with obs.node_span("gatekeeper", {"facts_count": len(facts)}):
                grouped_facts = gatekeeper.gatekeeper(facts)
            obs.log_node_output(
                "gatekeeper",
                {
                    "section_count": len(grouped_facts),
                    "facts_per_section": {k: len(v) for k, v in grouped_facts.items()},
                },
            )
            # Generate Report
            with obs.node_span(
                "generate_report",
                {
                    "grouped_sections": len(grouped_facts),
                    "sub_query_count": len(st.session_state.query_data["sub_queries"]),
                },
            ):
                report = generate_report.generate_report(
                    grouped_facts,
                    groq_client,
                    original_query=st.session_state.query_data["original"],
                    sub_queries=st.session_state.query_data["sub_queries"],
                )
            obs.log_node_output("generate_report", {"report_length_chars": len(report or ""), "report": report})
            
            st.session_state.report = report
            st.session_state.facts = facts # Optional

            obs.log_event(
                "run_finished",
                {
                    "status": "report_generated",
                    "approved_doc_count": len(st.session_state.approved_doc_ids),
                    "approved_chunk_count": len(st.session_state.approved_chunks),
                    "facts_count": len(facts),
                    "grouped_section_count": len(grouped_facts),
                    "report_length_chars": len(report or ""),
                },
            )
            
        sm.set_phase(AppPhase.FINAL_REPORT)
        st.rerun()

elif phase == AppPhase.FINAL_REPORT:
    obs.set_run_context(st.session_state.get("run_id"), phase=AppPhase.FINAL_REPORT.value)
    if st.session_state.chat_panel_open:
        left_col, right_col = st.columns([0.72, 0.28], gap="medium")
        with left_col:
            ui.render_icon_title("description", "Research Report", level=2)
            approved_docs = [doc for doc in st.session_state.documents if doc.id in st.session_state.approved_doc_ids]
            ui.render_report_with_source_cards(st.session_state.report, approved_docs, container_height=700)

        with right_col:
            chat_chunks = st.session_state.approved_chunks or st.session_state.chunks
            ui.render_chat_drawer(
                chunks=chat_chunks,
                rag_answer_fn=stateless_rag.stateless_rag_answer,
                client=groq_client,
                embedding_model=embedding_model,
            )
    else:
        header_left, header_right = st.columns([0.86, 0.14])
        with header_left:
            ui.render_icon_title("description", "Research Report", level=2)
        with header_right:
            if st.button("Open Chat", key="chat_panel_open_btn", use_container_width=True):
                st.session_state.chat_panel_open = True
                st.rerun()

        approved_docs = [doc for doc in st.session_state.documents if doc.id in st.session_state.approved_doc_ids]
        ui.render_report_with_source_cards(st.session_state.report, approved_docs, container_height=700)
    
    if st.button("Start New Research"):
        obs.log_event("run_closed", {"reason": "user_started_new_research"})
        _reset_from_phase(AppPhase.INPUT)
        st.session_state.visited_phase_values = {AppPhase.INPUT.value}
        sm.set_phase(AppPhase.INPUT)
        st.rerun()
