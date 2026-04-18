import streamlit as st
import re
from typing import Dict, List, Optional
from urllib.parse import urlparse


def inject_theme() -> None:
    """Inject minimal styling and Material Symbols font."""
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');

.material-symbols-outlined {
  font-family: 'Material Symbols Outlined';
  font-weight: normal;
  font-style: normal;
  font-size: 22px;
  line-height: 1;
  letter-spacing: normal;
  text-transform: none;
  display: inline-block;
  white-space: nowrap;
  word-wrap: normal;
  direction: ltr;
  -webkit-font-feature-settings: 'liga';
  -webkit-font-smoothing: antialiased;
  vertical-align: middle;
}

.title-with-icon {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0.2rem 0;
}

.phase-pill {
  border: 1px solid rgba(120, 120, 120, 0.35);
  border-radius: 10px;
  padding: 0.4rem 0.6rem;
  text-align: center;
  font-size: 0.9rem;
}

.phase-pill.active {
  border-color: rgba(30, 144, 255, 0.6);
}

.source-card-title {
    font-weight: 600;
    margin-bottom: 0.2rem;
}

.source-card-domain {
    font-size: 0.82rem;
    color: rgba(120, 120, 120, 0.95);
}

.citation-token {
    border: 1px solid rgba(120, 120, 120, 0.35);
    border-radius: 6px;
    padding: 0.1rem 0.35rem;
    font-size: 0.82rem;
    margin-right: 0.2rem;
}

.user-resource-badge {
    display: inline-block;
    font-size: 0.74rem;
    font-weight: 600;
    color: #166534;
    background: #dcfce7;
    border: 1px solid #86efac;
    border-radius: 999px;
    padding: 0.08rem 0.45rem;
    margin-top: 0.2rem;
}

div.stButton > button[kind="primary"] {
    background-color: #1f9d55;
    border-color: #1f9d55;
    color: white;
}

div.stButton > button[kind="primary"]:hover {
    background-color: #168447;
    border-color: #168447;
    color: white;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_icon_title(icon_name: str, text: str, level: int = 3) -> None:
    level = max(1, min(level, 6))
    st.markdown(
        f"""
<h{level} class="title-with-icon">
  <span class="material-symbols-outlined">{icon_name}</span>
  <span>{text}</span>
</h{level}>
""",
        unsafe_allow_html=True,
    )


def render_chat_drawer(chunks, rag_answer_fn, client, embedding_model):
    """Render the right-side chat drawer when open."""
    header_left, header_right = st.columns([0.74, 0.26])
    with header_left:
        render_icon_title("forum", "Stateless Chat", level=3)
    with header_right:
        if st.button("Close Chat", key="chat_panel_close", use_container_width=True):
            st.session_state.chat_panel_open = False
            st.rerun()

    chat_container = st.container(height=420, border=True)
    with chat_container:
        for item in st.session_state.chat_history:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])

    user_q = st.chat_input("Ask based on approved sources", key="chat_drawer_input")
    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.spinner("Generating answer..."):
            answer = rag_answer_fn(user_q, chunks, client, embedding_model)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()


def render_query_tuning(sub_queries: List[str]):
    render_icon_title("tune", "Query Tuning", level=2)
    st.write("Review and edit the decomposed sub-queries before searching.")

    updated_queries = []
    for i, sq in enumerate(sub_queries):
        new_sq = st.text_input(f"Sub-query {i + 1}", value=sq, key=f"sq_{i}")
        updated_queries.append(new_sq)

    return updated_queries


def _safe_domain(url: str) -> str:
    if url.startswith("uploaded://"):
        return "Uploaded file"
    return urlparse(url).netloc or "Unknown domain"


def render_source_cards(
    sources: List[Dict[str, str]],
    title_by_url: Optional[Dict[str, str]] = None,
    container_height: int = 260,
    show_header: bool = True,
):
    if show_header:
        render_icon_title("link", "Sources", level=3)

    scroll_container = st.container(height=container_height, border=True)
    with scroll_container:
        for source in sources:
            source_url = source.get("url", "")
            title = source.get("title") or (title_by_url.get(source_url) if title_by_url else source_url)
            domain = _safe_domain(source_url)
            with st.container(border=True):
                col1, col2 = st.columns([0.74, 0.26])
                with col2:
                    if source_url.startswith("http://") or source_url.startswith("https://"):
                        st.link_button("Open", source_url, use_container_width=True)
                    else:
                        st.caption("Local source")
                with col1:
                    st.markdown(f"<div class='source-card-title'>{title}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='source-card-domain'>{domain}</div>", unsafe_allow_html=True)


def render_source_links(documents, container_height: int = 380, user_resource_urls=None):
    render_icon_title("link", "Sources", level=3)
    st.write("Select sources to include in the final report.")

    user_resource_urls = user_resource_urls or set()

    selected_doc_ids = []
    scroll_container = st.container(height=container_height, border=True)
    with scroll_container:
        for doc in documents:
            domain = _safe_domain(doc.url)
            with st.container(border=True):
                col1, col2, col3 = st.columns([0.08, 0.66, 0.26])
                with col1:
                    is_selected = st.checkbox("", value=True, key=f"doc_{doc.id}")
                    if is_selected:
                        selected_doc_ids.append(doc.id)
                with col2:
                    st.markdown(f"<div class='source-card-title'>{doc.title}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='source-card-domain'>{domain}</div>", unsafe_allow_html=True)
                    if doc.url in user_resource_urls:
                        st.markdown("<span class='user-resource-badge'>User resource</span>", unsafe_allow_html=True)
                    score = doc.relevance_score if doc.relevance_score is not None else 0.0
                    st.caption(f"Relevance: {score:.2f} | Cluster: {doc.cluster_id}")
                with col3:
                    if doc.url.startswith("http://") or doc.url.startswith("https://"):
                        st.link_button("Open", doc.url, use_container_width=True)
                    else:
                        st.caption("Local source")

    return selected_doc_ids


def render_report_with_source_cards(report_text: str, documents, container_height: int = 700):
    markdown_link_pattern = r"\[([^\]]+)\]\((https?://[^\s\)<>\"']+)\)"
    bare_url_pattern = r"https?://[^\s\]\)<>\"']+"

    url_to_token = {}
    token_to_url = {}

    def _token_for_url(url: str) -> str:
        if url not in url_to_token:
            idx = len(url_to_token) + 1
            token = f"[S{idx}]"
            url_to_token[url] = token
            token_to_url[token] = url
        return url_to_token[url]

    def _replace_markdown(match):
        label = match.group(1)
        url = match.group(2)
        token = _token_for_url(url)
        return f"{label} {token}"

    cleaned = re.sub(markdown_link_pattern, _replace_markdown, report_text)

    def _replace_bare(match):
        url = match.group(0)
        return _token_for_url(url)

    cleaned = re.sub(bare_url_pattern, _replace_bare, cleaned)

    report_container = st.container(height=container_height, border=True)
    with report_container:
        st.markdown(cleaned)

    if not url_to_token:
        return

    token_rows = " ".join([
        f"<span class='citation-token'>{token}</span>"
        for token in token_to_url.keys()
    ])
    st.markdown(token_rows, unsafe_allow_html=True)

    title_by_url = {doc.url: doc.title for doc in documents if getattr(doc, "url", None)}
    sources = [{"url": url, "title": title_by_url.get(url, url)} for url in url_to_token.keys()]
    render_source_cards(sources, title_by_url=title_by_url, container_height=230, show_header=False)
