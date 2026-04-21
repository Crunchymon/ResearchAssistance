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

.report-body {
    white-space: pre-line;
    line-height: 1.6;
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
    token_pattern = r"\[S(\d+)\]"

    existing_token_pattern = r"\[\[S\d+\]\]"

    def _is_fragment_line(line: str) -> bool:
        token = line.strip()
        if not token:
            return False
        return bool(re.fullmatch(r"[A-Za-z0-9$€£%/.,:;()\[\]\-]+", token)) and len(token) <= 2

    def _repair_fragmented_lines(text: str) -> str:
        lines = text.splitlines()
        repaired = []
        i = 0
        while i < len(lines):
            if not _is_fragment_line(lines[i]):
                repaired.append(lines[i])
                i += 1
                continue

            j = i
            pieces = []
            while j < len(lines) and _is_fragment_line(lines[j]):
                pieces.append(lines[j].strip())
                j += 1

            # Only merge long runs; short runs are often legitimate bullet/text tokens.
            if len(pieces) >= 6:
                repaired.append("".join(pieces))
            else:
                repaired.extend(pieces)
            i = j

        return "\n".join(repaired)

    processed = (report_text or "").strip()
    processed = _repair_fragmented_lines(processed)
    processed = re.sub(existing_token_pattern, lambda m: m.group(0)[1:-1], processed)
    processed = re.sub(r"\n{2,}", "\n\n", processed)
    processed = re.sub(r"[ \t]+", " ", processed)

    existing_numbers = [int(n) for n in re.findall(token_pattern, processed)]
    next_token_num = (max(existing_numbers) + 1) if existing_numbers else 1

    url_to_token: Dict[str, str] = {}
    token_to_url: Dict[str, str] = {}

    def _token_for_url(url: str, preferred: Optional[str] = None) -> str:
        nonlocal next_token_num
        if url in url_to_token:
            return url_to_token[url]

        token = None
        if preferred and re.fullmatch(r"S\d+", preferred.strip()):
            candidate = preferred.strip()
            taken_url = token_to_url.get(candidate)
            if taken_url is None or taken_url == url:
                token = candidate

        if token is None:
            while f"S{next_token_num}" in token_to_url:
                next_token_num += 1
            token = f"S{next_token_num}"
            next_token_num += 1

        url_to_token[url] = token
        token_to_url[token] = url
        return token

    def _replace_markdown(match):
        label = match.group(1).strip()
        url = match.group(2)
        token = _token_for_url(url, preferred=label)
        return f"[{token}]"

    processed = re.sub(markdown_link_pattern, _replace_markdown, processed)
    def _replace_bare(match):
        url = match.group(0)
        token = _token_for_url(url)
        return f"[{token}]"

    processed = re.sub(bare_url_pattern, _replace_bare, processed)

    lines = processed.split("\n")
    if lines and re.fullmatch(r"\*\*(.+?)\*\*", lines[0].strip()):
        heading = re.fullmatch(r"\*\*(.+?)\*\*", lines[0].strip()).group(1)
        lines[0] = f"## {heading}"
    markdown_text = "\n".join(lines)

    report_container = st.container(height=container_height, border=True)
    with report_container:
        st.markdown(markdown_text)

    if not token_to_url:
        return

    render_icon_title("link", "Sources", level=3)
    title_by_url = {doc.url: doc.title for doc in documents if getattr(doc, "url", None)}

    source_container = st.container(height=240, border=True)
    with source_container:
        sorted_tokens = sorted(token_to_url.keys(), key=lambda t: int(t[1:]))
        for token in sorted_tokens:
            url = token_to_url[token]
            title = title_by_url.get(url, url)
            domain = _safe_domain(url)

            with st.container(border=True):
                c1, c2, c3 = st.columns([0.14, 0.62, 0.24])
                with c1:
                    st.markdown(f"<span class='citation-token'>[{token}]</span>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div class='source-card-title'>{title}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='source-card-domain'>{domain}</div>", unsafe_allow_html=True)
                with c3:
                    if url.startswith("http://") or url.startswith("https://"):
                        st.link_button("Open", url, use_container_width=True, key=f"report_src_{token}")
                    else:
                        st.caption("Local source")
