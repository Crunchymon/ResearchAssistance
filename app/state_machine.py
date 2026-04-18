import streamlit as st
from schemas.state_schema import AppPhase, AppState

def init_state():
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState()
    if "query_data" not in st.session_state:
        st.session_state.query_data = {"original": "", "sub_queries": []}
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "report" not in st.session_state:
        st.session_state.report = ""
    if "facts" not in st.session_state:
        st.session_state.facts = []
    if "approved_doc_ids" not in st.session_state:
        st.session_state.approved_doc_ids = []
    if "approved_chunks" not in st.session_state:
        st.session_state.approved_chunks = []
    if "chat_panel_open" not in st.session_state:
        st.session_state.chat_panel_open = True
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "visited_phase_values" not in st.session_state:
        st.session_state.visited_phase_values = {AppPhase.INPUT.value}

def set_phase(phase: AppPhase):
    st.session_state.app_state.phase = phase
