# 🧠 Agentic Research Assistant

A human-in-the-loop research assistant that breaks down broad queries, searches the web, and generates cited reports.

## 🚀 Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   Copy `.env.example` to `.env` and add your API keys:
   - `GROQ_API_KEY`: Get from Groq Cloud.
   - `TAVILY_API_KEY`: Get from Tavily.

3. **Run the application:**
   ```bash
   streamlit run app/main.py
   ```

## 🛠️ Features
- **Query Tuning:** Approve and edit sub-queries before searching.
- **Source Curation:** Select/remove web sources based on relevance and cluster distribution.
- **Fact Extraction:** Precise factual bullet points with source citations.
- **Stateless RAG:** Ask follow-up questions grounded strictly in the research documents.
