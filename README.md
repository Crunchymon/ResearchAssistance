# Agentic Research Assistant

**A Human-in-the-Loop RAG and Synthesis Framework**

## Abstract

An agentic research assistant designed to introduce transparency and granular control into Retrieval-Augmented Generation (RAG) systems. This framework utilizes a 13-step pipeline featuring dynamic query decomposition, parallel web retrieval, vector embedding aggregation, and a two-tier LLM extraction protocol. By incorporating dual-halt user control layers, the system ensures human oversight at critical junctures, producing policy-grade, heavily cited research briefs with dynamic thematic synthesis while mitigating API cost bottlenecks through optimized RAG clustering.

---

## Problem Statement

Current AI-driven research assistants and chatbots are highly capable and generate genuinely good, coherent outputs. However, their primary limitation lies in their **opacity and lack of granular tweakability**. Users feed a complex prompt into the system and receive a finished report, but they cannot easily:
- Intervene mid-process to adjust search parameters
- Curate the specific sources being read
- Refine the extraction logic

**Our Solution:** Build a system that maintains high-quality synthesis while returning control to the user through a transparent, halt-driven interface.

---

## System Architecture

The pipeline utilizes a **13-node agentic workflow**, separating the retrieval, extraction, and synthesis layers to allow for human intervention at critical junctures.

### The 13-Step Pipeline

1. **Query Input** — User provides the research question
2. **Structured Decomposition** — Heavy LLM decomposes query into 4 targeted dimensions (Mechanism, Systemic Costs, Differentiation, Policy)
3. **User Control Layer 1 (Query Tuning)** — User reviews and modifies search strings before execution
4. **Multi-Query Parallel Search** — Execute decomposed queries via web search API
5. **Data Cleaning & Text Chunking** — Strip HTML, normalize text, split into 100-200 word chunks
6. **Vector Embedding Generation** — Convert chunks to high-dimensional embeddings
7. **Chunk Aggregation & Document Clustering** — Aggregate chunk embeddings to document level; apply K-Means clustering
8. **Evaluation Node** — Score chunks for relevance and density
9. **User Control Layer 2 (Source Curation)** — User approves/rejects sources based on thematic clusters and relevance scores
10. **Chunk Retrieval & Fact Extraction** — Fast LLM with Boolean Gatekeeper protocol extracts facts from top chunks per cluster
11. **Gatekeeper Node** — Zero-cost protocol filters weak extractions
12. **Final Report Generation** — Heavy LLM synthesizes verified facts into policy brief with dynamic section headings
13. **Stateless Document RAG Chat** — Users interrogate research without conversational memory drift

---

## Methodology

### Structured Decomposition
A Heavy LLM identifies core entities within the user's prompt and decomposes the query into exactly **four highly targeted, keyword-rich search strings**. This prevents topic drift and ensures a 360-degree systems analysis.

### Lexical Preprocessing and Embedding
Scraped HTML is stripped of scripts and ads, normalized, and split into paragraphs of 100–200 words. These chunks are converted into high-dimensional vector embeddings.

### Aggregation and Clustering
Document-level embeddings are derived by averaging their constituent chunks:

$$E_{\text{doc}} = \frac{1}{N} \sum_{i=1}^{N} E_{\text{chunk}_i}$$

K-Means clustering is then applied directly to these aggregated vectors to group sources into thematic clusters for user curation.

### Two-Tier LLM Extraction (The Boolean Gate)
To extract facts, the system retrieves the top relevant chunks for each thematic cluster. A fast LLM is utilized with a strict **"Boolean Gatekeeper" prompt**, ensuring the model explicitly returns 'true' for both presence of core entities and concrete data before extracting any facts—effectively pruning irrelevant generalizations.

### Final Synthesis
A Heavy LLM synthesizes the verified, cited facts. The prompt enforces a strict separation of constructs, requires explicit causal chains, and demands dynamic, journalistic section headings rather than rigid sub-query titles.

---

## Optimization & Design Decisions

### RAG Implementation
Initial designs utilized a Heavy LLM to brute-force summarize every retrieved resource in its entirety, which immediately exhausted API free-tier limits. The system was redesigned around **Retrieval-Augmented Generation (RAG)**, embedding chunks and passing only the highest-value data to models.

### Stateless Chat Integration
By shifting to a vector database for the RAG pivot, the system natively unlocked the ability to support a **stateless chatbot (Node 13)**, allowing users to interrogate the research directly without conversational memory drift.

---

## Setup

### Prerequisites
- Python 3.10+
- API keys for:
  - `GROQ_API_KEY`: Get from [Groq Cloud](https://console.groq.com)
  - `TAVILY_API_KEY`: Get from [Tavily](https://tavily.com)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd AiResearchAgentV2
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

5. **Run the application:**
   ```bash
   streamlit run app/main.py
   ```

The app will launch at `http://localhost:8501`.

---

## Features

### 1. Query Tuning (Control Layer 1)
- **Automatic Decomposition:** LLM breaks your research question into 4 targeted search dimensions
- **Manual Refinement:** Edit and approve each search string before the system executes web searches
- **Transparency:** See exactly what the system will search for

### 2. Source Curation (Control Layer 2)
- **Thematic Clustering:** Sources are automatically grouped by semantic similarity
- **Relevance Scoring:** Each source is scored for relevance and information density
- **Manual Approval:** Select which sources should feed into the synthesis pipeline
- **Custom Injection:** Add your own PDFs or sources for inclusion

### 3. Fact Extraction (Boolean Gatekeeper)
- **Fast LLM Gate:** Quick filtering using a Boolean extraction protocol
- **Entity Verification:** Confirms presence of core entities and concrete data
- **Citation Tracking:** Every extracted fact is tied to its source chunk

### 4. Report Generation
- **Policy-Grade Output:** Heavy LLM synthesis produces publication-ready briefs
- **Dynamic Structure:** Section headings reflect actual findings, not static query dimensions
- **Full Citation Trail:** Every claim includes source attribution

### 5. Stateless RAG Chat
- **Vector-Backed Search:** Ask follow-up questions grounded strictly in the research
- **No Memory Drift:** Each query is independent, ensuring consistency
- **Source Tracing:** See which documents back each answer

---

## Usage Example

1. **Enter Research Query:**
   > "What are the economic implications of carbon pricing in developing nations?"

2. **Review Decomposed Queries:**
   The system generates:
   - Mechanism: How carbon pricing mechanisms function in policy frameworks
   - Systemic Costs: Macroeconomic impact on developing economies
   - Differentiation: How developing nation policies differ from OECD approaches
   - Policy: Current carbon pricing policies in developing regions

3. **Approve Search Strings:**
   Modify if needed, then proceed

4. **Review Source Clustering:**
   System fetches ~40–60 sources and clusters them by theme
   
5. **Curate Sources:**
   Select 8–12 high-quality sources per cluster

6. **Read Generated Report:**
   Policy brief with cited facts synthesized from approved sources

7. **Ask Follow-Up Questions:**
   "How do carbon prices in India compare to Nigeria?" (grounded in research)

---

## Performance & Limitations

### Strengths
- **Transparent:** User controls every stage of the pipeline
- **Cost-Optimized:** RAG approach minimizes LLM API consumption
- **Citation-Rich:** Every claim is traceable to a source
- **Scalable:** Modular architecture supports custom evaluation metrics and extraction logic

### Limitations
- **Latency:** Multi-minute workflow due to 13 nodes, web scraping, and human-in-the-loop pauses
- **Source Dependency:** Output quality depends entirely on search API results; low-density sources must be filtered during curation
- **API Costs:** Requires active API subscriptions (Groq, Tavily)

---

## Team Contributions

- **Ishan Maheshwari** (2401010194): Defined problem statement and transparent architecture vision; spearheaded pitch strategy
- **Anurag Kumar Tiwari** (2401020009): Designed core pipeline flow; managed search integration, text chunking, and embedding aggregation
- **Suryansh Chattree** (2401010466): Led architectural pivot to RAG system; engineered Boolean Gatekeeper and Heavy LLM synthesis framework
- **Sarabjeet Singh** (2401010417): Conducted limitation analysis; managed evaluation metrics and groundwork for stateless RAG chat

---

## Project Structure

```
AiResearchAgentV2/
├── app/                    # Streamlit UI and orchestration
│   ├── main.py
│   ├── graph_workflow.py   # LangGraph workflow
│   ├── state_machine.py    # State management
│   └── ui_components.py    # UI helpers
├── nodes/                  # Pipeline node implementations
│   ├── decomposition.py    # Query decomposition
│   ├── search.py           # Web search integration
│   ├── embed.py            # Embedding generation
│   ├── cluster.py          # K-Means clustering (2D PCA)
│   ├── extract_facts.py    # Fact extraction
│   ├── gatekeeper.py       # Boolean gate filtering
│   ├── generate_report.py  # Final synthesis
│   └── ...
├── schemas/                # Data models
│   ├── state_schema.py
│   ├── query_schema.py
│   ├── chunk_schema.py
│   └── ...
├── evals/                  # Evaluation metrics
│   ├── relevance_scorer.py
│   ├── density_checker.py
│   └── ...
├── utils/                  # Utilities
│   ├── llm_clients.py      # LLM integrations
│   ├── config.py
│   └── guardrails.py
└── requirements.txt        # Python dependencies
```

---

## References

- **Groq API:** https://console.groq.com
- **Tavily Search:** https://tavily.com
- **LangGraph:** https://langchain-ai.github.io/langgraph/
- **Streamlit:** https://streamlit.io

---

## License

This project is part of an academic research initiative. See the [Agent.md](Agent.md) and [PROJECT_FLOW.md](PROJECT_FLOW.md) for additional technical documentation.
