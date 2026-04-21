

# 🧠 Agentic Research Assistant — Conceptual Implementation Spec

## 1. 📌 Overview
A human-in-the-loop research assistant designed to:
* Break down broad queries into structured, targeted perspectives for parallel searching.
* Pause execution twice: first for **Query Tuning**, and second for **Source Curation**, ensuring maximum user control.
* Use vector embeddings for semantic retrieval, document clustering, and relevance evaluation.
* Calculate document-level embeddings by mathematically averaging their constituent chunks.
* Extract dense factual bullet points using a fast/cheap LLM, routed through a strict Gatekeeper protocol to prune empty or sparse responses.
* Generate structured, cited research reports via a heavy-lifter LLM using only verified bullet points.
* Support grounded follow-up questions via a strictly **stateless** retrieval-augmented generation (RAG) chat interface.

## 2. 🏗️ System Architecture Flow

1. **Query Input**
   * *User provides a high-level research question.*

2. **Structured Decomposition**
   * *The LLM breaks the query down into distinct sub-queries.*

3. ⛔ **User Control Layer (Query Tuning)**
   * *System halts. The user reviews, modifies, or approves the decomposed sub-queries before any external searches begin.*

4. **Multi-Query Search**
   * *System executes parallel web searches based on the approved sub-queries.*

5. **Data Cleaning & Text Chunking**
   * *HTML is stripped; text is normalized and split into chunks.*

6. **Embedding Generation**
   * *All text chunks are converted into vector embeddings.*

7. **Clustering**
   * *System groups chunks into thematic clusters.*

8. **Evaluation Node**
   * *System generates a fast overview of source relevance, cluster distribution, and data density.*

9. ⛔ **User Control Layer (Source Curation)**
    * *System halts. The user selects/removes sources or adds custom URLs/PDFs.* *(If new data is added, go to the **Data Cleaning & Text Chunking** node again for the new sources).*

10. **Chunk Retrieval & Fact Extraction** *(Merged Step)*
    * *System skips outline generation. It retrieves the Top-K chunks based directly on the approved sub-queries.*
    * *A fast, cheap LLM reads these chunks and extracts precise, highly detailed bullet points with `[Source URL]` tags to answer each sub-query.*

11. **Review Node (The Gatekeeper)**
    * *Filters out irrelevant or badly formatted bullet points.*
    * *Zero-Cost Protocol: If a sub-query is left with 0 bullets, it is pruned. If it has 1 bullet, it is merged into an adjacent sub-query's section.*

12. **Final Report Generation**
    * *The heavy-lifter LLM reads the verified bullet points and writes cohesive paragraphs for each sub-query. No raw chunks are passed here.*

13. **Chat Interface (Stateless Document RAG)**
    * *The chat interface has no conversational memory. Every user question is treated as a brand new, isolated search.*
    * *The system embeds the user's question, queries the vector database, and generates an answer grounded strictly in the approved document chunks.*

## 3. 📁 Folder Structure


### 🗂️ Project Tree Structure

```text
agentic-research-assistant/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── state_machine.py
│   └── ui_components.py
├── schemas/
│   ├── __init__.py
│   ├── query_schema.py
│   ├── document_schema.py
│   ├── chunk_schema.py
│   ├── fact_schema.py
│   └── state_schema.py
├── nodes/
│   ├── __init__.py
│   ├── decomposition.py
│   ├── search.py
│   ├── clean_chunk.py
│   ├── embed.py
│   ├── aggregate.py
│   ├── cluster.py
│   ├── extract_facts.py
│   ├── gatekeeper.py
│   ├── generate_report.py
│   └── stateless_rag.py
├── evals/
│   ├── __init__.py
│   ├── relevance_scorer.py
│   ├── cluster_distribution.py
│   └── density_checker.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_schemas/
│   │   ├── test_query_schema.py
│   │   └── test_doc_schema.py
│   ├── test_nodes/
│   │   ├── test_decomposition.py
│   │   ├── test_clean_chunk.py
│   │   └── test_gatekeeper.py
│   ├── test_evals/
│   │   └── test_relevance.py
│   └── test_state_machine.py
└── utils/
    ├── __init__.py
    ├── config.py
    ├── llm_clients.py
    └── validators.py
```

---

### 📄 File Contents Overview

#### Root Files
* **`.env`**: Stores your private API keys (LLMs, search providers). Ignored by Git.
* **`.gitignore`**: Excludes `.env`, `__pycache__/`, `.venv/`, etc.
* **`README.md` & `requirements.txt`**: Standard project documentation and Python dependency list.

#### `app/` (UI & User Control Layers)
* **`main.py`**: The Streamlit/frontend entry point that renders the UI.
* **`state_machine.py`**: The critical logic that pauses the app at the Query Tuning and Source Curation phases without losing data.
* **`ui_components.py`**: Modular layouts for the search bar, the checkbox grids for source approval, and the final reading pane.

#### `schemas/` (Isolated Data Contracts)
* **`query_schema.py`**: Defines the original query and the list of decomposed sub-queries.
* **`document_schema.py`**: Defines the parent URL, Title, Document-level Embedding, and Cluster ID.
* **`chunk_schema.py`**: Defines the raw text snippet, its specific vector embedding, and origin URL.
* **`fact_schema.py`**: The strict schema for the extracted bullet point and mandatory `[Source URL]`.
* **`state_schema.py`**: Tracks which phase the application is currently paused at.

#### `nodes/` (The Core Pipeline)
* **`decomposition.py`**: The LLM prompt logic to break the broad query into sub-queries.
* **`search.py`**: Parallel web search execution based on approved sub-queries.
* **`clean_chunk.py`**: HTML stripping and 100-200 word paragraph splitting.
* **`embed.py`**: Converts the text chunks into vectors.
* **`aggregate.py`**: Mathematically averages chunk vectors to create the document vector.
* **`cluster.py`**: Runs K-Means to group documents into themes.
* **`extract_facts.py`**: Retrieves Top-K chunks and uses the fast LLM to pull bullets.
* **`gatekeeper.py`**: The Zero-Cost Protocol to prune 0-bullet queries and merge 1-bullet queries.
* **`generate_report.py`**: The heavy-lifter LLM reads the vetted facts and writes the cohesive report.
* **`stateless_rag.py`**: The isolated, memory-free chat interface for document querying.

#### `evals/` (Evaluation Node)
* **`relevance_scorer.py`**: Calculates how closely a source matches the intent of the sub-queries.
* **`cluster_distribution.py`**: Maps the thematic groupings of the sources.
* **`density_checker.py`**: Flags sources that have too little meaningful text.

#### `tests/` (Automated Script Testing)
* **`conftest.py`**: Holds reusable mock data (mock embeddings, mock HTML chunks) for your tests.
* **`test_schemas/`**: Validates that your Pydantic/dataclasses correctly reject bad data.
* **`test_nodes/`**: Unit tests for individual pipeline steps (e.g., testing that `clean_chunk.py` actually splits a 500-word text properly).
* **`test_evals/`**: Ensures your scoring mathematics output the expected bounds.
* **`test_state_machine.py`**: Simulates user pauses to ensure session memory persists.

#### `utils/` (Global Helpers)
* **`config.py`**: Safely loads variables from `.env`.
* **`llm_clients.py`**: A centralized place to initialize connections to your fast LLM and heavy-lifter LLM.
* **`validators.py`**: Basic regex or URL health checks.

Alright — this is where we move from **“good architecture” → “agent-executable system spec.”**

I’m going to give you a **node-by-node deep spec**, exactly as you asked:

* tools / libraries
* input schema
* output schema
* function signature
* test cases
* model choice + tokens
* prompt
* where it lives

This will follow your flow in 

---

# 🧠 GLOBAL DEFINITIONS (used everywhere)

---

## Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
dim = 384
normalize = True
```

---

## Fast LLM

* Groq (LLaMA 3 8B / Mixtral-lite)
* temp = 0.2–0.3

---

## Heavy LLM

* Groq (70B class)
* temp = 0.3

---

## Token Budget

```text
chunk size: 100–200 words (~150–250 tokens)
max chunks: 6
context: ~1500–2000 tokens
```

---

# 🧩 NODE 1: Query Input

---

## 📍 Location

```text
app/main.py
schemas/query_schema.py
```

---

## Input Schema

```python
class QueryInput:
    query: str
```

---

## Output Schema

```python
class QueryState:
    query: str
```

---

## Tools

* Streamlit input

---

## Tests

```text
- empty query rejected
- max length constraint
```

---

## LLM

❌ none

---

# 🧩 NODE 2: Structured Decomposition

---

## 📍 Location

```text
nodes/decomposition.py
llm/prompts.py
```

---

## Function

```python
def decompose_query(query: str, llm) -> List[str]
```

---

## Input

```python
query: str
```

---

## Output

```python
List[str]  # 3–4 sub queries
```

---

## Prompt

```text
Break the query into 3–4 distinct search queries:

Rules:
- each must cover a different perspective
- no overlap
- must be directly searchable

Output JSON:
["...", "...", "..."]
```

---

## Model

* Fast LLM
* temp = 0.3
* input tokens: ~50
* output tokens: ~100

---

## Tests

```text
- exactly 3–4 queries
- no duplicates
- non-empty
```

---

# 🧩 NODE 3: Query Tuning (HALT)

---

## 📍 Location

```text
app/state_machine.py
app/ui_components.py
```

---

## Input

```python
sub_queries: List[str]
```

---

## Output

```python
approved_sub_queries: List[str]
```

---

## Tools

* Streamlit editable fields

---

## Tests

```text
- user edits persist
- cannot proceed without at least 1 query
```

---

## LLM

❌ none

---

# 🧩 NODE 4: Multi-Query Search

---

## 📍 Location

```text
nodes/search.py
```

---

## Function

```python
def search(sub_queries: List[str]) -> List[Document]
```

---

## Tools

* Tavily API

---

## Input

```python
List[str]
```

---

## Output

```python
List[Document]:
{
  url,
  title,
  raw_html
}
```

---

## Tests

```text
- returns non-empty results
- valid URLs
```

---

## LLM

❌ none

---

# 🧩 NODE 5: Cleaning + Chunking

---

## 📍 Location

```text
nodes/clean_chunk.py
```

---

## Tools

* trafilatura
* beautifulsoup4

---

## Function

```python
def clean_and_chunk(documents: List[Document]) -> List[Chunk]
```

---

## Input

```python
Document.raw_html
```

---

## Output

```python
Chunk:
{
  id,
  doc_id,
  text,
  url,
  title
}
```

---

## Rules

```text
- remove scripts
- split paragraphs
- merge <80 words
- split >250 words
```

---

## Tests

```text
- chunk size between 100–200 words
- no empty chunks
```

---

## LLM

❌ none

---

# 🧩 NODE 6: Embedding Generation

---

## 📍 Location

```text
nodes/embed.py
```

---

## Tools

* sentence-transformers

---

## Function

```python
def embed_chunks(chunks: List[Chunk]) -> List[Chunk]
```

---

## Output

```python
Chunk.embedding: List[float]
```

---

## Tests

```text
- embedding dim = 384
- no NaN
```

---

## LLM

❌ none

---

# 🧩 NODE 7: Aggregation

---

## 📍 Location

```text
nodes/aggregate.py
```

---

## Function

```python
def aggregate(chunks: List[Chunk]) -> List[Document]
```

---

## Logic

```python
doc_embedding = mean(chunk_embeddings)
```

---

## Output

```python
Document.embedding
```

---

## Tests

```text
- one embedding per doc
- correct grouping
```

---

# 🧩 NODE 8: Clustering

---

## 📍 Location

```text
nodes/cluster.py
```

---

## Tools

* sklearn KMeans

---

## Function

```python
def cluster_documents(doc_embeddings)
```

---

## Logic

```python
k = min(4, n_docs)
```

---

## Output

```python
doc.cluster_id
```

---

# 🧩 NODE 9: Evaluation Node

---

## 📍 Location

```text
evals/
```

---

## Output

```python
{
  relevance_scores,
  cluster_distribution,
  density_scores
}
```

---

## Tools

* numpy
* sklearn cosine similarity

---

## Tests

```text
- scores between 0–1
```

---

# 🧩 NODE 10: Source Curation (HALT)

---

## 📍 Location

```text
app/
```

---

## Output

```python
approved_docs: List[Document]
```

---

# 🧩 NODE 11: Retrieval + Fact Extraction

---

## 📍 Location

```text
nodes/extract_facts.py
```

---

## Function

```python
def extract_facts(sub_queries, chunks, llm) -> List[Fact]
```

---

## Retrieval Algorithm

```text
1. embed sub-queries + original query
2. cosine similarity with chunk embeddings
3. select top 8
4. dedup (>0.85)
5. keep top 6
```

---

## Prompt

```text
Extract factual bullet points.

Rules:
- 1 idea per bullet
- max 2 sentences
- include [Source URL]
- no interpretation

Output JSON:
[
  {
    "sub_query": "...",
    "fact": "...",
    "source": "..."
  }
]
```

---

## Model

* Fast LLM
* input tokens: ~1500
* output tokens: ~400

---

## Tests

```text
- every bullet has source
- JSON valid
```

---

# 🧩 NODE 12: Gatekeeper

---

## 📍 Location

```text
nodes/gatekeeper.py
```

---

## Function

```python
def gatekeeper(facts: List[Fact]) -> List[Fact]
```

---

## Logic

```text
0 facts → mark "No data"
1 fact → mark "Low evidence"
```

---

## Tests

```text
- no data loss
- grouping correct
```

---

# 🧩 NODE 13: Final Report

---

## 📍 Location

```text
nodes/generate_report.py
```

---

## Function

```python
def generate_report(facts, llm) -> str
```

---

## Prompt

```text
Write structured report:

Sections = sub-queries

Rules:
- use only facts
- keep citations
- no hallucination
```

---

## Model

* Heavy LLM
* input tokens: ~1200
* output tokens: ~800–1200

---

# 🧩 NODE 14: Stateless RAG Chat

---

## 📍 Location

```text
nodes/stateless_rag.py
```

---

## Function

```python
def answer(query, chunks, llm)
```

---

## Logic

```text
- embed query
- retrieve chunks
- generate answer
```

---

## Prompt

```text
Answer ONLY using provided chunks.
Cite sources.
```

---

## Model

* Fast LLM
