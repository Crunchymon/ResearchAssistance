# 🧠 AI Research Agent - Complete Project Flow

## 📊 Visual Pipeline Overview

```
START (Query) → DECOMPOSE → SEARCH → CLEAN & CHUNK → EMBED → AGGREGATE 
    → CLUSTER → EVALUATE → [USER HALT] → EXTRACT FACTS → GATEKEEPER 
    → GENERATE REPORT → END (Final Report)
    
BONUS: Stateless RAG for Follow-up Q&A
```

---

## 🔄 Detailed Node Breakdown

### 1️⃣ **DECOMPOSITION** - Query Breaking
**File:** `nodes/decomposition.py`

| Aspect | Details |
|--------|---------|
| **Function** | `decompose_query(query: str, llm_client)` |
| **Purpose** | Break broad query into focused sub-queries |
| **Input** | Original user query (string) |
| **Output** | List of 3-4 sub-queries |
| **Process** | 1. LLM analyzes original query<br/>2. Generates 3-4 distinct search angles<br/>3. Ensures no overlap<br/>4. Validates searchability |
| **Key Features** | ✓ No duplicate sub-queries<br/>✓ Each covers different perspective<br/>✓ Directly searchable terms<br/>✓ Temperature: 0.3 (low randomness) |
| **LLM Model** | Fast LLM (Groq)<br/>~50 input tokens<br/>~100 output tokens |

**Example:**
- **Query:** "What are the effects of social media on teen mental health?"
- **Sub-queries:**
  1. "Social media effects on adolescent anxiety and depression"
  2. "Screen time impact on teenage sleep and attention span"
  3. "Social comparison and self-esteem on TikTok Instagram platforms"

---

### 2️⃣ **SEARCH** - Web Search Execution
**File:** `nodes/search.py`

| Aspect | Details |
|--------|---------|
| **Function** | `search(sub_queries: List[str])` |
| **Purpose** | Execute parallel web searches per sub-query |
| **Input** | List of 3-4 approved sub-queries |
| **Output** | List of Document objects with raw HTML |
| **Process** | 1. Create Tavily API client<br/>2. For each sub-query: execute async search<br/>3. Fetch raw HTML from URLs<br/>4. Combine results |
| **Key Features** | ✓ Concurrent execution (ThreadPoolExecutor)<br/>✓ Raw HTML preservation<br/>✓ URL validation<br/>✓ Fallback handling for failed fetches |
| **API** | Tavily Search API |
| **Output per doc** | `{url, title, raw_html, metadata}` |

---

### 3️⃣ **CLEAN & CHUNK** - Text Processing
**File:** `nodes/clean_chunk.py`

| Aspect | Details |
|--------|---------|
| **Function** | `clean_and_chunk(documents: List[Document])` |
| **Purpose** | Extract readable text and split into chunks |
| **Input** | Raw HTML documents |
| **Output** | List of Chunk objects (100-200 words each) |
| **Process** | 1. Strip scripts, styles, metadata<br/>2. Use trafilatura for main content<br/>3. Split by paragraphs<br/>4. Merge small paragraphs (<80 words)<br/>5. Split large paragraphs (>250 words) |
| **Key Features** | ✓ Preserves document structure<br/>✓ Maintains source URL per chunk<br/>✓ Consistent chunk sizing<br/>✓ No empty chunks |
| **Tools** | trafilatura, BeautifulSoup4 |
| **Target** | 100-200 words per chunk<br/>Typically 15-30 chunks per document |

---

### 4️⃣ **EMBEDDING** - Vector Generation
**File:** `nodes/embed.py`

| Aspect | Details |
|--------|---------|
| **Function** | `embed_chunks(chunks: List[Chunk], embedding_model)` |
| **Purpose** | Convert text chunks to semantic vectors |
| **Input** | List of text chunks |
| **Output** | Same chunks with 384-dim embeddings |
| **Process** | 1. Load sentence-transformers model<br/>2. For each chunk: encode text to vector<br/>3. Normalize vectors<br/>4. Attach to chunk object |
| **Key Features** | ✓ 384-dimensional vectors<br/>✓ Normalized (L2)<br/>✓ No NaN values<br/>✓ Semantic similarity preserved |
| **Model** | sentence-transformers<br/>(all-MiniLM-L6-v2) |
| **Dimension** | 384 |

---

### 5️⃣ **AGGREGATION** - Document-Level Vectors
**File:** `nodes/aggregate.py`

| Aspect | Details |
|--------|---------|
| **Function** | `aggregate(chunks: List[Chunk], documents: List[Document])` |
| **Purpose** | Create document-level embeddings from chunks |
| **Input** | Chunks with embeddings + original documents |
| **Output** | Documents with aggregated embeddings |
| **Process** | 1. Group chunks by document ID<br/>2. For each doc: compute mean of chunk vectors<br/>3. Attach aggregated embedding to document<br/>4. Preserve document metadata |
| **Key Features** | ✓ One embedding per document<br/>✓ Accurate grouping<br/>✓ No NaN values<br/>✓ Maintains document links |
| **Math** | `doc_embedding = mean(chunk_embeddings)` |

---

### 6️⃣ **CLUSTERING** - Thematic Grouping
**File:** `nodes/cluster.py`

| Aspect | Details |
|--------|---------|
| **Function** | `cluster_documents(documents: List[Document])` |
| **Purpose** | Group documents by semantic theme |
| **Input** | Documents with embeddings |
| **Output** | Same documents with cluster_id assigned |
| **Process** | 1. Extract all document embeddings<br/>2. Determine optimal k = min(4, num_docs)<br/>3. Run K-Means clustering<br/>4. Assign cluster_id to each document |
| **Key Features** | ✓ Automatic k selection<br/>✓ Theme-based grouping<br/>✓ Cluster centroids computed<br/>✓ Efficient algorithm |
| **Algorithm** | K-Means (sklearn)<br/>k = min(4, n_docs) |
| **Output** | `document.cluster_id: int` |

---

### 7️⃣ **EVALUATION** - Relevance Scoring
**File:** `evals/relevance_scorer.py`

| Aspect | Details |
|--------|---------|
| **Function** | `calculate_relevance(documents, query_embedding)` |
| **Purpose** | Score each document's relevance to original query |
| **Input** | Documents with embeddings + query embedding |
| **Output** | Same documents with relevance scores (0-1) |
| **Process** | 1. Embed original query<br/>2. Compute cosine similarity between:<br/>   - Query embedding vs each document embedding<br/>3. Normalize scores to [0, 1]<br/>4. Attach to document |
| **Key Features** | ✓ Scores bounded [0, 1]<br/>✓ Cosine similarity metric<br/>✓ Identifies best sources<br/>✓ Enables user curation |
| **Tools** | numpy, sklearn cosine_similarity |
| **Output** | `document.relevance_score: float (0-1)` |

---

### 🛑 **USER HALT #1** - Source Curation
**Location:** `app/main.py` (Streamlit UI)

| Aspect | Details |
|--------|---------|
| **Purpose** | Allow user to review and select sources |
| **Input** | Documents with relevance scores + cluster assignments |
| **Output** | `approved_chunks[]` - filtered list of chunks |
| **UI Features** | ✓ Show relevance scores<br/>✓ Show cluster themes<br/>✓ Show source density<br/>✓ Select/deselect sources<br/>✓ Preview content |
| **Workflow** | 1. User sees all sources ranked by relevance<br/>2. User can remove low-quality/irrelevant<br/>3. User approves source subset<br/>4. Pipeline continues with approved sources |

---

### 8️⃣ **FACT EXTRACTION** - Bullet Point Generation
**File:** `nodes/extract_facts.py`

| Aspect | Details |
|--------|---------|
| **Function** | `extract_facts(chunks, documents, llm_client, top_k=3)` |
| **Purpose** | Extract factual bullet points from approved chunks |
| **Input** | Approved chunks + documents with metadata |
| **Output** | List of Fact objects (bullets with sources) |

#### **Per-Cluster Process:**
1. **Generate Cluster Heading** (LLM)
   - Summarizes cluster theme in 1-2 sentences
   - Uses heading for context in extraction prompt

2. **Retrieve Top-K Chunks** (Semantic Search)
   - Embed sub-queries
   - Cosine similarity with chunk embeddings
   - Select top-8 chunks
   - Dedup if similarity > 0.85
   - Keep top-6 final

3. **LLM Fact Extraction**
   - Pass chunks + cluster heading + sub-queries to Fast LLM
   - Request JSON output with bullets
   - Each bullet: max 2 sentences
   - Include source URL per bullet

4. **Tiered Semantic Evaluation** (Hotfix)
   - **Tier 1** (Strongest): mechanism keywords + causal clarity + score ≥ 0.45
   - **Tier 2** (Medium): behavioral keywords + system linkage + score ≥ 0.35
   - **Tier 3** (Contextual): trusted domain + query overlap ≥ 2 tokens + (behavioral OR causal) + score ≥ 0.30
   - Adaptive thresholds for non-dense facts (lower by 0.12-0.05)

5. **Cluster-Level Fallback Cascade**
   - If Tier 1 facts exist: use Tier 1 + Tier 2
   - Else if Tier 2 facts exist: use Tier 2 (fallback_mode="tier2_fallback")
   - Else if Tier 3 facts exist: use max 1 Tier 3 fact (fallback_mode="tier3_fallback")
   - Else: mark mechanism_gap=True, skip cluster

6. **Logging & Diagnostics**
   - Per-cluster: tier_1_count, tier_2_count, tier_3_count
   - Per-cluster: fallback_mode, mechanism_gap
   - Score distribution: min, max, avg, histogram (0.0-0.2, 0.2-0.4, etc.)
   - Dropped counters: by_semantics, by_tier3_cap, by_density, by_score, by_source

| **Key Features** | ✓ Tiered acceptance logic<br/>✓ Cluster fallback prevents collapse<br/>✓ Score distribution telemetry<br/>✓ Semantic filtering<br/>✓ Source preservation |
| **LLM Model** | Fast LLM (Groq)<br/>~1500 input tokens<br/>~400 output tokens |
| **Outputs** | `fact: {text, source_url, sub_query, score, tier, confidence}` |

---

### 9️⃣ **GATEKEEPER** - Fact Pruning & Grouping
**File:** `nodes/gatekeeper.py`

| Aspect | Details |
|--------|---------|
| **Function** | `gatekeeper(facts: List[Fact])` |
| **Purpose** | Prune low-evidence queries and group facts |
| **Input** | Raw extracted facts |
| **Output** | Grouped facts dict: `{sub_query: [facts]}` |
| **Logic** | 1. Group facts by sub_query<br/>2. **Zero-Cost Protocol:**<br/>   - 0 facts → mark "No data available"<br/>   - 1 fact → mark "Limited evidence"<br/>   - 2+ facts → keep as-is<br/>3. Prune 0-fact groups<br/>4. Merge single-fact groups<br/>5. Assign confidence levels |
| **Key Features** | ✓ No data loss<br/>✓ Correct grouping<br/>✓ Confidence marking<br/>✓ Handles edge cases |
| **Output** | `{sub_query_1: [fact1, fact2], ...}` |

---

### 🔟 **REPORT GENERATION** - Cohesive Synthesis
**File:** `nodes/generate_report.py`

| Aspect | Details |
|--------|---------|
| **Function** | `generate_report(grouped_facts, llm_client, original_query, sub_queries)` |
| **Purpose** | Write professional, cited report from facts |
| **Input** | Grouped facts + original query + sub-queries |
| **Output** | Markdown report with citations |
| **Process** | 1. Create sections for each sub-query<br/>2. Feed facts to Heavy LLM<br/>3. LLM writes flowing narrative<br/>4. Maintains all citations<br/>5. No hallucination (fact-only)<br/>6. Professional tone |
| **Key Features** | ✓ Structured sections<br/>✓ Cited bullet points<br/>✓ No made-up information<br/>✓ Markdown format<br/>✓ High quality synthesis |
| **LLM Model** | Heavy LLM (Groq)<br/>~1200 input tokens<br/>~800-1200 output tokens |
| **Prompt Rules** | - Use ONLY provided facts<br/>- Preserve all citations<br/>- Professional tone<br/>- Section per sub-query<br/>- No speculation |

---

### 🎁 **BONUS: Stateless RAG** - Follow-Up Q&A
**File:** `nodes/stateless_rag.py`

| Aspect | Details |
|--------|---------|
| **Function** | `answer(query: str, chunks: List[Chunk], llm_client)` |
| **Purpose** | Answer follow-up questions using research chunks |
| **Input** | User follow-up question + original research chunks |
| **Output** | Answer + source citations |
| **Process** | 1. Embed follow-up question<br/>2. Retrieve most relevant chunks (cosine sim)<br/>3. LLM generates answer from chunks<br/>4. Include source URLs<br/>5. Strict fact-grounding (no hallucination) |
| **Key Features** | ✓ Stateless (no conversation memory)<br/>✓ Fact-grounded<br/>✓ Source citations<br/>✓ Independent of main report |
| **Use Case** | User asks "What about X?" after report |

---

## 📋 Data Flow Summary

```
User Query
    ↓
[1] DECOMPOSITION → sub_queries[]
    ↓
[2] SEARCH → documents[] (raw_html)
    ↓
[3] CLEAN & CHUNK → chunks[] (text, 100-200 words)
    ↓
[4] EMBEDDING → chunks[].embedding (384-dim)
    ↓
[5] AGGREGATION → documents[].embedding (mean of chunks)
    ↓
[6] CLUSTERING → documents[].cluster_id (K-Means)
    ↓
[7] EVALUATION → documents[].relevance_score (cosine sim)
    ↓
[🛑 USER HALT - Source Curation]
    ↓
[8] FACT EXTRACTION → facts[] (tiered, with scores)
         ├─ Tier 1: mechanism + causal
         ├─ Tier 2: behavioral + system_linkage
         ├─ Tier 3: trusted_domain + query_overlap
         └─ Cluster fallback cascade
    ↓
[9] GATEKEEPER → grouped_facts{sub_query: [facts]}
    ↓
[10] GENERATE REPORT → Final Report (markdown)
    ↓
Report + Sources
```

---

## 🔧 Configuration & Controls

**File:** `utils/config.py`

| Control | Purpose |
|---------|---------|
| `FACT_SCORE_THRESHOLD_TIER1` | 0.45 - Strong mechanism + causal requirement |
| `FACT_SCORE_THRESHOLD_TIER2` | 0.35 - Behavioral + system linkage bar |
| `FACT_SCORE_THRESHOLD_TIER3` | 0.30 - Contextual facts requirement |
| `MAX_TIER3_PER_CLUSTER` | 1 - Cap contextual facts per cluster |
| `MECHANISM_KEYWORDS` | List of mechanism-related terms |
| `BEHAVIORAL_KEYWORDS` | List of behavioral pattern terms |
| `CAUSAL_KEYWORDS` | List of causal relationship terms |
| `TRUSTED_DOMAINS_TIER3` | Domains trusted for contextual facts |

---

## 🛡️ Guardrails & Validation

**File:** `utils/guardrails.py`

- **Input Validation:** Schema checking on all node inputs
- **Output Validation:** Schema checking on all node outputs
- **Type Safety:** Pydantic models ensure data contracts
- **Error Handling:** Graceful fallbacks on LLM failures
- **Observability:** Structured logging of all operations

---

## 📊 Observability & Logging

**File:** `utils/observability.py`

```python
# Logged events per node:
- node_started: timing, input payload
- node_finished: duration_ms
- node_error: error type, message

# Per extract_facts cluster:
- tier_1_count, tier_2_count, tier_3_count
- fallback_mode: "none" | "tier2_fallback" | "tier3_fallback"
- mechanism_gap: boolean flag
- score_min, score_max, score_avg
- score_histogram: distribution buckets
- dropped_by_*: counters for filtering decisions
```

---

## ✅ Testing Structure

**Location:** `tests/` directory

```
tests/
├── test_nodes/          # Unit tests for each node
│   ├── test_search.py
│   ├── test_extract_facts.py
│   ├── test_extract_facts_hotfix.py
│   └── ...
├── test_schemas/        # Pydantic model validation
├── test_evals/          # Evaluation metrics tests
└── conftest.py          # Shared fixtures & mocks
```

---

## 🚀 Execution Flow in LangGraph

**File:** `app/graph_workflow.py`

```python
# State machine phases:
START → decompose → search → clean_chunk → embed → aggregate 
→ cluster → evaluate → extract_facts → gatekeeper → generate_report → END

# User halts (human-in-loop):
1. After evaluate: Source curation (select/remove documents)

# Validations at each step:
- Input validation before node execution
- Output validation after node execution
- Schema enforcement via Pydantic models
```

---

## 🎯 Key Architectural Principles

1. **Modular Nodes:** Each node has clear input/output contracts
2. **State Preservation:** LangGraph maintains shared WorkflowState
3. **Human-in-Loop:** Pause points for user approval
4. **Tiered Semantics:** Multi-level acceptance for robustness
5. **Observability:** Comprehensive logging for debugging
6. **Type Safety:** Pydantic everywhere
7. **Fallback Cascades:** Graceful degradation when confidence drops
8. **Score Distribution Tracking:** Empirical calibration telemetry

---

## 📖 Usage Example

```python
# From app/main.py
from app.graph_workflow import build_graph
from utils.llm_clients import create_llm_client

# Initialize
llm_client = create_llm_client()
embedding_model = load_embedding_model()
graph = build_graph(llm_client, embedding_model)

# Execute
result = graph.invoke({
    "original_query": "What are effects of social media on teen mental health?",
    "approved_chunks": [],  # User will select after source curation
    "phase": WorkflowPhase.DECOMPOSE
})

print(result["report"])  # Final markdown report
```

---

## 🔍 Troubleshooting Guide

| Issue | Node | Solution |
|-------|------|----------|
| Few/no facts extracted | extract_facts | Check tier thresholds, inspect score distributions in logs |
| Low relevance scores | evaluate | Verify query embedding quality |
| Too many false clusters | cluster | Adjust k selection logic |
| Duplicate chunks | clean_chunk | Check chunking rules (merge/split logic) |
| HTML parsing failures | search | Inspect raw_html, check trafilatura settings |

---

**Generated:** April 2026  
**Version:** 2.0 (with Extraction Recall Recovery Hotfix)
