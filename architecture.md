# System Architecture & Sequence Diagrams

This document describes the high-level architecture and execution flows of the **RAG Evaluation System**.

---

## 1. System Architecture Diagram

The system comprises three main layers:
1. **Data & Storage Layer**: Source document corpus (`animals.jsonl`), evaluation dataset (`eval_dataset.jsonl`), and the vector database (Qdrant `rag_eval_collection`).
2. **Embedding & Ingestion Layer**: Semantic embedding generator using `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")` and Qdrant ingestion client (`retrieval_load.py`).
3. **Retrieval Evaluation & RAG Pipeline Layer**:
   - **Evaluation Subsystem** (`retrieval_evaluation.py`): Vector search retriever, gold-standard sorting, and multi-metric calculator (`Precision@k`, `Recall@k`, `MRR@k`, `MAP@k`, `nDCG@k`).
   - **RAG Subsystem** (`rag.py`): In-memory document similarity matching and response synthesis via OpenAI or Ollama LLMs.

```mermaid
flowchart TB
    subgraph Storage ["Data Storage & Vector Database"]
        JSONL["Document Corpus\n(data/animals.jsonl)"]
        GOLD["Gold Standard Dataset\n(data/eval_dataset.jsonl)"]
        QDRANT[("Qdrant Vector DB\nCollection: rag_eval_collection\nDimension: 384 | Distance: Cosine")]
    end

    subgraph Embedding ["Embedding Engine"]
        ST_MODEL["SentenceTransformer\n(sentence-transformers/all-MiniLM-L6-v2)"]
    end

    subgraph Ingestion ["Ingestion Subsystem (retrieval_load.py)"]
        PARSER["JSONL Parser"]
        LOAD_SCRIPT["Document Loader"]
    end

    subgraph Evaluation ["Retrieval Evaluation Subsystem (retrieval_evaluation.py)"]
        EVAL_SCRIPT["Evaluation Controller"]
        QUERY_EMB["Query Encoder"]
        RETRIEVER["Qdrant Point Retriever"]
        
        subgraph Metrics ["Metrics Engine"]
            P_K["Precision@k"]
            R_K["Recall@k"]
            MRR_K["MRR@k"]
            MAP_K["MAP@k"]
            NDCG_K["nDCG@k"]
        end
    end

    subgraph RAG_Pipeline ["RAG Generation Pipeline (rag.py)"]
        RAG_CLASS["RAG Engine"]
        COSINE_SEARCH["NumPy Cosine Similarity Search"]
        LLM_PROVIDER["LLM Provider\n(ChatOpenAI / OllamaLLM)"]
    end

    %% Ingestion Flow
    JSONL --> PARSER
    PARSER --> LOAD_SCRIPT
    LOAD_SCRIPT --> ST_MODEL
    ST_MODEL -- "Vectors (384-dim) + Payloads" --> QDRANT

    %% Evaluation Flow
    GOLD --> EVAL_SCRIPT
    EVAL_SCRIPT --> QUERY_EMB
    QUERY_EMB --> ST_MODEL
    QUERY_EMB -- "Query Vector" --> RETRIEVER
    RETRIEVER <--> QDRANT
    RETRIEVER -- "Top-k Point IDs" --> EVAL_SCRIPT
    EVAL_SCRIPT --> Metrics

    %% RAG Flow
    RAG_CLASS --> COSINE_SEARCH
    COSINE_SEARCH --> RAG_CLASS
    RAG_CLASS --> LLM_PROVIDER
```

---

## 2. Sequence Diagrams

### 2.1 Document Ingestion & Indexing Flow (`retrieval_load.py`)

This workflow reads documents from `data/animals.jsonl`, initializes or resets the Qdrant vector collection, computes 384-dimensional embeddings, and upserts points with text payloads and line-number document IDs.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Script as retrieval_load.py
    participant File as data/animals.jsonl
    participant Model as SentenceTransformer
    participant Qdrant as Qdrant Server (localhost:6333)

    User->>Script: Execute ingestion script
    Script->>Qdrant: Check if collection 'rag_eval_collection' exists
    alt Collection exists
        Qdrant-->>Script: Exists
        Script->>Qdrant: delete_collection('rag_eval_collection')
    end
    Script->>Qdrant: create_collection(name='rag_eval_collection', size=384, distance=Cosine)
    Qdrant-->>Script: Collection created

    Script->>File: Read document lines
    File-->>Script: JSON documents (text, author, category)
    Script->>Script: Extract text & assign 1-indexed document IDs

    Script->>Model: model.encode(texts, convert_to_numpy=True)
    Model-->>Script: 384-dimensional numpy vectors

    Script->>Script: Construct PointStruct list (id, vector, payload)
    Script->>Qdrant: client.upsert(collection_name, points)
    Qdrant-->>Script: Upsert confirmation
    Script-->>User: Report successful document loading
```

---

### 2.2 Retrieval Evaluation Flow (`retrieval_evaluation.py`)

This workflow evaluates retrieval performance against `data/eval_dataset.jsonl` across various values of \(k\) (1 to 5). It measures Precision@k, Recall@k, MRR@k, MAP@k, and nDCG@k.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant EvalScript as retrieval_evaluation.py
    participant GoldFile as data/eval_dataset.jsonl
    participant Model as SentenceTransformer
    participant Qdrant as Qdrant Server
    participant Metrics as Metric Evaluators

    User->>EvalScript: Run evaluation (e.g. --k 5 or loop k=1..5)
    EvalScript->>GoldFile: load_gold_standard(eval_file)
    GoldFile-->>EvalScript: Queries & ground truth tuples [[doc_id, relevance], ...]

    loop For each query in gold standard
        EvalScript->>EvalScript: Sort ground truth neighbors by relevance descending
        EvalScript->>Model: model.encode(query, convert_to_numpy=True)
        Model-->>EvalScript: Query vector (384-dim)
        
        EvalScript->>Qdrant: query_points(collection='rag_eval_collection', query=q_emb, limit=k)
        Qdrant-->>EvalScript: Top-k ScoredPoints (Point IDs)

        EvalScript->>Metrics: precision_at_k(retrieved, relevant_ids, k)
        EvalScript->>Metrics: recall_at_k(retrieved, relevant_ids, k)
        EvalScript->>Metrics: mrr_at_k(retrieved, relevant_ids, k)
        EvalScript->>Metrics: average_precision_at_k(retrieved, relevant_ids, k)
        EvalScript->>Metrics: ndcg_at_k(retrieved, neighbors_sorted, k)
        Metrics-->>EvalScript: Individual query scores
    end

    EvalScript->>EvalScript: Compute arithmetic mean for each metric
    EvalScript-->>User: Output aggregated metrics table
```

---

### 2.3 RAG Answer Generation Flow (`rag.py`)

This workflow demonstrates an end-to-end RAG pipeline using in-memory vector dot product matching and LLM response generation via LangChain with OpenAI or Ollama.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant RAG as RAG Pipeline (rag.py)
    participant Embedder as Embeddings Engine (OpenAI / HuggingFace)
    participant LLM as LLM Provider (ChatOpenAI / OllamaLLM)

    User->>RAG: load_documents(documents)
    RAG->>Embedder: embed_documents(documents)
    Embedder-->>RAG: Matrix of document embeddings
    
    User->>RAG: get_most_relevant_docs(query)
    RAG->>Embedder: embed_query(query)
    Embedder-->>RAG: Query vector
    RAG->>RAG: Compute Cosine Similarities (np.dot / norms)
    RAG->>RAG: Find argmax index
    RAG-->>User: Return most relevant document
    
    User->>RAG: generate_answer(query, relevant_doc)
    RAG->>LLM: invoke([SystemMessage, HumanMessage])
    LLM-->>RAG: Generated response string
    RAG-->>User: Return synthesized answer
```
