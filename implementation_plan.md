# System Architecture and Sequence Diagrams Implementation Plan

This implementation plan outlines the system architecture diagram and sequence diagrams for the RAG Evaluation System project.

## Proposed Diagrams Overview

### 1. High-Level Architecture Diagram
Visualizes the overall system structure, covering:
- **Data Layer**: Raw JSONL corpora (`animals.jsonl`) and Gold Standard Evaluation data (`eval_dataset.jsonl`).
- **Embedding & Vector DB Layer**: `SentenceTransformer` (`all-MiniLM-L6-v2`) and `Qdrant` vector database (`rag_eval_collection`).
- **Evaluation Engine Layer**: Top-K retrieval, sorting gold standard neighbors, and evaluating retrieval quality metrics (`Precision@k`, `Recall@k`, `MRR@k`, `MAP@k`, `nDCG@k`).
- **RAG Generation Layer (`rag.py`)**: Document similarity search and LLM answer synthesis (`ChatOpenAI` / `OllamaLLM`).

```mermaid
flowchart TB
    subgraph Storage ["Data Storage & Vector Database"]
        JSONL["Document Corpus\n(animals.jsonl)"]
        GOLD["Gold Standard Dataset\n(eval_dataset.jsonl)"]
        QDRANT[("Qdrant Vector DB\n(Collection: rag_eval_collection\nDim: 384, Distance: Cosine)")]
    end

    subgraph Embedding ["Embedding Engine"]
        ST_MODEL["SentenceTransformer\n(all-MiniLM-L6-v2)"]
    end

    subgraph Ingestion ["1. Indexing & Ingestion Subsystem (retrieval_load.py)"]
        LOAD_SCRIPT["Ingestion Script"]
        PARSER["JSONL Parser & Doc Extractor"]
    end

    subgraph Evaluation ["2. Retrieval Evaluation Subsystem (retrieval_evaluation.py)"]
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

    subgraph RAG_Pipeline ["3. Basic RAG Generation Pipeline (rag.py)"]
        RAG_CLASS["RAG Engine"]
        COSINE_SEARCH["NumPy Vector Cosine Similarity"]
        LLM_PROVIDER["LLM Provider\n(ChatOpenAI / OllamaLLM)"]
    end

    %% Flow connections
    JSONL --> PARSER
    PARSER --> LOAD_SCRIPT
    LOAD_SCRIPT --> ST_MODEL
    ST_MODEL -- Vectors + Metadata Payloads --> QDRANT

    GOLD --> EVAL_SCRIPT
    EVAL_SCRIPT --> QUERY_EMB
    QUERY_EMB --> ST_MODEL
    QUERY_EMB -- Embedded Query --> RETRIEVER
    RETRIEVER <--> QDRANT
    RETRIEVER -- Top-k Doc IDs --> EVAL_SCRIPT
    EVAL_SCRIPT --> Metrics

    RAG_CLASS --> COSINE_SEARCH
    COSINE_SEARCH --> RAG_CLASS
    RAG_CLASS --> LLM_PROVIDER
```

---

### 2. Sequence Diagrams

#### Diagram A: Document Ingestion & Indexing Sequence (`retrieval_load.py`)
```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Script as retrieval_load.py
    participant File as animals.jsonl
    participant Model as SentenceTransformer (all-MiniLM-L6-v2)
    participant Qdrant as Qdrant Server (localhost:6333)

    User->>Script: Execute loading script
    Script->>Qdrant: Check if collection 'rag_eval_collection' exists
    alt Collection exists
        Qdrant-->>Script: Exists
        Script->>Qdrant: delete_collection('rag_eval_collection')
    end
    Script->>Qdrant: create_collection(name='rag_eval_collection', size=384, distance=Cosine)
    Qdrant-->>Script: Collection created confirmation

    Script->>File: Read document lines
    File-->>Script: JSON documents (text, author, category)
    Script->>Script: Extract texts & assign incremental document IDs (line numbers)

    Script->>Model: model.encode(texts, convert_to_numpy=True)
    Model-->>Script: 384-dimensional vector embeddings

    Script->>Script: Construct PointStruct list (id, vector, payload)
    Script->>Qdrant: client.upsert(collection_name, points)
    Qdrant-->>Script: Upsert completion status
    Script-->>User: Report loading success (e.g., 100 documents loaded)
```

#### Diagram B: Retrieval Evaluation Sequence (`retrieval_evaluation.py`)
```mermaid
sequenceDiagram
    autonumber
    actor User
    participant EvalScript as retrieval_evaluation.py
    participant GoldFile as eval_dataset.jsonl
    participant Model as SentenceTransformer (all-MiniLM-L6-v2)
    participant Qdrant as Qdrant Server
    participant Metrics as Metric Functions

    User->>EvalScript: Run evaluation (e.g., --k 5 or loop k=1..5)
    EvalScript->>GoldFile: load_gold_standard(eval_file)
    GoldFile-->>EvalScript: List of queries & ground truth neighbors [[doc_id, relevance], ...]

    loop For each query item in gold standard
        EvalScript->>EvalScript: Sort neighbors by relevance score descending
        EvalScript->>Model: model.encode(query)
        Model-->>EvalScript: Query Vector (384-dim)
        
        EvalScript->>Qdrant: query_points(collection='rag_eval_collection', query=q_emb, limit=k)
        Qdrant-->>EvalScript: Top-k ScoredPoints (Point IDs)

        EvalScript->>Metrics: precision_at_k(retrieved, relevant_ids, k)
        EvalScript->>Metrics: recall_at_k(retrieved, relevant_ids, k)
        EvalScript->>Metrics: mrr_at_k(retrieved, relevant_ids, k)
        EvalScript->>Metrics: average_precision_at_k(retrieved, relevant_ids, k)
        EvalScript->>Metrics: ndcg_at_k(retrieved, neighbors_sorted, k)
        Metrics-->>EvalScript: Scores for query
    end

    EvalScript->>EvalScript: Compute mean scores across all queries
    EvalScript-->>User: Display evaluation report table (Precision, Recall, MRR, MAP, nDCG)
```

#### Diagram C: End-to-End RAG Generation Sequence (`rag.py`)
```mermaid
sequenceDiagram
    autonumber
    actor User
    participant RAG as RAG Pipeline (rag.py)
    participant Embedder as Embeddings Engine (OpenAI / HuggingFace)
    participant LLM as LLM Model (ChatOpenAI / OllamaLLM)

    User->>RAG: load_documents(documents)
    RAG->>Embedder: embed_documents(documents)
    Embedder-->>RAG: Matrix of document embeddings
    
    User->>RAG: get_most_relevant_docs(query)
    RAG->>Embedder: embed_query(query)
    Embedder-->>RAG: Query vector
    RAG->>RAG: Calculate Cosine Similarities (np.dot / norm product)
    RAG->>RAG: Find max similarity index
    RAG-->>User: Return top relevant document
    
    User->>RAG: generate_answer(query, relevant_doc)
    RAG->>LLM: invoke([system_prompt, human_prompt])
    LLM-->>RAG: AI response string
    RAG-->>User: Final synthesized answer
```

---

## Target Documents

- [architecture.md](architecture.md): Contains system architecture and sequence diagrams.
- [README.md](README.md): Links to `architecture.md` with overview diagram embedded.
