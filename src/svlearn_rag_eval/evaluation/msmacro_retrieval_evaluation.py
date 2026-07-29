# -------------------------------------------------------------------------------------------
#  Copyright (c) 2024.  SupportVectors AI Lab
#
#  This code is part of the training material, and therefore part of the intellectual property.
#  It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#  Use is limited to the duration and purpose of the training at SupportVectors.
#
#  Author: SupportVectors AI Training
# -------------------------------------------------------------------------------------------
import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import math
import argparse
import json
from pathlib import Path
import os
import torch
from dotenv import load_dotenv

load_dotenv()

def get_device():
    """Determine the best device to use for model inference.
    
    Checks CUDA availability and compatibility, falls back to CPU if needed.
    
    Returns:
        str: Device name ('cuda' or 'cpu')
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return 'cpu'
    
    try:
        # Try to create a simple tensor on CUDA to test compatibility
        test_tensor = torch.zeros(1).cuda()
        del test_tensor
        torch.cuda.empty_cache()
        
        # Get GPU name and capability for logging
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        print(f"CUDA is available. Using GPU: {gpu_name} (compute capability {capability[0]}.{capability[1]})")
        return 'cuda'
    except Exception as e:
        print(f"CUDA is available but incompatible with current PyTorch installation: {e}")
        print("Falling back to CPU.")
        return 'cpu'

def load_gold_standard(queries_file_path, qrels_file_path):
    """Load gold standard evaluation data from msmarco format.
    
    Args:
        queries_file_path: Path to queries.jsonl file with format {"_id": "query_id", "text": "query text", ...}
        qrels_file_path: Path to qrels TSV file with format query-id\tcorpus-id\tscore
    
    Returns:
        List of dicts with:
        - query: the query string to be embedded
        - neighbors: list of [document_id, relevance_score] tuples
                     where higher relevance_score means more important
    """
    # Load queries into a dictionary: query_id -> query_text
    queries = {}
    queries_path = Path(queries_file_path)
    
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                query_id = item["_id"]
                query_text = item["text"]
                queries[query_id] = query_text
    
    # Load qrels into a dictionary: query_id -> list of [doc_id, score]
    qrels = {}
    qrels_path = Path(qrels_file_path)
    
    with open(qrels_path, "r", encoding="utf-8") as f:
        # Skip header line
        next(f)
        for line in f:
            if line.strip():
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    query_id = parts[0]
                    doc_id = int(parts[1])  # Convert to int to match corpus IDs
                    score = int(parts[2])
                    
                    if query_id not in qrels:
                        qrels[query_id] = []
                    qrels[query_id].append([doc_id, score])
    
    # Combine queries and qrels into gold_standard format
    gold_standard = []
    for query_id, query_text in queries.items():
        if query_id in qrels:
            # Only include queries that have relevance judgments
            neighbors = qrels[query_id]
            gold_standard.append({
                "query": query_text,
                "neighbors": neighbors  # List of [doc_id, relevance] tuples
            })
    
    return gold_standard

# Determine and set device
device = get_device()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

def retrieve(query, k=5, client=None, collection_name=None):
    q_emb = model.encode(query, convert_to_numpy=True)
    search_result = client.query_points(
        collection_name=collection_name,
        query=q_emb,
        limit=k
    )
    points = search_result.points
    return [r.id for r in points]

# ---------------------------------------------------------
# Metric functions
# ---------------------------------------------------------
def precision_at_k(retrieved, relevant, k):
    retrieved_set = retrieved[:k]
    return len(set(retrieved_set) & set(relevant)) / k

def recall_at_k(retrieved, relevant, k):
    retrieved_set = retrieved[:k]
    return len(set(retrieved_set) & set(relevant)) / len(relevant)

def mrr_at_k(retrieved, relevant, k):
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0

def average_precision_at_k(retrieved, relevant, k):
    score = 0
    hit_count = 0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            hit_count += 1
            score += hit_count / (i + 1)
    return score / len(relevant)

def dcg(retrieved, relevance_map, k):
    """Calculate DCG using actual relevance scores from relevance_map.
    
    Args:
        retrieved: List of retrieved document IDs
        relevance_map: Dictionary mapping doc_id to relevance score
        k: Number of top documents to consider
    """
    score = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        rel = relevance_map.get(doc_id, 0)
        score += rel / math.log2(i + 2)
    return score

def ndcg_at_k(retrieved, neighbors, k):
    """Calculate nDCG@k using relevance scores from neighbors.
    
    Args:
        retrieved: List of retrieved document IDs
        neighbors: List of [doc_id, relevance_score] tuples, sorted by relevance descending
        k: Number of top documents to consider
    """
    # Create relevance map for quick lookup
    relevance_map = {doc_id: relevance for doc_id, relevance in neighbors}
    
    # Ideal ranking: neighbors already sorted by relevance (descending)
    ideal_doc_ids = [doc_id for doc_id, _ in neighbors[:k]]
    
    # Calculate IDCG using ideal ranking
    idcg = dcg(ideal_doc_ids, relevance_map, k)
    if idcg == 0:
        return 0.0
    
    # Calculate DCG for retrieved documents
    dcg_score = dcg(retrieved, relevance_map, k)
    
    return dcg_score / idcg

# ---------------------------------------------------------
# Evaluate all queries
# ---------------------------------------------------------
def evaluate(gold_standard, k=5, client=None, collection_name=None):
    p_scores = []
    r_scores = []
    mrr_scores = []
    map_scores = []
    ndcg_scores = []

    for item in gold_standard:
        query = item["query"]
        neighbors = item["neighbors"]  # List of [doc_id, relevance] tuples
        
        # Sort neighbors by relevance descending (higher relevance = more important)
        neighbors_sorted = sorted(neighbors, key=lambda x: x[1], reverse=True)
        
        # Extract relevant document IDs for binary metrics (precision, recall, etc.)
        relevant_doc_ids = [doc_id for doc_id, _ in neighbors_sorted]

        retrieved = retrieve(query, k, client=client, collection_name=collection_name)

        p_scores.append(precision_at_k(retrieved, relevant_doc_ids, k))
        r_scores.append(recall_at_k(retrieved, relevant_doc_ids, k))
        mrr_scores.append(mrr_at_k(retrieved, relevant_doc_ids, k))
        map_scores.append(average_precision_at_k(retrieved, relevant_doc_ids, k))
        ndcg_scores.append(ndcg_at_k(retrieved, neighbors_sorted, k))

    return {
        "Precision@k": np.mean(p_scores),
        "Recall@k": np.mean(r_scores),
        "MRR@k": np.mean(mrr_scores),
        "MAP@k": np.mean(map_scores),
        "nDCG@k": np.mean(ndcg_scores)
    }

# ---------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval system metrics")
    parser.add_argument(
        "--k",
        type=int,
        default=-1,
        help="Number of top documents to retrieve and evaluate (default: -1) means do in a loop for k = 1 to 5"
    )
    parser.add_argument(
        "--queries_file",
        type=str,
        default=None,
        help="Path to queries.jsonl file (default: uses BOOTCAMP_ROOT_DIR/data/msmarco/queries.jsonl)"
    )
    parser.add_argument(
        "--qrels_file",
        type=str,
        default=None,
        help="Path to qrels TSV file (default: uses BOOTCAMP_ROOT_DIR/data/msmarco/qrels/dev.tsv)"
    )
    parser.add_argument(
        "--qdrant_url",
        type=str,
        default="http://localhost:6333",
        help="Qdrant server URL, e.g., http://localhost:6333" # http://inference:6333
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="msmacro_rag_eval_collection",
        help="Name of the Qdrant collection to use (default: msmacro_rag_eval_collection)"
    )
    args = parser.parse_args()
    
    # Set default paths if not provided
    if args.queries_file is None or args.qrels_file is None:
        root_dir = os.getenv("BOOTCAMP_ROOT_DIR")
        data_dir = f"{root_dir}/data/msmarco"
        if args.queries_file is None:
            args.queries_file = f"{data_dir}/queries.jsonl"
        if args.qrels_file is None:
            args.qrels_file = f"{data_dir}/qrels/dev.tsv"
    
    # Initialize Qdrant client and collection name from arguments
    client = QdrantClient(
        url=args.qdrant_url,
        timeout=300  # seconds
    )
    collection_name = args.collection_name
    
    # Load gold standard from msmarco format
    gold_standard = load_gold_standard(args.queries_file, args.qrels_file)
    print(f"Loaded {len(gold_standard)} queries with relevance judgments")
    
    if args.k == -1:
        for k in range(1, 6):
            results = evaluate(gold_standard, k, client=client, collection_name=collection_name)
            print("-" * 40)
            print(f"Evaluation Metrics for k = {k}:")
            for metric, value in results.items():
                print(f"{metric}: {value:.4f}")
    else:
        results = evaluate(gold_standard, args.k, client=client, collection_name=collection_name)
        print(f"Evaluation Metrics for k = {args.k}:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
