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

def load_gold_standard(eval_file_path):
    """Load gold standard evaluation data from JSONL file.
    
    Each line should have:
    - query: the query string to be embedded
    - neighbors: list of [document_id, relevance_score] tuples
                 where higher relevance_score means more important
    """
    gold_standard = []
    eval_path = Path(eval_file_path)
    
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                gold_standard.append({
                    "query": item["query"],
                    "neighbors": item["neighbors"]  # List of [doc_id, relevance] tuples
                })
    
    return gold_standard

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client = QdrantClient(host="localhost", port=6333)

collection_name = "rag_eval_collection"

def retrieve(query, k=5):
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
def evaluate(gold_standard, k=5):
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

        retrieved = retrieve(query, k)

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
        "--eval_file",
        type=str,
        default="data/eval_dataset.jsonl",
        help="Path to evaluation dataset JSONL file (default: data/eval_dataset.jsonl)"
    )
    args = parser.parse_args()
    
    # Load gold standard from eval_dataset.jsonl
    gold_standard = load_gold_standard(args.eval_file)
    
    if args.k == -1:
        for k in range(1, 6):
            results = evaluate(gold_standard, k)
            print("-" * 40)
            print(f"Evaluation Metrics for k = {k}:")
            for metric, value in results.items():
                print(f"{metric}: {value:.4f}")
    else:
        results = evaluate(gold_standard, args.k)
        print(f"Evaluation Metrics for k = {args.k}:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
