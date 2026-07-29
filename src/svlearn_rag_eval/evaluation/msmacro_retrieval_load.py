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
import json
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import os
import torch
from dotenv import load_dotenv
import argparse
load_dotenv()

# Get root directory - use env var if set, otherwise use relative path
root_dir = os.getenv("BOOTCAMP_ROOT_DIR")
if root_dir is None:
    # Use relative path from current working directory (assumes script run from project root)
    root_dir = "."
    
# Path to msmarco corpus.jsonl file
data_dir = f"{root_dir}/data"
corpus_file = f"{data_dir}/msmarco/corpus.jsonl"

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

# Determine and set device
device = get_device()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

dimension = 384  # MiniLM embedding dim

def create_collection(client, collection_name):
    if client.collection_exists(collection_name):
        print(f"Collection already exists: {collection_name}, deleting...")
        client.delete_collection(collection_name)
    print(f"Creating collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
    )
    print(f"Collection created successfully: {collection_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load msmarco corpus into Qdrant collection.")
    parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="Name of the Qdrant collection to use."
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        required=False,
        default="http://localhost:6333",
        help="Qdrant server URL, e.g., http://localhost:6333"
    )
    args = parser.parse_args()
    collection_name = args.collection
    qdrant_url = args.qdrant_url

    client = QdrantClient(
        url=qdrant_url,
        timeout=1000  # seconds
    )

    create_collection(client, collection_name)
    # Load and upsert documents from msmarco corpus.jsonl incrementally
    num_loaded = 0
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                doc = json.loads(line)
                # Use _id from msmarco format (convert to int for Qdrant)
                doc_id = int(doc["_id"])
                text = doc["text"]
                embedding = model.encode(text, convert_to_numpy=True)
                point = PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload={
                        **doc,  # Include all fields from JSON
                        "id": doc_id  # Also include the integer ID in the payload
                    }
                )
                client.upsert(collection_name=collection_name, points=[point])
                num_loaded += 1
                if num_loaded % 1000 == 0:
                    print(f"{num_loaded} documents loaded so far...")
    print(f"Documents loaded successfully: {num_loaded} documents from msmarco/corpus.jsonl")
