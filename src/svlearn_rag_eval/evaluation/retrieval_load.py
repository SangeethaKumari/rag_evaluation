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
from dotenv import load_dotenv

load_dotenv()

root_dir = os.getenv("BOOTCAMP_ROOT_DIR")
# Path to animals.jsonl file
data_dir = f"{root_dir}/data"
animals_file = f"{data_dir}/animals.jsonl"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client = QdrantClient(host="localhost", port=6333)

collection_name = "rag_eval_collection"
dimension = 384  # MiniLM embedding dim

def create_collection():
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
    create_collection()
    # Load documents from animals.jsonl
    documents = []
    texts = []
    
    with open(animals_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if line.strip():  # Skip empty lines
                doc = json.loads(line)
                documents.append({
                    "id": line_num,
                    "text": doc["text"],
                    "metadata": doc  # All fields from JSON
                })
                texts.append(doc["text"])
    
    # Generate embeddings for all texts
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    # Create points with incremental IDs and all metadata
    points = [
        PointStruct(
            id=doc["id"],
            vector=emb,
            payload={
                **doc["metadata"],  # Include all fields from JSON
                "id": doc["id"]  # Also include the incremental ID
            }
        )
        for doc, emb in zip(documents, embeddings)
    ]
    
    client.upsert(collection_name=collection_name, points=points)
    print(f"Documents loaded successfully: {len(documents)} documents from animals.jsonl")
