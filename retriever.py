# from retriever import retrieve_chunks

# query = "What is retrieval-augmented generation?"
# top_chunks = retrieve_chunks(query, top_k=5)

# for i, c in enumerate(top_chunks):
#     print(f"[{i+1}] {c['text']}\n--")

import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import numpy as np

# Constants
VECTOR_DIR = "vectorstore"
MODEL_NAME = "all-MiniLM-L6-v2"

# Loading embedding model
model = SentenceTransformer(MODEL_NAME)

# Load FAISS index
index = faiss.read_index(os.path.join(VECTOR_DIR, "index.faiss"))

# Load metadata
with open(os.path.join(VECTOR_DIR, "metadata.pkl"), "rb") as f:
    metadata = pickle.load(f)

def retrieve(query, top_k = 3):
    query_embedding = model.encode([query])
    query_embedding = normalize(query_embedding)
    D, I = index.search(query_embedding, top_k)
    results = []

    for idx in I[0]:
        if idx < len(metadata):
            entry = metadata[idx]
            results.append({
                "source": entry['source'],
                "text": entry['text'],
            })
    
    return results

if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() in ['exit', 'quit']:
            break
        
        results = retrieve(query)
        print("\n Top matching chunks:")
        for i, res in enumerate(results):
            print(f"[{i+1}] From {res['source']}\n{res['text'][:500]}\n--\n")