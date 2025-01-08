import os
import fitz # PyMuPDF
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# constants
PDF_DIR = "data/sample_papers"
CHUNK_DIR = "chunks"
VECTOR_DIR = "vectorstore"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500

# Loading embedding model
model = SentenceTransformer(MODEL_NAME)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Chunk text into smaller pieces."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks

def embed_chunks(chunks):
    """Embed text chunks using the SentenceTransformer model."""
    embeddings = model.encode(chunks, show_progress_bar=True)
    return normalize(embeddings)

def save_faiss_index(embeddings, metadata):
    """Save the FAISS index to a file."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(VECTOR_DIR, "index.faiss"))
    with open(os.path.join(VECTOR_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

if __name__ == "__main__":
    all_chunks = []
    metadata = []

    for filename in os.listdir(PDF_DIR):
        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_DIR, filename)
        print(f"Processing {filename}...")

        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}_chunk_{i}"
            all_chunks.append(chunk)
            metadata.append({"id": chunk_id, "source": filename, "text": chunk})
            
    print(f"Total chunks {len(all_chunks)}")
    embeddings = embed_chunks(all_chunks)
    save_faiss_index(embeddings, metadata)
    print("Embeddin gand indexing complete")