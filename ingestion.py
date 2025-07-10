import json 
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader # For PDF parsing
from docx import Document # For DOCX parsing
from datetime import datetime
from dotenv import load_dotenv


load_dotenv()

DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR","Documents")
DATA_DIR = os.getenv("DATA_DIR","data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

CHUNK_SIZE = 500 # characters
CHUNK_OVERLAP = 100 

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path,'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"


    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text


def extract_text_from_docx(docx_path):
    text = ""
    try:
        doc = Document(docx_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:  
        print(f"Error reading DOCX {docx_path}: {e}")
    return text

def chunk_text(text,chunk_size,chunk_overlap):
    chunks = []
    if not text:
        return chunks
    
    start_idx = 0
    while start_idx < len(text):
        end_idx = min(start_idx + chunk_size, len(text))
        chunk = text[start_idx:end_idx]
        chunks.append(chunk)
        start_idx += chunk_size - chunk_overlap
        if start_idx >= len(text) - chunk_overlap and end_idx == len(text):
            break

    return chunks    

def ingest_documents():

    
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"Using embedding model: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        print(f"Error loading embedding model {EMBEDDING_MODEL_NAME}: {e}")
        return  

    all_chunks = []
    all_metadata = []
    chunk_id_counter = 0

    for filename in os.listdir(DOCUMENTS_DIR):
        file_path = os.path.join(DOCUMENTS_DIR, filename)
        if filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif filename.lower().endswith('.docx'):
            text = extract_text_from_docx(file_path)
        else:
            print(f"Unsupported file type: {filename}. Only PDF and DOCX files are supported.")
            continue

        if not text:
            print(f"no text is extracted from {filename}")
            continue

        chunks = chunk_text(text,CHUNK_SIZE,CHUNK_OVERLAP)
        print(f"Extracted {len(chunks)} chunks from {filename}")

        for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    "chunk_id": chunk_id_counter,
                    "text": chunk,
                    "source_file": filename,
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
                chunk_id_counter += 1


    if not all_chunks:
        print("No chunks to process. Exiting.")
        return
    
    print(f"Total chunks to process: {len(all_chunks)}")
    chunk_embeddings = model.encode(all_chunks,show_progress_bar=True,convert_to_tensor=True).cpu().numpy()

    print(f"Embeddings generated for all chunks. {len(chunk_embeddings)}")

    # Ensure embeddings are float32 for FAISS
    chunk_embeddings = chunk_embeddings.astype('float32')
    
    dimension = chunk_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension) # Using L2 distance for similarity
    faiss_index.add(chunk_embeddings)
    print(f"FAISS index built with {faiss_index.ntotal} vectors.")

    # Save FAISS index and metadata
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")
    print(f"Metadata saved to {METADATA_PATH}")
    print("Ingestion complete.")


if __name__ == "__main__":
    ingest_documents()

