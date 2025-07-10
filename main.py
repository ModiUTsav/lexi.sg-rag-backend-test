import json
import os
from fastapi import FastAPI, HTTPException,Request,status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import requests



load_dotenv()


DATA_DIR = os.getenv("DATA_DIR","data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(DATA_DIR,"metadata.json")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") # Keep blank for Canvas auto-provisioning
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
print(f"Using Gemini API URL: {GEMINI_API_KEY}")

model: SentenceTransformer = None
faiss_index: faiss.IndexFlatL2 = None
metadata: list = []

app = FastAPI()


class QueryRequest(BaseModel):
    query: str

class Citation(BaseModel):
    text: str
    source: str

class QueryResponse(BaseModel):
    answer:str
    citation: list[Citation]



# helper function to call llm

async def genrate_answer_with_llm(query:str,context_chunks:list[str])->str:


    context_str = "\n".join([f"Document Snippet:\n{chunk}" for chunk in context_chunks])
    
    prompt = (
        f"You are a legal assistant. Answer the following legal question based ONLY on the provided document snippets. "
        f"If the answer cannot be found in the snippets, state that you cannot answer based on the provided information.\n\n"
        f"Legal Question: {query}\n\n"
        f"Document Snippets:\n{context_str}\n\n"
        f"Answer:"
    )

    payload = {
        "contents":[
            {
                "role":"user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.1, # Keep temperature low for factual/legal answers
            "maxOutputTokens": 500 # Limit output length
        }
            
        
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        result = response.json()
        
        # --- DEBUGGING LOGS FOR LLM RESPONSE ---
        print(f"Gemini API Response Status: {response.status_code}")
        print(f"Gemini API Raw Result: {json.dumps(result, indent=2)}")
        # --- END DEBUGGING LOGS ---

        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
            if generated_text: # Ensure the text is not empty
                return generated_text
            else:
                print("LLM generated empty text.")
                return "The LLM generated an empty response."
        else:
            print(f"LLM response structure unexpected or missing content.")
            return "Could not generate an answer: LLM response structure was unexpected."
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return f"Failed to generate an answer due to an API error: {e}"
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Gemini API: {e}")
        return f"Failed to generate an answer: Invalid JSON response from LLM API."
    except Exception as e:
        print(f"Unexpected error during LLM generation: {e}")
        return f"An unexpected error occurred during answer generation: {e}"


@app.on_event("startup")
async def startup_event():
    global model, faiss_index,metadata

    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load embedding model: {e}")    
        print(f"Failed to load embedding model: {e}")


    try:
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print("FAISS index and metadata loaded successfully.")
    except FileNotFoundError:
        print(f"Warning: FAISS index or metadata not found. Please run ingest.py first.")
        faiss_index = None
        metadata = []
    except Exception as e:
        print(f"Error loading FAISS index or metadata: {e}")
        faiss_index = None
        metadata = []


@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_endpoint(request: QueryRequest):
    User_query = request.query 


    if faiss_index is None or model is None or not metadata:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG system not initialized. Please run ingest.py first or check model/index loading errors."
        )

    print(f"Received query: {User_query}")    
    
    try:



        query_embedding = model.encode(User_query, convert_to_tensor=True).cpu().numpy().astype('float32').reshape(1, -1)
        print("Query embedding shape:", query_embedding.shape)
        k = 2 # Number of top relevant chunks to retrieve
        distances, indices = faiss_index.search(query_embedding, k)

        retrieved_citations_data = []
        context_chunks = []

        for i,idx in enumerate(indices[0]):
            if idx < len(metadata):
                chunk_metadata = metadata[idx]
                retrieved_citations_data.append(Citation(
                    text = chunk_metadata['text'],
                    source= chunk_metadata['source_file']
                ))
                context_chunks.append(chunk_metadata['text'])
            else:
                print(f"Index {idx} out of bounds for metadata length {len(metadata)}. Skipping this index.")


        genrated_answer = await genrate_answer_with_llm(User_query,context_chunks)

        response_payload = QueryResponse(
            answer=genrated_answer,
            citation=retrieved_citations_data
        )
        print("Query processed successfully.")
        return response_payload
        
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing query: {e}")
    
@app.get("/")
async def read_root():
    return {"message": "Lexi.sg RAG Backend is running. Use /query endpoint for legal queries."}






