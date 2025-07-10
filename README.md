## lexi.sg-rag-backend-test
This repository provides a FastAPI-based backend service implementing a Retrieval-Augmented Generation (RAG) system for legal queries. It processes legal documents, builds a searchable vector index, and retrieves relevant citations to answer natural language questions.

## Watch the demo video here:
```
https://drive.google.com/file/d/1cUqNp4goBFbSo5zDwabMW0eoI5v9jXyN/view?usp=sharing
```
## Objective
The primary objective is to build a backend service that accepts a natural language legal query and returns a generated answer along with a list of relevant citations from original documents.

## Features
-Document Ingestion: Reads text from PDF and DOCX files, chunks them, and embeds them.

-Vector Database: Uses FAISS (Facebook AI Similarity Search) as a local vector store to efficiently search for relevant document chunks.

-Embedding Model: Utilizes a free, open-source sentence-transformers model (all-MiniLM-L6-v2) for generating embeddings of text chunks and queries.

## RAG Pipeline:

Receives a natural language legal query.

Embeds the query.

Retrieves the most semantically similar document chunks from the FAISS index.

LLM Answer Generation: Integrates with the Gemini API (gemini-2.0-flash) to generate answers based on the retrieved context chunks and the user's query. The LLM is prompted to answer only from the provided snippets.

Citation Generation: Returns relevant snippets from the retrieved document chunks as citations, along with their source file names.

API Endpoint: Provides a POST /query endpoint for interacting with the RAG system.

Metadata Management: Stores metadata (original chunk text, source filename, chunk ID) alongside embeddings in the vector store for citation retrieval.

## Technologies Used
Python 3.x

FastAPI: Modern, fast (high-performance) web framework for building APIs.

Uvicorn: ASGI server to run the FastAPI application.

Sentence-Transformers: For generating text embeddings.

FAISS (Facebook AI Similarity Search): For efficient similarity search on vector embeddings.

PyPDF2: For extracting text from PDF documents.

python-docx: For extracting text from DOCX documents.

python-dotenv: For managing environment variables.

NumPy: For numerical operations, especially with embeddings.

Pydantic: For data validation and serialization (integrated with FastAPI).

Requests: For making HTTP calls to the Gemini API.

## Setup Instructions
Prerequisites
Before you begin, ensure you have the following installed:

Python 3.7+: python.org

Git: git-scm.com

Google Gemini API Key: Obtain one from Google AI Studio.

1. Clone the Repository
```
git clone https://github.com/YOUR_GITHUB_USERNAME/lexi.sg-rag-backend-test.git
cd lexi.sg-rag-backend-test
```
2. Virtual Environment Setup

It's highly recommended to use a virtual environment to manage dependencies:
```
python -m venv venv
```

On Windows (Command Prompt):
```
venv\Scripts\activate.bat
```
On Linux/macOS (Bash/Zsh):
```
source venv/bin/activate
```
3. Install Dependencies
Once your virtual environment is active, install the required Python packages:
```
pip install -r requirements.txt
```
4. Prepare Input Documents
Create a documents/ directory in the root of the repository. Place your legal documents (PDFs and DOCX files) inside this directory.

For demonstration, you can create documents/sample_doc_1.docx and documents/sample_doc_2.pdf with some placeholder legal text. Ensure these documents contain content relevant to the example queries and expected citations. For instance, sample_doc_1.docx should contain phrases like:

"Use of a vehicle in a public place without a permit is a fundamental statutory infraction. The said situations cannot be equated with absence of licence or a fake licence or a licence for different kind of vehicle, or, for that matter, violation of a condition of carrying more number of passengers."

"Therefore, the tribunal as well as the High Court had directed that the insurer shall be entitled to recover the same from the owner and the driver."

5. Configure Gemini API Key
Create a file named .env in the root directory of your project (the same directory as main.py). Add your Gemini API key to this file:
```
GEMINI_API_KEY=YOUR_ACTUAL_GEMINI_API_KEY_HERE
```
Important: Replace YOUR_ACTUAL_GEMINI_API_KEY_HERE with the API key you obtained from Google AI Studio. Do not commit this file to your public repository.

6. Ingest Documents and Build Vector Store
Run the ingest.py script to process your documents. This will extract text, chunk it, embed the chunks, and build/save the FAISS index and associated metadata into the data/ directory.
```
python ingest.py
```
Note: The first time you run this, sentence-transformers will download the embedding model, which requires an internet connection.

7. Run the FastAPI Application
Start the FastAPI backend service using Uvicorn:
```
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```
The FastAPI app will start on ```http://0.0.0.0:5000/.``` You should see output indicating the model and FAISS index have loaded. The --reload flag is useful for development as it restarts the server on code changes.

You can access the interactive API documentation (Swagger UI) at ```http://127.0.0.1:5000/docs``` (or http://0.0.0.0:5000/docs).

How to Test the API
You can test the /query endpoint using tools like curl, Postman, Insomnia, or directly from the FastAPI interactive documentation (Swagger UI).

Endpoint Details:
URL: http://127.0.0.1:5000/query

Method: POST

Content-Type: application/json

Example Input:
```
{
  "query": "Is an insurance company liable to pay compensation if a transport vehicle involved in an accident was being used without a valid permit?"
}
```
Example curl Command:
```
curl -X POST -H "Content-Type: application/json" \
     -d '{"query": "Is an insurance company liable to pay compensation if a transport vehicle involved in an accident was being used without a valid permit?"}' \
     http://127.0.0.1:5000/query
```
Example Expected Output:
```
{
    "answer": "Based on the provided document snippets, the insurance company's liability to pay compensation when a transport vehicle involved in an accident was being used without a valid permit is addressed. The document states that using a vehicle in a public place without a permit is a fundamental statutory infraction. However, if the accident was caused by unforeseen circumstances unrelated to the driver's lack of a proper license, the insurer may not be able to avoid liability.\n",
    "citation": [
        {
            "text": " the Appellants :- Abhishek Atrey, Advocate. \nFor the Respondents :- Amit Kumar Singh, Advocate. \nMotor Vehicles Act, 1988 Sections 166, 66 and 149 Accident - No permit - Liability to pay compensation - Vehicle at time of accident did not have permit - Use of vehicle in public place without permit is fundamental statutory infraction - Said situations cannot be equated with absence of licence or fake licence or licence for different kind of vehicle, or, for that matter, violation of condition of ",
            "source": "Amrit Paul Singh v. TATA AIG (SC NO ROUTE Permit insurance Co. Recover from Owner).docx"
        },
        {
            "text": " the contract of insurance as also the provisions of the Act by consciously allowing any person to drive a vehicle who did not have a valid driving licence. \n15. The Court held that if, on facts, it is found that the accident was caused solely because of some other unforeseen or intervening causes like mechanical failures and similar other causes having no nexus with the driver not possessing the requisite type of licence, the insurer will not be allowed to avoid its liability merely for technic",
            "source": "Amrit Paul Singh v. TATA AIG (SC NO ROUTE Permit insurance Co. Recover from Owner).docx"
        }
    ]
}
```


<img width="1887" height="1011" alt="Image" src="https://github.com/user-attachments/assets/7c8c30c3-75c1-40f7-9017-77b09f917943" />

<img width="1915" height="1136" alt="Image" src="https://github.com/user-attachments/assets/f2871543-e95d-4f60-a032-522d4967e873" />

<img width="1901" height="1026" alt="Image" src="https://github.com/user-attachments/assets/2a6bde49-7a55-4576-a642-dad97ec59ca2" />



Author
Utsav Modi
modiutsav2003@gmail.com
