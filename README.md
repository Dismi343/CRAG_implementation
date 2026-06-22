# Corrective Retrieval-Augmented Generation (CRAG) Project

This project implements a Corrective Retrieval-Augmented Generation (CRAG) system, consisting of a FastAPI backend and a Streamlit frontend. 

## Architecture Overview

The system follows the CRAG logic:
1. **Retrieve**: Fetch relevant documents from a local vector store (ChromaDB).
2. **Grade**: Evaluate the relevance of the retrieved documents to the user's question.
3. **Decide**: 
    - If documents are relevant, proceed to **Generate**.
    - If documents are not relevant, **Rewrite** the question and perform a **Web Search**.
4. **Generate**: Synthesize an answer using an LLM (via OpenRouter) based on the best available context.

## Project Structure

```text
.
├── Crag_backend/               # FastAPI Backend
│   ├── main.py                 # API endpoints
│   ├── connect_function_nodes.py # LangGraph workflow definition
│   ├── define_nodes.py         # Node functions for LangGraph
│   ├── nodes/                  # Individual node logic (RAG, Rewriter, Eval, Search)
│   ├── schema/                 # State definitions for LangGraph
│   ├── chroma_db/              # Local vector database
│   └── requirements.txt        # Backend dependencies
├── RAG_frontend_streamlit/     # Streamlit Frontend
│   ├── app.py                  # Main application UI
│   └── requirements.txt        # Frontend dependencies
└── README.md                   # Project documentation
```

## Features

- **Dynamic Retrieval**: Uses LangGraph to manage a stateful RAG workflow.
- **PDF Support**: Upload PDF documents directly from the UI those are then stored in vector database and also local temp file.
- **Web Search Integration**: Uses Tavily Search API for external information retrieval.
- **Vector Store**: store data in ChromaDB with Hugging Face embeddings (`BAAI/bge-m3`) using api calls.
- **LLM Integration**: Uses OpenRouter to access gpt-oss-120b to answer the user questions which comes with the context provided by the CRAG.

## Setup Instructions

### Prerequisites
- Python 3.9+
- API Keys for:
    - [Hugging Face](https://huggingface.co/settings/tokens) (for embeddings)
    - [Tavily](https://tavily.com/) (for web search)
    - [OpenRouter](https://openrouter.ai/) (for LLM generation)

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd Crag_backend
   ```
2. Create a `.env` file and add your API keys:
   ```env
   HUGGINGFACEHUB_API_KEY=your_huggingface_key
   TAVILY_API_KEY=your_tavily_key
   OPENROUTER_API_KEY=your_openrouter_key
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd RAG_frontend_streamlit
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Use the **Upload** section in the Streamlit UI to add PDF documents.
2. Ask questions in the **Chat** interface.
3. The system will decide whether to answer from your documents or search the web if relevant information isn't found locally.
