from fastapi import FastAPI, UploadFile, File

from test import test_ms
from nodes.rag_chain import rag_generate
from nodes.vector_store import storing_chunks
from connect_function_nodes import rag_app
from schema.question_state import QuestionRequest
from nodes.vector_store import store_pdf_chunks, get_all_documents
import shutil
import os
app = FastAPI()


os.environ["USER_AGENT"] = "MyRAGAgent/1.0"

@app.get("/")
async def root():
    storing_chunks()
    return rag_generate("What is machine learning?")
@app.get("/test")
async def test():
    return test_ms()


@app.post("/upload-pdfs")
async def upload_pdf(files: list[UploadFile] = File(...)):
    # Create temp directory if it doesn't exist
    os.makedirs("./temp_pdfs", exist_ok=True)
    
    saved_files = []
    for file in files:
        file_path = f"./temp_pdfs/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file.filename)
    
    # Process the uploaded files into the DB
    store_pdf_chunks(folder_path="./temp_pdfs")
    
    return {"message": "Files uploaded and processed successfully", "files": saved_files}


@app.get("/ask")
async def ask_rag(request: QuestionRequest):
    inputs = {"question": request.question}
    output = rag_app.invoke(inputs)
    return {
        "question": output["question"],
        "answer": output["generation"]
    }

@app.get("/list-docs")
async def list_docs():
    docs = get_all_documents()
    return {
        "count": len(docs),
        "documents": docs
    }

#app.include_router(router)