from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
#from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader


load_dotenv()

os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

huggingface_api_key = os.getenv("HUGGINGFACEHUB_API_KEY")

def storing_chunks():
    urls = [
        "https://www.datacamp.com/tutorial/llama-3-1-rag",
        "https://www.datacamp.com/blog/what-is-retrieval-augmented-generation-rag?utm_cid=23950607746&utm_aid=196253697294&utm_campaign=230119_1-ps-other~dsa-tofu~ai_2-b2c_3-apac_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na&utm_loc=9232621-&utm_mtd=-c&utm_kw=&utm_source=google&utm_medium=paid_search&utm_content=ps-other~apac-en~dsa~tofu~blog~artificial-intelligence&gad_source=1&gad_campaignid=23950607746&gbraid=0AAAAADQ9WsGvbhAMpyBOxWEt2cQl_rCKP&gclid=Cj0KCQjwi8nRBhDhARIsAHZf_pa_TG8NMfQ7cXKEsQRW2QdRj7e6Pm_Cv4b2o-M2HIFfp34lP6OG9WsaAgieEALw_wcB"
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    #split text to chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    #embedding the chunks
    embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-m3",
        huggingfacehub_api_token=huggingface_api_key
    )

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embeddings,
        persist_directory="./chroma_db"

    )


def store_pdf_chunks(folder_path="./data/pdfs"):
    """
    Loads all PDFs from a directory, splits them into chunks, and stores them in Chroma.
    """
    # 1. Load Documents from a directory
    # Ensure this directory exists and contains your PDF files
    loader = PyPDFDirectoryLoader(folder_path)
    docs = loader.load()
    
    print(f"---LOADED {len(docs)} PAGES FROM PDFs---")

    # 2. Split text to chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, # Increased size for better PDF context
        chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs)

    # 3. Embedding the chunks
    embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-m3",
        huggingfacehub_api_token=huggingface_api_key
    )

    # 4. Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("---PDF DATA STORED IN VECTOR DATABASE---")

def get_retriever(state):
    # Check if state is a dict (from LangGraph) or a string (manual call)
    if isinstance(state, dict):
        question = state["question"]
    else:
        question = state # state is the question string
        
    print(f"---RETRIEVING FOR: {question}---")
    
    embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-m3",
        huggingfacehub_api_token=huggingface_api_key
    )

    vectorstore = Chroma(
        collection_name="rag-chroma",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    retriever = vectorstore.as_retriever()
    documents = retriever.invoke(question)
    
    # If it was called by a LangGraph node, return a dict update
    if isinstance(state, dict):
        return {"documents": documents, "question": question}
    # If it was called as a helper function (like in your old rag_generate), return docs
    return documents


def get_all_documents():
    """
    Retrieves metadata for all documents currently in the vector store.
    """
    embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-m3",
        huggingfacehub_api_token=huggingface_api_key
    )
    
    vectorstore = Chroma(
        collection_name="rag-chroma",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    # .get() returns a dict with 'ids', 'metadatas', 'documents', etc.
    results = vectorstore.get()
    
    # Extract unique sources to avoid seeing the same PDF multiple times for each chunk
    if results['metadatas']:
        unique_sources = list(set([m.get('source') for m in results['metadatas'] if m.get('source')]))
        return unique_sources
        
    return []