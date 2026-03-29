# ============================================================
# 1. Imports & Environment Setup
# ============================================================

import os
from dotenv import load_dotenv
os.chdir("../")
from typing import List
from langchain_core.documents import Document

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ============================================================
# 2. Configuration
# ============================================================

DATA_DIR = "data"
INDEX_NAME = "medibot"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 20
EMBEDDING_DIM = 384


# ============================================================
# 3. Document Loading
# ============================================================

def load_pdf_files(data_dir: str) -> List[Document]:
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        show_progress=True,
        loader_cls=PyPDFLoader,
    )
    return loader.load()


# ============================================================
# 4. Metadata Filtering
# ============================================================

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs = []

    for doc in docs:
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": doc.metadata.get("source", "unknown")
                },
            )
        )

    return minimal_docs


# ============================================================
# 5. Text Splitting
# ============================================================

def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


# ============================================================
# 6. Embedding Model
# ============================================================

def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# ============================================================
# 7. Pinecone Initialization
# ============================================================

def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [idx["name"] for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Created Pinecone index: {INDEX_NAME}")
    else:
        print(f"Pinecone index already exists: {INDEX_NAME}")

    return pc


# ============================================================
# 8. Vector Store Operations
# ============================================================

def ingest_documents(chunks: List[Document], embeddings):
    print("Uploading documents to Pinecone...")
    
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
    )

    print("Upload complete.")


def load_vector_store(embeddings):
    return PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )


# ============================================================
# 9. Query Function
# ============================================================

def query_vector_store(query: str, k: int = 3):
    embeddings = load_embeddings()
    store = load_vector_store(embeddings)

    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    retrieved_docs = retriever.invoke(query)

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(retrieved_docs)} documents:\n")

    for i, doc in enumerate(retrieved_docs, 1):
        print(f"Result {i}")
        print(f"Source: {doc.metadata.get('source')}")
        print(doc.page_content[:300])
        print("-" * 40)


# ============================================================
# 10. Full Ingestion Pipeline
# ============================================================

def run_ingestion():
    print("Loading PDFs...")
    docs = load_pdf_files(DATA_DIR)

    print("Filtering metadata...")
    docs = filter_to_minimal_docs(docs)

    print("Splitting documents...")
    chunks = split_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    embeddings = load_embeddings()

    init_pinecone()
    ingest_documents(chunks, embeddings)


# ============================================================
# 11. Main Execution
# ============================================================

if __name__ == "__main__":
    
    # Step 1: Run once to upload documents
    run_ingestion()

    # Step 2: Query
    query_vector_store("What is Acne?")