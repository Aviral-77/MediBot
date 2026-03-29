import os, json
os.chdir("../")

# -------------------------Load environment variables-----------------------------------------
from dotenv import load_dotenv
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ------------------------------------------------------------------


# -----------------------Load document-------------------------------------------
def load_pdf_files(data):
    loader = DirectoryLoader(
        data, 
        glob="*.pdf", 
        show_progress=True, 
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents

# ----------------------------------------------------------------------------------

# -----------------------Filter data------------------------------------------------
from typing import List
from langchain_core.documents import Document 

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        minimal_doc = Document(
            page_content=doc.page_content,
            metadata={"source": src}
        )
        minimal_docs.append(minimal_doc)
    return minimal_docs
# ------------------------------------------------------------------------------------

# -----------------------Split the document into smaller chunks-----------------------
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=20,
        )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk


# -------------------------------------------------------------------------------------

# -----------------------Create embeddings-----------------------
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import SpacyEmbeddings

def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings
# def download_embeddings():
#     return SpacyEmbeddings(model_name="en_core_web_md")
# -------------------------------------------------------------------------------------

# -----------------------Create vector db (Chroma)-----------------------
from langchain_community.vectorstores import Chroma

def create_chroma_db(documents, embeddings):
    # This automatically saves to the 'db' directory
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    return vectorstore

# -----------------------Create vector db (Chroma)-----------------------
from pinecone import Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)
print(pc)

from pinecone import ServerlessSpec

index_name = "medibot"
try:
    if index_name not in pc.list_indexes():
        pc.create_index(
            name=index_name,
            dimension=384,  # Dimension of the embedding vectors
            metric="cosine",  # Similarity metric
            spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Auto-scaling configuration
        )
except Exception as e:
    print(f"Index creation note: {e}")

from langchain_community.vectorstores import Pinecone as PineconeVectorStore
def create_pinecone_db(texts_chunk, embeddings):
    vectorstore = PineconeVectorStore.from_documents(
        documents=texts_chunk,
        embedding=embeddings,
        index_name=index_name
    )
    return vectorstore



# -------------------------------------------------------------------------------------


extracted_data = load_pdf_files("data")     #LangChain returns in Document format which has page_content and metadata
# print(len(extracted_data))
# with open("extracted_data.txt", "w", encoding="utf-8") as f:
#     for doc in extracted_data:
#         # doc.page_content contains the actual text from the PDF
#         f.write(doc.page_content + "\n" + "="*20 + "\n")
minimal_docs = filter_to_minimal_docs(extracted_data)
texts_chunk = text_split(minimal_docs)
# print(texts_chunk)
print(f"Number of text chunks: {len(texts_chunk)}")
embedding = download_embeddings()
# create_pinecone_db(texts_chunk, embedding)
from langchain_pinecone import PineconeVectorStore
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)
# -----------------------------------------------------------------------------------
#To add more documents in the future, you can use the following code:
# dswith = Document(
#     page_content="dswithbappy is a youtube channel that provides tutorials on various topics.",
#     metadata={"source": "Youtube"}
# )
# docsearch.add_documents(documents=[dswith])
# -----------------------------------------------------------------------------------

# query = embedding.embed_query("Hello World")
# print(query)
# vector_db = create_chroma_db(texts_chunk, embedding)
# vector_db.persist()
# print("Vector store created and persisted to ./chroma_db")

# Define your medical query
user_query = "What is Acne?"

# Search the database for the top 3 most relevant chunks
# docs = vector_db.similarity_search(user_query, k=3)

# print("\n--- Search Results ---")
# for i, doc in enumerate(docs):
#     print(f"Result {i+1}:")
#     print(f"Source: {doc.metadata['source']}")
#     print(f"Content: {doc.page_content[:200]}...") # Print first 200 chars
#     print("-" * 30)
# print(filtered_docs)


