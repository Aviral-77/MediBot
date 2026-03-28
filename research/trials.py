import os, json
os.chdir("../")

#-------------------------Load environment variables-----------------------------------------
from dotenv import load_dotenv
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#------------------------------------------------------------------`


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
from langchain.schema import Document 

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
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import SpacyEmbeddings

# def download_embeddings():
#     model_name = "sentence-transformers/all-MiniLM-L6-v2"
#     embeddings = HuggingFaceEmbeddings(
#         model_name=model_name
#     )
#     return embeddings
def download_embeddings():
    return SpacyEmbeddings(model_name="en_core_web_md")
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
# -------------------------------------------------------------------------------------

extracted_data = load_pdf_files("data")     #LangChain return in Document format which has page_content and metadata
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
# query = embedding.embed_query("Hello World")
# print(query)
vector_db = create_chroma_db(texts_chunk, embedding)
vector_db.persist()
print("Vector store created and persisted to ./chroma_db")

# Define your medical query
user_query = "What is Acne?"

# Search the database for the top 3 most relevant chunks
docs = vector_db.similarity_search(user_query, k=3)

print("\n--- Search Results ---")
for i, doc in enumerate(docs):
    print(f"Result {i+1}:")
    print(f"Source: {doc.metadata['source']}")
    print(f"Content: {doc.page_content[:200]}...") # Print first 200 chars
    print("-" * 30)
# print(filtered_docs)


