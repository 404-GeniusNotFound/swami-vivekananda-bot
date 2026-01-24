import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Paths - assuming this script is running from inside vivekananda_bot folder OR handling relative paths correctly
# Let's use relative to this file to be safe
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Complete Works of Swami Vivekananda.pdf")
DB_PATH = os.path.join(BASE_DIR, "chroma_db")

def ingest():
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        # convert to absolute to show clearly
        print(f"Absolute path checked: {os.path.abspath(DATA_PATH)}")
        return

    print(f"Loading PDF from {DATA_PATH}...")
    loader = PyPDFLoader(DATA_PATH)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    print("Creating Embeddings and storing in ChromaDB...")
    # This will download the model if not present
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=DB_PATH
    )
    print(f"Ingestion complete! Database saved to {DB_PATH}")

if __name__ == "__main__":
    ingest()
