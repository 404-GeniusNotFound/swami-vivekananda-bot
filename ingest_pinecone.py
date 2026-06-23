import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Complete Works of Swami Vivekananda.pdf")

def ingest_to_pinecone():
    # 1. Check credentials
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables.")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME not found in environment variables.")
        
    print(f"Connecting to Pinecone index: {index_name}")
    
    # 2. Check and create index using Pinecone client
    pc = Pinecone(api_key=api_key)
    
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if index_name in existing_indexes:
        print(f"Index '{index_name}' already exists. Deleting it to start a fresh ingestion...")
        pc.delete_index(index_name)
        print("Waiting for index deletion...")
        while index_name in [idx.name for idx in pc.list_indexes()]:
            time.sleep(2)
        print("Index deleted.")
        
    print(f"Creating serverless index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1024,  # multilingual-e5-large is 1024-dimensional
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Waiting for index to initialize...")
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    print(f"Index '{index_name}' created successfully!")

    # 3. Load PDF
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"PDF not found at {DATA_PATH}")
        
    print(f"Loading PDF from {DATA_PATH}...")
    loader = PyPDFLoader(DATA_PATH)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")

    # 4. Split PDF into chunks
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")

    # 5. Initialize Pinecone Embeddings
    print("Initializing PineconeEmbeddings (multilingual-e5-large)...")
    embeddings = PineconeEmbeddings(
        model="multilingual-e5-large", 
        pinecone_api_key=api_key
    )

    # 6. Upload documents in batches
    print("Uploading documents to Pinecone vector store...")
    # Using batches to avoid payload size/timeout issues with large uploads
    batch_size = 100
    vectorstore = None
    
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        print(f"Uploading batch {i // batch_size + 1}/{(len(splits) - 1) // batch_size + 1} ({len(batch)} chunks)...")
        
        # Retry with exponential backoff on rate limit errors
        retries = 5
        backoff = 15
        while retries > 0:
            try:
                if vectorstore is None:
                    vectorstore = PineconeVectorStore.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        index_name=index_name,
                        pinecone_api_key=api_key
                    )
                else:
                    vectorstore.add_documents(batch)
                break  # Success, break retry loop
            except Exception as e:
                print(f"Error uploading batch: {e}")
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e) or "Too Many Requests" in str(e):
                    print(f"Rate limit hit. Sleeping for {backoff} seconds before retrying...")
                    time.sleep(backoff)
                    backoff *= 2
                    retries -= 1
                else:
                    raise e
        else:
            raise RuntimeError("Failed to upload batch after maximum retries.")
            
        # Add a proactive static delay to stay under the 250k token/min limit
        time.sleep(5)
            
    print(f"Ingestion complete! All chunks uploaded successfully to Pinecone index '{index_name}'.")

if __name__ == "__main__":
    ingest_to_pinecone()
