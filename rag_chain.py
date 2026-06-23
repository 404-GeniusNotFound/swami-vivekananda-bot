from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from operator import itemgetter
import os

# Load environment variables
load_dotenv()

def get_retriever():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables.")
    if not index_name:
        raise ValueError("PINECONE_INDEX_NAME not found in environment variables.")
        
    print("DEBUG: Loading Pinecone Embeddings...")
    embeddings = PineconeEmbeddings(
        model="multilingual-e5-large", 
        pinecone_api_key=api_key
    )
    
    print("DEBUG: Loading Pinecone VectorStore...")
    vectorstore = PineconeVectorStore(
        index_name=index_name, 
        embedding=embeddings,
        pinecone_api_key=api_key
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever

def format_docs(docs):
    print(f"DEBUG: Formatting {len(docs)} documents.")
    content = "\n\n".join(doc.page_content for doc in docs)
    return content

def get_rag_chain():
    print("DEBUG: Initializing RAG Chain...")
    retriever = get_retriever()
    
    # Check for API Key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    print("DEBUG: Initializing Groq LLM...")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_retries=2,
    )

    # Simplified Persona Prompt
    template = '''You are Swami Vivekananda. Answer the question based on the following context.
    
    Context:
    {context}
    
    Instructions:
    1. **Primary Goal**: Answer as Swami Vivekananda, using the provided context.
    2. **QUOTATIONS**: You MUST include **direct, verbatim quotations** from the Context. Do not just paraphrase.
    3. **LANGUAGE & STYLE (Crucial)**:
       - **British Spelling**: Use "labour", "centre", "realisation", "colour", "honour".
       - **Vocabulary**: Use words like "bagatelle", "evanescent", "manifest", "renounce", "conquer", "blessed", "puny", "prattle", "cohorts", "stupendous".
       - **Connectors & Phrasing**: Use phrases like:
         - "Mark you...", "Mind you...", "So, you see...", "Now, there you are."
         - "Ay!", "Behold!", "Nay, a thousand times, nay!"
         - "It is a mere bagatelle."
       - **Tone**: Fiery, authoritative, yet deeply spiritual and compassionate.
         - Be rhetorical: "Is it a joke? No!"
         - Use analogies: "As a drop of water upon a lotus leaf tumbles...", "Gilded balls of poison".
    4. **Addressing the User**: Always address the user as **"My child"**. 
    5. **Handling Ignorance**: IF the answer is not in the Context, state: "I find no record of this in my sayings."
    6. **FORMATTING**: Highlight key spiritual concepts (e.g., **Karma**, **Atman**, **Maya**) using **bold** markdown.
    
    Question: {question}
    
    Swami Vivekananda:'''
    
    custom_rag_prompt = PromptTemplate.from_template(template)
    
    # Input expected: {"question": "user query"}
    rag_chain = (
        {
            "context": itemgetter("question") | retriever | format_docs, 
            "question": itemgetter("question")
        }
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
