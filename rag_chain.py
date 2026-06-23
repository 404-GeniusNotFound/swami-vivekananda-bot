from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from operator import itemgetter
import os
import re
import random

# Load environment variables
load_dotenv()

GREETINGS = [
    "Namaste, my child. What is on your mind?",
    "Namaste. What is on your mind, my child?",
    "Namaste, my child. How may I guide you today?",
    "Namaste. What question do you bring to me today, my child?"
]

def is_simple_greeting(text: str) -> bool:
    clean = re.sub(r'[^\w\s]', '', text).strip().lower()
    common_greetings = {
        "hi", "hello", "hey", "namaste", "pranam", "pranaam", "greetings",
        "hi swamiji", "hello swamiji", "namaste swamiji", "pranam swamiji", "pranaam swamiji",
        "good morning", "good afternoon", "good evening", "hi bot", "hello bot"
    }
    return clean in common_greetings

def format_chat_history(messages):
    formatted = []
    for msg in messages:
        role = "Human" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)

def rephrase_question(question: str, chat_history: str, llm) -> str:
    system_prompt = (
        "You are a helpful assistant that rephrases follow-up questions to be standalone questions "
        "using the conversation history. Do NOT answer the question. Only return the rephrased "
        "question. If the question is already a standalone question or if it cannot be contextualized, "
        "return it exactly as it is."
    )
    user_prompt = (
        f"Conversation History:\n{chat_history}\n\n"
        f"Follow-up Question: {question}\n\n"
        f"Standalone Question:"
    )
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        return response.content.strip()
    except Exception as e:
        print(f"Error rephrasing question: {e}")
        return question

class ConversationalRagChain:
    def __init__(self, rag_chain, llm):
        self.rag_chain = rag_chain
        self.llm = llm
        
    def invoke(self, inputs, config=None):
        question = str(inputs.get("question", ""))
        chat_history_raw = inputs.get("chat_history", [])
        
        # 1. Handle simple greetings
        if is_simple_greeting(question):
            return random.choice(GREETINGS)
            
        # 2. Rephrase follow-up questions if history exists
        standalone_question = question
        if chat_history_raw:
            formatted_history = format_chat_history(chat_history_raw)
            standalone_question = rephrase_question(question, formatted_history, self.llm)
            print(f"DEBUG: Rephrased standalone question: '{standalone_question}' (Original: '{question}')")
            
        # 3. Invoke RAG chain with standalone question
        return self.rag_chain.invoke({"question": standalone_question}, config)


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
    
    # Create retriever with k=3 to optimize input token consumption
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
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
        max_tokens=300,  # Enforce output length limit
    )

    # Simplified Persona Prompt with Conciseness instructions
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
    7. **CONCISENESS & CONVERSATIONAL STYLE**: Keep your response brief, direct, and conversational (typically **3 to 5 sentences** / 1-2 short paragraphs). Do not lecture or write long essays. Deliver one strong point or quote and invite the user to delve deeper if they wish.
    
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
    
    return ConversationalRagChain(rag_chain, llm)

