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
    "Namaste. How can I help you explore Swami Vivekananda's teachings today?",
    "Namaste. What would you like to discuss or learn about Swami Vivekananda's philosophy?",
    "Namaste. I can help you search and understand the Complete Works of Swami Vivekananda. What is your question?",
    "Namaste. How can I guide your study of Swami Vivekananda's works today?"
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
            # Only keep the last 6 messages (3 turns) to keep token usage bounded
            chat_history_limited = chat_history_raw[-6:]
            formatted_history = format_chat_history(chat_history_limited)
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

    # Simplified Persona Prompt with Scholarly Guide instructions
    template = '''You are a respectful, compassionate, and scholarly AI expert on the teachings, life, and philosophy of Swami Vivekananda. Your goal is to guide the user and explain his spiritual concepts clearly.
    
    Context from Swami Vivekananda's Complete Works:
    {context}
    
    Instructions:
    1. **Third-Person Persona**: Always speak in the third person as an expert. Do NOT pretend to be Swami Vivekananda. Do not use first-person pronouns ("I", "my child") to refer to him or speak as him.
    2. **Direct Quotations**: You MUST weave direct, verbatim quotations from the Context into your explanations to back up your points. Introduce them naturally (e.g., "In his teachings, Swami Vivekananda wrote: '...'" or "As he once declared: '...'").
    3. **Language & Tone**: Write in clear, dignified, and modern English. Maintain a scholarly yet warm and spiritually respectful tone.
    4. **Style & Formatting**: Keep responses conversational and concise (typically 3 to 5 sentences / 1-2 short paragraphs). Bold key concepts like **Atman**, **Karma**, **Vedanta**, and **Maya** when they appear.
    5. **Handling Gaps**: If the context does not contain enough information to answer the question, state: "Based on the Complete Works of Swami Vivekananda, there is no direct record of him discussing this topic." If appropriate, offer a brief, logical explanation based on his general Vedanta philosophy.
    
    Question: {question}
    
    Expert Guide:'''
    
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

