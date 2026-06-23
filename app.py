import streamlit as st
import os
import json
from rag_chain import get_rag_chain
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Swami Vivekananda", page_icon="🕉️", layout="wide")

# Custom CSS for a rich, spiritual theme
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #FF9933 0%, #FFFFFF 50%, #138808 100%);
        background: #FFF5E6; /* Fallback light cream */
        background-image: radial-gradient(#FFD1A4 1px, transparent 1px);
        background-size: 20px 20px;
    }

    /* Titles and Headers */
    h1, h2, h3 {
        font-family: 'Cinzel', serif !important;
        color: #8B4513 !important; /* SaddleBrown */
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Main container styling */
    .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #8B4513;
        background-image: linear-gradient(180deg, #8B4513 0%, #A0522D 100%);
        color: #FFFAF0;
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #FFD700 !important; /* Gold text for sidebar headers */
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] li {
        color: #FFE4B5 !important; /* Moccasin text */
        font-family: 'Lora', serif;
    }

    /* Chat Message Styling */
    .stChatMessage {
        background-color: transparent;
    }
    
    /* User Message Bubble */
    [data-testid="stChatMessage"][data-testid="user"] {
        background-color: rgba(255, 255, 255, 0.6);
        border-left: 5px solid #138808; /* India Green */
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        padding: 15px;
        margin-bottom: 10px;
    }

    /* Assistant (Swami) Message Bubble */
    [data-testid="stChatMessage"][data-testid="assistant"] {
        background-color: rgba(255, 248, 220, 0.8); /* Cornsilk */
        border-left: 5px solid #FF9933; /* Saffron */
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        padding: 15px;
        margin-bottom: 10px;
    }

    /* Text Typography */
    .stMarkdown p {
        font-family: 'Lora', serif !important;
        font-size: 1.15rem !important;
        line-height: 1.6 !important;
        color: #2F2F2F !important;
    }

    /* Input Box */
    .stChatInputContainer textarea {
        background-color: #FFFFFF !important;
        border: 2px solid #CD853F !important; /* Peru */
        border-radius: 25px !important;
        font-family: 'Lora', serif !important;
    }
    
    /* Spinner/Loading Text Styling */
    [data-testid="stSpinner"] {
        border-color: #8B4513 !important;
    }
    [data-testid="stSpinner"] > div {
        border-top-color: #8B4513 !important;
    }
    [data-testid="stSpinner"] p, 
    [data-testid="stSpinner"] span, 
    [data-testid="stSpinner"] div {
        color: #8B4513 !important;
        font-family: 'Cinzel', serif !important;
        font-weight: bold;
    }
    
    /* Footer/Caption styling */
    .caption {
        font-family: 'Cinzel', serif;
        color: #8B4513;
        text-align: center;
        font-size: 0.9rem;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar with context
with st.sidebar:
    st.title("Swami Vivekananda")
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/0b/Swami_Vivekananda-1893-09-signed.jpg", caption="Chicago, 1893")
    st.markdown("""
    ---
    **"Arise, awake, and stop not till the goal is reached."**
    
    This bot is trained on the **Complete Works of Swami Vivekananda**. 
    
    It adopts his persona to guide you on:
    - **Vedanta & Philosophy**
    - **Yoga & Meditation**
    - **Strength & Character**
    - **The Nature of the Soul**
    """)
    st.markdown("---")
    st.text("Om Tat Sat.")
    
    st.markdown("---")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Main Header
col1, col2 = st.columns([1, 5])
with col1:
    st.image("assets/swamiji_avatar.png",  width=100) # Use the fixed avatar
with col2:
    st.title("The Voice of Vedanta")
    st.markdown('<p class="caption">"Strength is Life, Weakness is Death."</p>', unsafe_allow_html=True)

# Initialize chat history (Session State only)
if "messages" not in st.session_state:
    st.session_state.messages = []
    
    # Add an initial greeting from Swamiji if history is empty
    greeting = {
        "role": "assistant", 
        "content": "Namaste! What is on your mind, my child?"
    }
    st.session_state.messages.append(greeting)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="assets/swamiji_avatar.png" if message["role"] == "assistant" else "👤"):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question to Swamiji...", max_chars=500):
    # Convert input to string just in case
    prompt = str(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="assets/swamiji_avatar.png"):
        with st.spinner("Meditating on the question..."):
            try:
                # Direct call without caching to avoid connection pool issues
                chain = get_rag_chain()
                response = chain.invoke({
                    "question": prompt,
                    "chat_history": st.session_state.messages[:-1]
                })
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")
