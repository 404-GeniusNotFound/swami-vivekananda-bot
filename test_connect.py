import os
import requests
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

print("--- DIAGNOSTIC START ---")
print(f"API Key present: {bool(api_key)}")

# Test 1: Raw REST API
print("\n[Test 1] Raw REST API (curl equivalent)...")
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={api_key}"
data = {
    "contents": [{"parts": [{"text": "Say hello"}]}]
}
try:
    start = time.time()
    response = requests.post(url, json=data, timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text[:100]}...")
    print(f"Time: {time.time() - start:.2f}s")
except Exception as e:
    print(f"REST API Failed: {e}")

# Test 2: LangChain
print("\n[Test 2] LangChain Invocation...")
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        max_retries=0, 
        timeout=10,
        transport="rest"
    )
    start = time.time()
    res = llm.invoke("Say hello")
    print(f"LangChain Result: {res.content}")
    print(f"Time: {time.time() - start:.2f}s")
except Exception as e:
    print(f"LangChain Failed: {e}")

print("--- DIAGNOSTIC END ---")
