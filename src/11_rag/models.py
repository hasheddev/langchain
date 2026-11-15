from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

gemini_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_version="v1beta",
)
qwen_model = ChatGroq(model="qwen/qwen3.8-27b")
