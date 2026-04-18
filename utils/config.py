import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
FAST_LLM_MODEL = os.getenv("FAST_LLM_MODEL", "llama-3.1-8b-instant")
HEAVY_LLM_MODEL = os.getenv("HEAVY_LLM_MODEL", "llama-3.3-70b-versatile")
