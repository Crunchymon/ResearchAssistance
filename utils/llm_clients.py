from groq import Groq
from sentence_transformers import SentenceTransformer
from utils import config

def get_groq_client():
    return Groq(api_key=config.GROQ_API_KEY)

def get_embedding_model():
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)
