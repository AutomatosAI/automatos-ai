from huggingface_hub import InferenceClient
import psycopg2
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)

class ContextManager:
    def __init__(self):
        self.client = InferenceClient(model="sentence-transformers/all-MiniLM-L6-v2", token=os.getenv("HUGGINGFACE_TOKEN"))
        self.conn = psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            host="db"
        )
        self.cur = self.conn.cursor()

    def rag_retrieve(self, query: str) -> str:
        try:
            embedding = self.client.feature_extraction(query)
            embedding_str = '[' + ','.join(map(str, embedding[0])) + ']'
            self.cur.execute("SELECT context FROM embeddings ORDER BY embedding <-> %s LIMIT 1;", (embedding_str,))
            result = self.cur.fetchone()
            return result[0] if result else "No context found"
        except Exception as e:
            logging.error(f"RAG error: {str(e)}")
            return "No context retrieved"

    def __del__(self):
        self.conn.close()