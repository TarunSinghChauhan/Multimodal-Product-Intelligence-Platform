import os
from typing import List, Dict, Any
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from qdrant_client import QdrantClient
import pandas as pd
from sentence_transformers import CrossEncoder

# Configuration
DATA_DIR = "data"
API_KEY = os.getenv("ANTHROPIC_API_KEY", "your_key_here")

class ProductRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.client = QdrantClient(path="qdrant_db")
        self.llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", anthropic_api_key=API_KEY)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.vector_store = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def ingest_data(self, df_products):
        print("Ingesting data into Qdrant...")
        texts = (df_products['title'] + " | " + df_products['description']).tolist()
        metadatas = df_products.to_dict('records')
        
        self.vector_store = Qdrant.from_texts(
            texts,
            self.embeddings,
            metadatas=metadatas,
            path="qdrant_db",
            collection_name="products"
        )

    def classify_query(self, query: str) -> str:
        """Determines if query is a search or a question."""
        prompt = f"Classify this query as 'SEARCH' (user looking for products) or 'QUESTION' (user asking about product details): '{query}'. Reply with only one word."
        response = self.llm.invoke(prompt).content.strip()
        return response

    def retrieve_and_rerank(self, query: str, k=10) -> List[Dict]:
        # 1. Semantic Retrieval
        docs = self.vector_store.similarity_search(query, k=k)
        
        # 2. Re-ranking with Cross-Encoder
        if not docs:
            return []
            
        doc_texts = [d.page_content for d in docs]
        pairs = [[query, text] for text in doc_texts]
        scores = self.cross_encoder.predict(pairs)
        
        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d[0] for d in ranked_docs[:3]] # Top 3

    def generate_answer(self, query: str) -> str:
        relevant_docs = self.retrieve_and_rerank(query)
        context = "\n\n".join([f"Product: {d.metadata['title']}\nDescription: {d.page_content}\nPrice: ${d.metadata['price']}\nRating: {d.metadata['avg_rating']}" for d in relevant_docs])
        
        prompt = f"""Use the following product context to answer the user's question accurately. 
        If the answer is not in the context, say you don't know. Do not hallucinate.
        Context:
        {context}
        
        Question: {query}
        Answer:"""
        
        response = self.llm.invoke(prompt)
        return response.content

    def chat(self, query: str) -> str:
        # Full LangChain chain with memory could be used here
        return self.generate_answer(query)

if __name__ == "__main__":
    # Example usage
    pass
