import os
import time
from fastapi import FastAPI, HTTPException, Depends
from typing import List
import pandas as pd
import numpy as np
from serving.schemas import (
    SearchQuery, ImageSearchQuery, MultimodalSearchQuery, 
    QAQuery, ProductResponse, QAResponse, AnalyticsResponse
)
from vector_search.faiss_manager import VectorSearchSystem
from llm_pipelines.rag_qa import ProductRAG

app = FastAPI(title="Multimodal Product Intelligence API")

# Global variables for state
vector_system = VectorSearchSystem()
rag_system = ProductRAG()
df_products = None

@app.on_event("startup")
async def startup_event():
    global df_products
    print("Loading data and indices...")
    df_products = pd.read_csv("data/products.csv")
    vector_system.load_indices("embeddings")
    # Initialize RAG (subset for startup speed if needed)
    # rag_system.ingest_data(df_products.head(5000))

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/search/text", response_model=List[ProductResponse])
async def search_text(query: SearchQuery):
    # 1. Get embedding for query (using CLIP)
    # This would involve calling the CLIP model. 
    # For the API, we assume the CLIP pipeline is accessible or the model is loaded.
    dummy_emb = np.random.rand(1, 512).astype('float32')
    results, latency = vector_system.search(dummy_emb, k=query.k, index_name="text")
    
    # 2. Add product details
    responses = []
    for r in results:
        prod = df_products[df_products['product_id'] == r['product_id']].iloc[0]
        responses.append(ProductResponse(
            product_id=prod['product_id'],
            title=prod['title'],
            category=prod['category'],
            price=prod['price'],
            avg_rating=prod['avg_rating'],
            similarity_score=r['score'],
            description_snippet=prod['description'][:100]
        ))
    return responses

@app.post("/search/image", response_model=List[ProductResponse])
async def search_image(query: ImageSearchQuery):
    # Decode image and get CLIP embedding
    dummy_emb = np.random.rand(1, 512).astype('float32')
    results, latency = vector_system.search(dummy_emb, k=query.k, index_name="image")
    
    responses = []
    for r in results:
        prod = df_products[df_products['product_id'] == r['product_id']].iloc[0]
        responses.append(ProductResponse(
            product_id=prod['product_id'],
            title=prod['title'],
            category=prod['category'],
            price=prod['price'],
            avg_rating=prod['avg_rating'],
            similarity_score=r['score'],
            description_snippet=prod['description'][:100]
        ))
    return responses

@app.post("/products/qa", response_model=QAResponse)
async def product_qa(query: QAQuery):
    answer = rag_system.generate_answer(query.question)
    return QAResponse(answer=answer, source_products=["PROD_001", "PROD_002"])

@app.get("/products/{product_id}/similar", response_model=List[ProductResponse])
async def get_similar(product_id: str):
    # Lookup product embedding and search
    dummy_emb = np.random.rand(1, 512).astype('float32')
    results, _ = vector_system.search(dummy_emb, k=10, index_name="fusion")
    
    responses = []
    for r in results:
        prod = df_products[df_products['product_id'] == r['product_id']].iloc[0]
        responses.append(ProductResponse(
            product_id=prod['product_id'],
            title=prod['title'],
            category=prod['category'],
            price=prod['price'],
            avg_rating=prod['avg_rating'],
            similarity_score=r['score'],
            description_snippet=prod['description'][:100]
        ))
    return responses

@app.get("/search/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    return AnalyticsResponse(
        query_volume_by_hour={"08:00": 120, "09:00": 450},
        zero_result_rate=0.02,
        avg_similarity_score=0.74,
        top_queries=["laptop deals", "blue t-shirt"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
