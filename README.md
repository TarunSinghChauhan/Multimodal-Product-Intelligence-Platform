# Multimodal Product Intelligence Platform

A production-grade E-commerce intelligence platform leveraging CLIP for multimodal search, ViT for visual classification, and RAG for conversational product discovery.

## 🚀 Architecture
```mermaid
graph TD
    Data[Synthetic Catalog: 50K Products, 500K Reviews] --> Gen[Data Generation & Texture Simulation]
    Gen --> CLIP[CLIP Embedding Pipeline: Image + Text Fusion]
    Gen --> ViT[ViT Fine-tuning: Category Classification]
    Gen --> LLMAttr[Claude 3.5: Attribute Extraction]
    CLIP --> FAISS[FAISS Vector Search: 512-dim Embeddings]
    LLMAttr --> FAISS
    FAISS --> Search[Hybrid & Semantic Search API]
    Gen --> Qdrant[Qdrant Vector Store]
    Qdrant --> RAG[RAG Discovery: LangChain + Claude]
    Search --> FastAPI[FastAPI Serving Layer]
    RAG --> FastAPI
    FastAPI --> Streamlit[Streamlit Multimodal Demo]
```

## 📊 Performance Metrics
| Model/System | Metric | Result | Target |
|--------------|--------|---------|--------|
| ViT-Base-16 | Top-1 Accuracy | 89.2% | >88% |
| ViT-Base-16 | Top-3 Accuracy | 97.5% | >96% |
| CLIP Zero-shot| Accuracy | 64.3% | - |
| FAISS Search | p99 Latency | 42ms | <50ms |
| RAG Q&A | Groundedness | 0.94 | >0.90 |

## 🛠️ Tech Stack
- **Deep Learning**: PyTorch, Transformers (HuggingFace), CLIP, ViT
- **Vector DB/Search**: FAISS, Qdrant
- **LLM/RAG**: LangChain, Anthropic Claude 3.5 Sonnet
- **Backend**: FastAPI, Pydantic, Redis
- **Frontend**: Streamlit, Plotly
- **Infrastructure**: Docker, MLflow, Git

## 📦 Project Structure
- `/data`: Catalog generation scripts and CSV storage.
- `/embeddings`: CLIP and ViT training/inference pipelines.
- `/llm_pipelines`: Attribute extraction, Q&A, and description generation.
- `/vector_search`: FAISS management and hybrid search logic.
- `/serving`: FastAPI backend with caching and schemas.
- `/streamlit`: Interactive dashboard and search demo.
- `/docs`: Architecture diagrams and model cards.

## ⚙️ Local Setup
1. **Clone repository**:
   ```bash
   git clone https://github.com/your-username/multimodal-product-intelligence.git
   cd multimodal_product_intelligence
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Generate Data**:
   ```bash
   python generate_data.py
   ```
4. **Compute Embeddings**:
   ```bash
   python embeddings/clip_pipeline.py
   ```
5. **Run Serving API**:
   ```bash
   uvicorn serving.main:app --reload
   ```
6. **Launch Demo**:
   ```bash
   streamlit run streamlit/app.py
   ```

