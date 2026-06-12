# Multimodal Product Intelligence: Interview Preparation

### 1. How does CLIP align image and text embeddings in the same space, and why does this enable zero-shot classification?
**Answer:** CLIP (Contrastive Language-Image Pre-training) is trained using a contrastive loss (InfoNCE) on millions of image-text pairs. It pushes the embedding of an image and its corresponding text together while pulling apart embeddings of non-matching pairs. This creates a Shared Latent Space. Since categories can be represented as text strings (e.g., "a photo of a laptop"), we can compute their embeddings and find the nearest neighbor for any image embedding without ever training on those specific labels.

### 2. Difference between IndexFlatIP and IndexIVFFlat in FAISS?
**Answer:** `IndexFlatIP` performs an exact exhaustive search using Inner Product (cosine similarity on normalized vectors). It is accurate but O(N) complexity. `IndexIVFFlat` (Inverted File) is an approximate search; it clusters vectors into `nlist` centroids. At search time, it only probes the closest `nprobe` clusters, significantly reducing search time to O(sqrt(N)) or less, though with a small drop in recall.

### 3. How to handle a product where image and text are contradictory?
**Answer:** In production, we use a "Modality Gap" analysis or a Conflict Score. If the cosine similarity between the image and text embeddings is below a threshold (e.g., < 0.3), we flag it for manual review or fallback to a text-only index. Alternatively, we use an attention-based fusion mechanism that learns which modality to trust more based on the category.

### 4. Why use Cross-Encoder re-ranking instead of just Bi-Encoder retrieval?
**Answer:** Bi-Encoders (like CLIP or Sentence-Transformers) compute embeddings independently, which is fast for retrieval but ignores interaction between query and doc. Cross-Encoders process both together via self-attention, capturing fine-grained nuances. We use Bi-Encoders for a fast "Recall" stage (top 100) and Cross-Encoders for a slower "Precision" stage (top 10).

### 5. How to evaluate RAG without ground truth?
**Answer:** We use the "RAG Triad" (Ragas framework): 
1. **Faithfulness**: Is the answer derived solely from the retrieved context? (NLI check).
2. **Relevance**: Does the answer actually address the user query?
3. **Context Precision**: Are the retrieved documents actually relevant to the query?
We can use an LLM-as-a-judge (Claude/GPT-4) to score these metrics on a scale of 1-5.

### 6. What is Training-Serving Skew?
**Answer:** It occurs when the data distribution at inference time differs from training. For embeddings, this often happens if we update the model version but don't re-index the vector database. We detect this by monitoring the distribution of similarity scores or using a "Golden Test Set" that is run through both the old and new models.

### 7. How to fine-tune CLIP on limited domain data?
**Answer:** We use **LoRA (Low-Rank Adaptation)** or **Prompt Tuning**. Instead of updating all 150M+ parameters, we inject small trainable matrices into the transformer layers. This prevents catastrophic forgetting of general features while adapting the model to specific product textures/vocabularies with only a few thousand samples.

### 8. BERTScore vs. BLEU?
**Answer:** BLEU measures exact n-gram overlap, which is poor for creative descriptions (it penalizes synonyms). BERTScore uses contextual embeddings to measure semantic similarity. It captures the meaning even if different words are used, making it much better for evaluating LLM-generated summaries or descriptions.

### 9. Scaling FAISS from 50K to 50M products?
**Answer:** 
1. **Product Quantization (PQ)**: Compress 512-dim vectors to 64 bytes to fit in RAM.
2. **HNSW Index**: Use a graph-based index for faster sub-millisecond search at scale.
3. **Sharding**: Distribute the index across multiple machines (e.g., using FAISS-Server or Milvus).

### 10. How to A/B test embedding models for purchases?
**Answer:** We use an **Online Experiment** where 50% of users see results from Model A (current) and 50% from Model B (new). We track **Conversion Rate (CVR)** and **Revenue per Search (RPS)**. We also use **Interleaving**, where we mix results from both models in a single list and track which results get more clicks to get faster signal than a traditional A/B test.
