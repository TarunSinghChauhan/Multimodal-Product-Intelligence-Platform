import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
from embeddings.utils.fusion import fuse_embeddings
from embeddings.utils.accuracy import calculate_accuracy

# Configuration
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
EMBEDDING_DIR = "embeddings"
MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32

os.makedirs(EMBEDDING_DIR, exist_ok=True)

def run_clip_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading CLIP model: {MODEL_NAME}...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # 1. Load Data
    df_products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    df_mapping = pd.read_csv(os.path.join(DATA_DIR, "product_image_mapping.csv"))
    
    unique_images = df_mapping['image_filename'].unique()
    categories = df_products['category'].unique().tolist()

    # 2. Compute Image Embeddings (for the 500 clusters)
    print(f"Computing embeddings for {len(unique_images)} images...")
    image_embeddings_dict = {}
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(unique_images), BATCH_SIZE)):
            batch_filenames = unique_images[i:i+BATCH_SIZE]
            images = [Image.open(os.path.join(IMAGE_DIR, f)) for f in batch_filenames]
            
            inputs = processor(images=images, return_tensors="pt").to(device)
            image_features = model.get_image_features(**inputs)
            image_features = F.normalize(image_features, p=2, dim=-1)
            
            for filename, feat in zip(batch_filenames, image_features):
                image_embeddings_dict[filename] = feat.cpu().numpy()

    # 3. Compute Text Embeddings for 50k products
    print(f"Computing embeddings for {len(df_products)} product texts...")
    text_embeddings = []
    product_ids = df_products['product_id'].values
    
    # Text = Title + Description
    texts = (df_products['title'] + " " + df_products['description']).tolist()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE)):
            batch_texts = texts[i:i+BATCH_SIZE]
            # Clip text to max length (77 tokens for CLIP)
            inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            text_features = model.get_text_features(**inputs)
            text_features = F.normalize(text_features, p=2, dim=-1)
            text_embeddings.append(text_features.cpu().numpy())
            
    text_embeddings = np.vstack(text_embeddings)

    # 4. Map Image Embeddings to each product
    print("Mapping image embeddings to products...")
    full_image_embeddings = []
    prod_to_img = dict(zip(df_mapping['product_id'], df_mapping['image_filename']))
    
    for p_id in product_ids:
        img_file = prod_to_img[p_id]
        full_image_embeddings.append(image_embeddings_dict[img_file])
    
    full_image_embeddings = np.vstack(full_image_embeddings)

    # 5. Multimodal Fusion (0.4 image, 0.6 text)
    print("Fusing embeddings...")
    fusion_embeddings = fuse_embeddings(full_image_embeddings, text_embeddings, image_weight=0.4, text_weight=0.6)

    # Save results
    np.save(os.path.join(EMBEDDING_DIR, "image_embeddings.npy"), full_image_embeddings)
    np.save(os.path.join(EMBEDDING_DIR, "text_embeddings.npy"), text_embeddings)
    np.save(os.path.join(EMBEDDING_DIR, "fusion_embeddings.npy"), fusion_embeddings)
    np.save(os.path.join(EMBEDDING_DIR, "product_ids.npy"), product_ids)

    # 6. Zero-Shot Category Classification
    print("Running zero-shot classification evaluation...")
    prompts = [f"a photo of a {cat} product" for cat in categories]
    with torch.no_grad():
        inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
        category_features = model.get_text_features(**inputs)
        category_features = F.normalize(category_features, p=2, dim=-1) # (12, 512)
    
    # Compute similarity between fusion embeddings and prompts
    fusion_tensor = torch.from_numpy(fusion_embeddings).to(device)
    similarities = torch.matmul(fusion_tensor, category_features.T) # (50000, 12)
    predictions = torch.argmax(similarities, dim=1).cpu().numpy()
    
    # Calculate accuracy
    true_labels = df_products['category'].map({cat: i for i, cat in enumerate(categories)}).values
    accuracy = calculate_accuracy(predictions, true_labels)
    
    print(f"Zero-shot Classification Accuracy: {accuracy:.4f}")
    
    with open(os.path.join(EMBEDDING_DIR, "accuracy_report.txt"), "w") as f:
        f.write(f"Zero-shot Category Classification Accuracy: {accuracy:.4f}\n")

if __name__ == "__main__":
    run_clip_pipeline()
