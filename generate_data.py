import os
import random
import uuid
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm

# Initialize Faker
fake = Faker()

# Configuration
NUM_PRODUCTS = 50000
NUM_REVIEWS = 500000
NUM_QUERIES = 200000
NUM_ATTRIBUTES = 150000
NUM_IMAGE_CLUSTERS = 500
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")

CATEGORIES = {
    "Electronics": ["Smarphones", "Laptops", "Headphones", "Cameras", "Smartwatches"],
    "Clothing": ["T-Shirts", "Jeans", "Jackets", "Dresses", "Shoes"],
    "Home & Kitchen": ["Blenders", "Coffee Makers", "Cookware", "Bedding", "Furniture"],
    "Beauty": ["Skincare", "Makeup", "Fragrance", "Haircare", "Nail Polish"],
    "Toys": ["Action Figures", "Board Games", "Dolls", "Puzzles", "LEGO"],
    "Sports": ["Yoga Mats", "Dumbbells", "Running Gear", "Bicycles", "Camping Gear"],
    "Books": ["Fiction", "Non-Fiction", "Sci-Fi", "Mystery", "Biography"],
    "Automotive": ["Tires", "Car Seats", "Dash Cams", "Polish", "Batteries"],
    "Food": ["Snacks", "Coffee Beans", "Cereal", "Pasta", "Organic Fruits"],
    "Pet Supplies": ["Dog Food", "Cat Toys", "Aquariums", "Bird Cages", "Pet Beds"],
    "Health": ["Vitamins", "First Aid", "Thermometers", "Masks", "Supplements"],
    "Office": ["Printers", "Desks", "Chairs", "Notebooks", "Pens"]
}

CATEGORY_COLORS = {
    "Electronics": (128, 128, 128),  # Grey/Black
    "Clothing": None,                 # Varied
    "Home & Kitchen": (245, 245, 220), # Beige
    "Beauty": (255, 192, 203),        # Pinkish
    "Toys": (255, 255, 0),            # Yellow
    "Sports": (0, 0, 255),            # Blue
    "Books": (139, 69, 19),           # Brown
    "Automotive": (0, 0, 0),          # Black
    "Food": (255, 165, 0),            # Warm (Orange)
    "Pet Supplies": (34, 139, 34),    # Forest Green
    "Health": (255, 255, 255),        # White
    "Office": (0, 128, 128)           # Teal
}

def generate_synthetic_image(category, cluster_id):
    """Generates a 224x224 synthetic product image with realistic textures."""
    width, height = 224, 224
    base_color = CATEGORY_COLORS.get(category)
    
    if base_color is None: # Varied for clothing etc
        base_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    # Create base image
    img_array = np.full((height, width, 3), base_color, dtype=np.uint8)
    
    # Add Gradient
    for i in range(height):
        grad = i / height * 50
        img_array[i, :, :] = np.clip(img_array[i, :, :].astype(int) - int(grad), 0, 255).astype(np.uint8)

    # Add Gaussian Noise
    noise = np.random.normal(0, 15, (height, width, 3)).astype(np.int16)
    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Add geometric shapes (product-like blobs)
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    
    shape_color = tuple(np.clip(np.array(base_color) + 40, 0, 255).astype(int))
    
    # Simple rectangle/ellipse to simulate product body
    draw.chord([50, 50, 170, 170], 0, 360, fill=shape_color, outline=(255,255,255))
    
    # Save image
    filename = f"cluster_{cluster_id:03d}.jpg"
    filepath = os.path.join(IMAGE_DIR, filename)
    img.save(filepath, "JPEG", quality=90)
    return filename

def generate_data():
    print("Starting Synthetic Data Generation...")
    
    # 1. Generate Image Clusters
    print("Generating 500 Image Clusters...")
    cluster_mapping = []
    category_list = list(CATEGORIES.keys())
    for i in range(NUM_IMAGE_CLUSTERS):
        cat = random.choice(category_list)
        filename = generate_synthetic_image(cat, i)
        cluster_mapping.append({"cluster_id": i, "category": cat, "filename": filename})
    
    df_clusters = pd.DataFrame(cluster_mapping)
    
    # 2. Generate Products
    print(f"Generating {NUM_PRODUCTS} Products...")
    products = []
    product_image_mapping = []
    
    for i in range(NUM_PRODUCTS):
        category = random.choice(category_list)
        subcategory = random.choice(CATEGORIES[category])
        
        # Map to a cluster of the same category
        valid_clusters = df_clusters[df_clusters['category'] == category]['filename'].tolist()
        img_filename = random.choice(valid_clusters)
        
        p_id = f"PROD_{i:05d}"
        
        # Domain specific vocab for titles
        title_length = random.randint(8, 15)
        title = " ".join(fake.words(nb=title_length, ext_word_list=[category, subcategory] + fake.words(50)))
        
        desc_length = random.randint(50, 200)
        description = fake.paragraph(nb_sentences=desc_length // 10)
        
        products.append({
            "product_id": p_id,
            "title": title[:200],
            "description": description,
            "category": category,
            "subcategory": subcategory,
            "brand": fake.company(),
            "price": round(random.uniform(9.99, 1499.99), 2),
            "avg_rating": round(np.clip(np.random.normal(4.1, 0.8), 1, 5), 1),
            "review_count": random.randint(0, 10000),
            "image_url_placeholder": f"images/{img_filename}",
            "color": fake.color_name(),
            "material": random.choice(["Plastic", "Metal", "Cotton", "Wood", "Glass"]),
            "size_options": random.choice(["S,M,L", "One Size", "4GB,8GB,16GB"]),
            "weight_grams": random.randint(100, 5000),
            "days_since_listed": random.randint(1, 1000),
            "is_active": True
        })
        
        product_image_mapping.append({
            "product_id": p_id,
            "image_filename": img_filename
        })
    
    df_products = pd.DataFrame(products)
    df_prod_img = pd.DataFrame(product_image_mapping)
    df_products.to_csv(os.path.join(DATA_DIR, "products.csv"), index=False)
    df_prod_img.to_csv(os.path.join(DATA_DIR, "product_image_mapping.csv"), index=False)

    # 3. Generate Reviews
    print(f"Generating {NUM_REVIEWS} Reviews...")
    reviews = []
    prod_ids = df_products['product_id'].tolist()
    
    for i in tqdm(range(NUM_REVIEWS)):
        r_id = f"REV_{i:06d}"
        reviews.append({
            "review_id": r_id,
            "product_id": random.choice(prod_ids),
            "user_id": f"USER_{random.randint(1, 100000):06d}",
            "rating": round(np.clip(np.random.normal(4.1, 0.8), 1, 5), 0),
            "review_text": fake.text(max_nb_chars=200),
            "review_date": fake.date_between(start_date='-2y', end_date='now'),
            "verified_purchase": random.random() < 0.8,
            "helpful_votes": random.randint(0, 100)
        })
    pd.DataFrame(reviews).to_csv(os.path.join(DATA_DIR, "reviews.csv"), index=False)

    # 4. Generate User Queries
    print(f"Generating {NUM_QUERIES} User Queries...")
    queries = []
    for i in tqdm(range(NUM_QUERIES)):
        q_id = f"QRY_{i:06d}"
        
        # Simulation: 18% CTR, 3.2% Conversion
        is_click = random.random() < 0.18
        is_purchase = random.random() < 0.032
        
        clicked_ids = ""
        purchased_id = ""
        
        if is_click:
            clicked_ids = ",".join([random.choice(prod_ids) for _ in range(random.randint(1, 3))])
        if is_purchase and is_click:
            purchased_id = clicked_ids.split(",")[0]
            
        queries.append({
            "query_id": q_id,
            "user_id": f"USER_{random.randint(1, 100000):06d}",
            "query_text": fake.sentence(nb_words=random.randint(2, 6)).replace(".", ""),
            "query_timestamp": fake.date_time_between(start_date='-1y', end_date=datetime.now()),
            "clicked_product_ids": clicked_ids,
            "purchased_product_id": purchased_id,
            "session_id": str(uuid.uuid4())
        })
    pd.DataFrame(queries).to_csv(os.path.join(DATA_DIR, "user_queries.csv"), index=False)

    # 5. Product Attributes
    print(f"Generating {NUM_ATTRIBUTES} Attribute Records...")
    attributes = []
    attr_names = ["Resolution", "Fabric Weight", "Energy Class", "Compatibility", "Maintenance", "Warranty"]
    for i in tqdm(range(NUM_ATTRIBUTES)):
        a_id = f"ATTR_{i:06d}"
        attributes.append({
            "attribute_id": a_id,
            "product_id": random.choice(prod_ids),
            "attribute_name": random.choice(attr_names),
            "attribute_value": fake.word(),
            "extraction_source": random.choice(["manual", "llm_extracted", "vision_extracted"])
        })
    pd.DataFrame(attributes).to_csv(os.path.join(DATA_DIR, "product_attributes.csv"), index=False)

    print("Data Generation Complete.")

if __name__ == "__main__":
    generate_data()
