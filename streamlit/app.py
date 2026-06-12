import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image
import base64
import plotly.express as px
import io

# Page Config
st.set_page_config(page_title="Multimodal Product Intelligence", layout="wide")

# Custom CSS for Amazon-like feel
st.markdown("""
    <style>
    .main {
        background-color: #f6f6f6;
    }
    .stButton>button {
        background-color: #febd69;
        color: #131921;
        border-radius: 5px;
        border: none;
        font-weight: bold;
    }
    .product-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .score-badge {
        background-color: #e67e22;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
    }
    .attr-chip {
        background-color: #f0f0f0;
        color: #555;
        padding: 2px 8px;
        border-radius: 5px;
        font-size: 0.7em;
        margin-right: 5px;
    }
    h1, h2, h3 {
        color: #232f3e;
    }
    </style>
""", unsafe_allow_html=True)

# Helper functions
def get_image_base64(image_file):
    return base64.b64encode(image_file.getvalue()).decode()

# Sidebar Navigation
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=150)
page = st.sidebar.radio("Navigate", ["Text Search", "Image Search", "Multimodal Search", "Product Q&A", "Analytics Dashboard"])

# Mock Data for UI demonstration
df_products = pd.read_csv("data/products.csv") if os.path.exists("data/products.csv") else pd.DataFrame()

if page == "Text Search":
    st.title("🔍 Semantic Product Search")
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.subheader("Filters")
        category = st.multiselect("Category", df_products['category'].unique() if not df_products.empty else [])
        price_range = st.slider("Price Range", 0, 1500, (0, 1500))
        min_rating = st.slider("Minimum Rating", 1.0, 5.0, 4.0)

    with col2:
        query = st.text_input("Enter your search query...", placeholder="e.g., portable wireless headphones with noise cancellation")
        
        if query:
            st.write(f"Showing results for: **{query}**")
            
            # Simple grid
            cols = st.columns(3)
            # Simulated results
            test_results = df_products.sample(6) if not df_products.empty else []
            
            for i, (_, row) in enumerate(test_results.iterrows()):
                with cols[i % 3]:
                    st.markdown(f"""
                        <div class="product-card">
                            <span class="score-badge">0.92 Match</span>
                            <h4>{row['title'][:40]}...</h4>
                            <p><b>{row['category']}</b></p>
                            <p style="color: #B12704; font-size: 1.2em;">${row['price']}</p>
                            <p>⭐ {row['avg_rating']} ({row['review_count']})</p>
                            <div>
                                <span class="attr-chip">{row['material']}</span>
                                <span class="attr-chip">{row['color']}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

elif page == "Image Search":
    st.title("📸 Visual Product Search")
    uploaded_file = st.file_uploader("Upload a product image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", width=200)
        st.info("Searching for visually similar products...")
        
        # Grid of results
        cols = st.columns(4)
        for i in range(4):
            with cols[i]:
                st.markdown(f"""
                    <div class="product-card">
                        <span class="score-badge">0.88 Similarity</span>
                        <img src="https://via.placeholder.com/150" width="100%"/>
                        <h5>Similar Product {i+1}</h5>
                        <p>$24.99</p>
                    </div>
                """, unsafe_allow_html=True)

elif page == "Product Q&A":
    st.title("💬 Product Assistant (RAG)")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about our products..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = "Based on our catalog, this product features high-quality materials and has a 4.5-star rating. Customers frequently mention its durability."
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Show sources
            st.subheader("Source Products")
            cols = st.columns(3)
            for i in range(2):
                with cols[i]:
                    st.markdown("""<div class="product-card">Product Info</div>""", unsafe_allow_html=True)

elif page == "Analytics Dashboard":
    st.title("📊 Platform Analytics")
    
    c1, c2 = st.columns(2)
    with c1:
        # Pie chart for category distribution
        if not df_products.empty:
            fig = px.pie(df_products, names='category', title='Catalog Distribution by Category')
            st.plotly_chart(fig)
            
    with c2:
        # Histogram for ratings
        if not df_products.empty:
            fig = px.histogram(df_products, x='avg_rating', title='Rating Distribution', color_discrete_sequence=['#febd69'])
            st.plotly_chart(fig)

    # Line chart for simulated query volume
    query_data = pd.DataFrame({
        'Hour': list(range(24)),
        'Queries': [100, 80, 50, 40, 60, 150, 300, 600, 800, 1000, 1100, 1050, 950, 1000, 1200, 1150, 1300, 1100, 900, 700, 500, 400, 300, 200]
    })
    fig = px.line(query_data, x='Hour', y='Queries', title='Hourly Query Volume')
    st.plotly_chart(fig, use_container_width=True)
