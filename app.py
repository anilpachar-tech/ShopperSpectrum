"""
Shopper Spectrum
Customer Segmentation and Product Recommendation
"""
import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    kmeans = joblib.load("models/kmeans.pkl")
    scaler = joblib.load("models/scaler.pkl")
    similarity_df = joblib.load("models/similarity.pkl")

    return kmeans, scaler, similarity_df

kmeans, scaler, similarity_df = load_models()

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_products(product_name, n=5):

    if product_name not in similarity_df.index:
        return []

    recommendations = (
        similarity_df[product_name]
        .sort_values(ascending=False)
        .iloc[1:n+1]
    )

    return recommendations.index.tolist()

# -----------------------------
# Cluster Mapping
# -----------------------------
cluster_mapping = {
    0: "Occasional",
    1: "At-Risk",
    2: "High-Value",
    3: "Regular"
}

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🛒 Shopper Spectrum")

page = st.sidebar.radio(
    "Select Module",
    [
        "Product Recommendation",
        "Customer Segmentation"
    ]
)

# ====================================================
# PRODUCT RECOMMENDATION MODULE
# ====================================================
if page == "Product Recommendation":

    st.title("🛍 Product Recommendation System")

    st.write(
        "Enter a product name and get the top 5 similar product recommendations."
    )

    product_name = st.selectbox(
        "select Product",
        sorted(similarity_df.index)
    )

    if st.button("Get Recommendations"):

        if product_name.strip() == "":
            st.warning("Please enter a product name.")
        else:

            recommendations = recommend_products(
                product_name
            )

            if recommendations:

                st.success(
                    "Top 5 Recommended Products"
                )

                for i, item in enumerate(
                    recommendations,
                    start=1
                ):
                    st.write(
                        f"{i}. {item}"
                    )

            else:

                st.error(
                    "Product not found in dataset."
                )

# ====================================================
# CUSTOMER SEGMENTATION MODULE
# ====================================================
elif page == "Customer Segmentation":

    st.title("👥 Customer Segmentation")

    st.write(
        "Enter Recency, Frequency and Monetary values to predict customer segment."
    )

    recency = st.number_input(
        "Recency (Days Since Last Purchase)",
        min_value=0,
        value=30
    )

    frequency = st.number_input(
        "Frequency (Number of Purchases)",
        min_value=0,
        value=1
    )

    monetary = st.number_input(
        "Monetary (Total Spend)",
        min_value=0.0,
        value=100.0
    )

    if st.button("Predict Segment"):

        input_df = pd.DataFrame(
            [[
                recency,
                frequency,
                monetary
            ]],
            columns=[
                "Recency",
                "Frequency",
                "Monetary"
            ]
        )

        scaled_input = scaler.transform(
            input_df
        )

        cluster = kmeans.predict(
            scaled_input
        )[0]

        segment = cluster_mapping.get(
            cluster,
            "Unknown"
        )

        st.success(
            f"Predicted Segment: {segment}"
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "Shopper Spectrum | Customer Segmentation and Product Recommendation System"
)
