# Shopper Spectrum

## Customer Segmentation and Product Recommendation System

### Project Overview

Shopper Spectrum is a machine learning project that combines customer segmentation and product recommendation to improve customer understanding and personalized shopping experiences.

The project uses:

- RFM Analysis
- KMeans Clustering
- Item-Based Collaborative Filtering
- Cosine Similarity
- Streamlit

---

## Dataset

Online Retail Dataset

Features:

- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- CustomerID
- Country

---

## Project Workflow

### Day 1 - Data Cleaning and EDA

- Dataset Understanding
- Missing Value Handling
- Duplicate Analysis
- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis

### Day 2 - Customer Segmentation

- RFM Analysis
- Standardization
- Elbow Method
- Silhouette Score
- KMeans Clustering

Customer Segments:

- High-Value Customers
- Regular Customers
- Occasional Customers
- At-Risk Customers

### Day 3 - Product Recommendation System

- Customer Product Matrix
- Item-Based Collaborative Filtering
- Cosine Similarity
- Top 5 Product Recommendations

### Day 4 - Streamlit Deployment

- Product Recommendation Module
- Customer Segmentation Module
- Model Integration

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib
- Streamlit

---

## Project Structure

```text
ShopperSpectrum/
│
├── data/
├── models/
├── notebooks/
├── outputs/
├── app.py
├── requirements.txt
└── README.md
```

---

## Run Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit app:

```bash
streamlit run app.py
```

---

## Author

Anil Pachar