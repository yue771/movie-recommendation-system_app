# 🎬 Movie Recommendation System (Hybrid + Explainable AI)

An end-to-end movie recommendation system combining collaborative filtering and content-based methods, with an interactive Streamlit dashboard and live deployment.

👉 **Live Demo:**  
https://movie-recommendation-systemapp-2vavrtwedmswdr4ytun4mr.streamlit.app/

---

## 📌 Project Overview

This project builds a **hybrid recommendation system** that integrates:

- **Collaborative Filtering** (user behavior)
- **Content-Based Filtering** (movie genres & descriptions)
- **Hybrid Model** (weighted combination)

The system is deployed using **Streamlit**, allowing real-time interaction, visualization, and explainable recommendations.

---

## ⚙️ Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn (TF-IDF, Cosine Similarity)
- Streamlit (Web App)
- Matplotlib (Visualization)

---

## 🧠 Methodology

### 1. Content-Based Filtering
- Extract features from:
  - Movie genres
  - Movie overview
- Apply **TF-IDF vectorization**
- Compute similarity using **cosine similarity**

---

### 2. Collaborative Filtering
- Construct **user-movie matrix**
- Compute similarity between movies based on user ratings
- Use cosine similarity for item-item recommendation

---

### 3. Hybrid Recommendation
Final score is calculated as:
Hybrid Score = α × Collaborative + (1 - α) × Content

- Adjustable weight `α`
- Combines behavior + semantic similarity

---

## 💡 Key Features

- 🔍 **Search-based movie selection**
- 🎬 **Poster visualization (TMDB API)**
- ⭐ **Hybrid recommendation system**
- 💡 **Explainable recommendations**
- 📊 **Score visualization (bar chart)**
- 📥 **Download recommendation results**

---

## 🖥️ Demo Preview

| Feature | Description |
|--------|------------|
| Search | Input movie name dynamically |
| Recommendation | Netflix-style card layout |
| Explainability | Shows why a movie is recommended |
| Visualization | Displays recommendation scores |
| Deployment | Accessible via Streamlit Cloud |

---

## 📂 Project Structure
movie-recommendation-system/
│
├── app_v5.py # Main Streamlit app
├── movies_small1.csv # Movie metadata
├── ratings_small.csv # User ratings
├── links_small.csv # Mapping file
├── requirements.txt
└── README.md

---

--- 

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app_v5.py
