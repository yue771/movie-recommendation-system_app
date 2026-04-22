import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


# =========================
# UI Style
# =========================
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.markdown("""
<style>
.stApp {background-color: #0e1117;}
h1, h2, h3, h4 {color: white;}
</style>
""", unsafe_allow_html=True)


# =========================
# Helper
# =========================
def get_poster_url(poster_path):
    if pd.isna(poster_path) or poster_path == "" or poster_path is None:
        return None
    return f"https://image.tmdb.org/t/p/w500{poster_path}"


# =========================
# Load Data
# =========================
@st.cache_data
def load_data():
    movies = pd.read_csv("movies_small1.csv")
    ratings = pd.read_csv("ratings_small.csv")
    links = pd.read_csv("links_small.csv")
    return movies, ratings, links

movies, ratings, links = load_data()


# =========================
# Content-based
# =========================
@st.cache_data
def prepare_content(df):
    df = df.copy()
    df["overview"] = df["overview"].fillna("")
    df["genres"] = df["genres"].fillna("")

    def parse(x):
        try:
            return " ".join([i["name"] for i in ast.literal_eval(x)])
        except:
            return ""

    df["genres_text"] = df["genres"].apply(parse)
    df["content"] = df["genres_text"] + " " + df["overview"]

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    mat = tfidf.fit_transform(df["content"])

    sim = cosine_similarity(mat, mat)

    return df, pd.DataFrame(sim, index=df["title"], columns=df["title"])

movies_content, content_sim = prepare_content(movies)


# =========================
# Collaborative
# =========================
@st.cache_data
def prepare_cf(ratings, movies, links):
    movie_map = movies[["id", "title"]].copy()
    movie_map["id"] = pd.to_numeric(movie_map["id"], errors="coerce")
    movie_map = movie_map.dropna()
    movie_map["id"] = movie_map["id"].astype(int)

    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce")
    links = links.dropna()

    df = ratings.merge(links, on="movieId")
    df = df.merge(movie_map, left_on="tmdbId", right_on="id")

    user_movie = df.pivot_table(index="userId", columns="title", values="rating")

    movie_user = user_movie.T.fillna(0)

    sim = cosine_similarity(movie_user)

    return pd.DataFrame(sim, index=movie_user.index, columns=movie_user.index)

movie_sim = prepare_cf(ratings, movies, links)


# =========================
# Recommendation
# =========================
def recommend(movie, top_n=10, alpha=0.5):
    content_scores = content_sim[movie].sort_values(ascending=False)[1:50]
    cf_scores = movie_sim[movie].sort_values(ascending=False)[1:50] if movie in movie_sim else pd.Series()

    df = pd.DataFrame({"Movie": content_scores.index})
    df["Content"] = content_scores.values
    df["CF"] = df["Movie"].map(cf_scores).fillna(0)
    df["Hybrid"] = alpha * df["CF"] + (1 - alpha) * df["Content"]

    return df.sort_values("Hybrid", ascending=False).head(top_n)


# =========================
# Explainability（🔥核心）
# =========================
def explain(selected, rec):
    try:
        a = movies_content[movies_content["title"] == selected].iloc[0]
        b = movies_content[movies_content["title"] == rec].iloc[0]

        g1 = set(a["genres_text"].split())
        g2 = set(b["genres_text"].split())

        common = g1 & g2

        if common:
            return f"Both are {' / '.join(list(common))} movies"
        else:
            return "Similar themes or storyline"
    except:
        return ""


# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ Settings")

search = st.sidebar.text_input("🔍 Search movie")

movies_list = sorted(movies_content["title"].tolist())
filtered = [m for m in movies_list if search.lower() in m.lower()]

selected_movie = st.sidebar.selectbox("Choose movie", filtered if filtered else movies_list)

top_n = st.sidebar.slider("Top N", 5, 20, 10)
alpha = st.sidebar.slider("Hybrid Weight", 0.0, 1.0, 0.5)


# =========================
# KPI
# =========================
k1, k2, k3 = st.columns(3)
k1.metric("🎬 Movies", movies_content.shape[0])
k2.metric("⭐ Ratings", ratings.shape[0])
k3.metric("📊 CF Movies", movie_sim.shape[0])


# =========================
# Movie Info
# =========================
st.markdown(f"## 🎥 {selected_movie}")

row = movies_content[movies_content["title"] == selected_movie].iloc[0]

c1, c2 = st.columns([1, 2])

with c1:
    poster = get_poster_url(row.get("poster_path"))
    if poster:
        st.image(poster, use_container_width=True)
    else:
        st.write("No Poster")

with c2:
    st.write(f"**Genres:** {row['genres_text']}")
    st.write(f"**Overview:** {row['overview']}")


# =========================
# Recommendations（🔥）
# =========================
st.markdown("## 🍿 Recommended for you")

result = recommend(selected_movie, top_n, alpha)

# merge海报
movies_content["title_clean"] = movies_content["title"].str.lower().str.strip()
result["title_clean"] = result["Movie"].str.lower().str.strip()

result = result.merge(
    movies_content[["title_clean", "poster_path"]],
    on="title_clean",
    how="left"
)

cols = st.columns(5)

for i, r in result.iterrows():
    col = cols[i % 5]

    with col:
        poster = get_poster_url(r["poster_path"])

        if poster:
            st.image(poster, use_container_width=True)
        else:
            st.write("No Poster")

        st.caption(r["Movie"])
        st.write(f"⭐ {r['Hybrid']:.2f}")

        st.caption("💡 " + explain(selected_movie, r["Movie"]))


# =========================
# Visualization（🔥）
# =========================
st.markdown("## 📊 Recommendation Scores")

fig, ax = plt.subplots()
ax.barh(result["Movie"], result["Hybrid"])
ax.invert_yaxis()
st.pyplot(fig)


# =========================
# Download
# =========================
csv = result.to_csv(index=False).encode("utf-8")

st.download_button(
    "📥 Download CSV",
    data=csv,
    file_name="recommendations.csv",
    mime="text/csv",
    key="download_unique"
)