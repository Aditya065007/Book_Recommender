import os
import pickle
import gdown
import streamlit as st
import pandas as pd

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="📚 Book Recommendation Dashboard",
    layout="wide"
)

# =====================================================
# DOWNLOAD FILES FROM GOOGLE DRIVE (ONCE)
# =====================================================
def fetch(file_id, filename):
    if not os.path.exists(filename):
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            filename,
            quiet=False
        )
FILES = {
    "book_meta.pkl": "1R02sJLBmQNVG__RqlIEMCVo052Q6e0pI",
    "item_data.csv": "1b5j6e69PuVm7Od21oEmfSDUZLayi2t6y",
    "item_similarity_topk.pkl": "1UkSttFZK2JAu9qTJOh-SrGan9HGgdZLm",
    "ratings_with_meta.csv": "19EsVVj1aZVUBUZnvz4u0ATKxtSiVKPcX",
    "svd_model.pkl": "1s_k4svq1ZwTxeaJDHfn0OZiJMtIImQOT",
    "tfidf_vectorizer.pkl": "1UwoY5WY2KR5Ag2mwhsXD46iK4boebK0C",
    "users_10001_names.csv": "1cUYVoMpXwhbPDnud_TswfaQWP8Z5MsiC"
}

for fname, fid in FILES.items():
    fetch(fid, fname)

# =====================================================
# LOAD DATA & MODELS
# =====================================================
@st.cache_data(show_spinner="Loading models and data...")
def load_data():
    svd_model = pickle.load(open("svd_model.pkl", "rb"))
    tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    item_similarity_topk = pickle.load(open("item_similarity_topk.pkl", "rb"))

    item_df = pd.read_csv("item_data.csv")
    ratings_df = pd.read_csv("ratings_with_meta.csv", low_memory=False)
    book_meta = pickle.load(open("book_meta.pkl", "rb"))
    user_map = pd.read_csv("users_10001_names.csv")

    return (
        svd_model,
        tfidf_vectorizer,
        item_similarity_topk,
        item_df,
        ratings_df,
        book_meta,
        user_map
    )

(
    svd_model,
    tfidf_vectorizer,
    item_similarity_topk,
    item_df,
    ratings_df,
    book_meta,
    user_map
) = load_data()

# Normalize column names just in case
user_map.columns = user_map.columns.str.lower()

# Create user name → ID mapping
user_name_to_id = dict(zip(user_map["user_name"], user_map["user_id"]))

# =====================================================
# HEADER
# =====================================================
st.title("📚 Book Recommendation System")
st.divider()

# =====================================================
# MODE SELECTION
# =====================================================
mode = st.radio(
    "Choose Recommendation Mode",
    ["👤 Recommended by a critic", "📘 Recommend Similar Books"],
    horizontal=True
)

# =====================================================
# COLLABORATIVE FILTERING
# =====================================================
if mode == "👤 Recommended by a critic":

    st.subheader("👤 Recommendations by critic")

    selected_user_name = st.selectbox(
        "Select Critic",
        options=sorted(user_name_to_id.keys())
    )

    user_id = user_name_to_id[selected_user_name]
    top_n = st.slider("Number of recommendations", 5, 15, 10)

    if st.button("Get Recommendations"):

        seen_items = set(
            ratings_df[ratings_df["user_id"] == user_id]["item_id"]
        )
        all_items = set(ratings_df["item_id"].unique())
        unseen_items = all_items - seen_items

        preds = [
            (iid, svd_model.predict(user_id, iid).est)
            for iid in unseen_items
        ]
        preds.sort(key=lambda x: x[1], reverse=True)

        top_items = [i[0] for i in preds[:top_n]]

        recommendations = book_meta[
            book_meta["item_id"].isin(top_items)
        ]

        st.markdown("### 📚 Recommended Books")

        for _, row in recommendations.iterrows():
            st.markdown(
                f"""
                <div style="padding:16px; margin-bottom:12px;
                            border-radius:10px; background-color:#f8f9fa;
                            border-left:6px solid #4CAF50;">
                    <h4>📘 {row['title']}</h4>
                    <p>
                        ✍️ <b>{row['author']}</b><br>
                        📅 {row['year']}<br>
                        🏢 {row['publisher']}<br>
                        ⭐ Average Rating: {row['avg_rating']:.2f}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

# =====================================================
# CONTENT-BASED FILTERING (TOP-K SAFE)
# =====================================================
else:

    st.subheader("📘 Find Similar Books")

    book_title = st.selectbox(
        "Select a Book",
        sorted(item_df["title"].unique())
    )

    top_n = st.slider("Number of similar books", 5, 15, 10)

    if st.button("Find Similar Books"):

        idx = item_df[item_df["title"] == book_title].index[0]

        # Top-K similarity lookup (already sorted)
        similar_items = item_similarity_topk[idx][:top_n]
        similar_item_ids = [
            item_df.iloc[i]["item_id"] for i, _ in similar_items
        ]

        recommendations = book_meta[
            book_meta["item_id"].isin(similar_item_ids)
        ]

        st.markdown("### 📖 Similar Book Recommendations")

        for _, row in recommendations.iterrows():
            st.markdown(
                f"""
                <div style="padding:16px; margin-bottom:12px;
                            border-radius:10px; background-color:#eef3ff;
                            border-left:6px solid #3f51b5;">
                    <h4>📘 {row['title']}</h4>
                    <p>
                        ✍️ <b>{row['author']}</b><br>
                        📅 {row['year']}<br>
                        🏢 {row['publisher']}<br>
                        ⭐ Average Rating: {row['avg_rating']:.2f}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
