import os
import pickle
import gdown
import streamlit as st
import pandas as pd

# --------------------------------------------------
# Download helper (Google Drive → local)
# --------------------------------------------------
def fetch(file_id, filename):
    if not os.path.exists(filename):
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}",
            filename,
            quiet=False
        )

# --------------------------------------------------
# Google Drive FILE IDS (REPLACE THESE)
# --------------------------------------------------
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

# --------------------------------------------------
# Load data & models
# --------------------------------------------------
book_meta = pickle.load(open("book_meta.pkl", "rb"))
item_df = pd.read_csv("item_data.csv")
ratings_df = pd.read_csv("ratings_with_meta.csv")
users_df = pd.read_csv("users_10001_names.csv")

item_similarity_topk = pickle.load(open("item_similarity_topk.pkl", "rb"))
svd_model = pickle.load(open("svd_model.pkl", "rb"))

# --------------------------------------------------
# Recommendation functions
# --------------------------------------------------
def recommend_collaborative(user_id, top_n=5):
    seen = set(ratings_df[ratings_df["user_id"] == user_id]["item_id"])
    all_items = set(ratings_df["item_id"].unique())
    unseen = all_items - seen

    preds = [(iid, svd_model.predict(user_id, iid).est) for iid in unseen]
    preds.sort(key=lambda x: x[1], reverse=True)

    top_items = [i[0] for i in preds[:top_n]]
    return book_meta[book_meta["item_id"].isin(top_items)][
        ["title", "author", "year", "publisher", "avg_rating"]
    ]

def recommend_content(book_title, top_n=5):
    idx = item_df[item_df["title"] == book_title].index[0]

    # Top-K similarity (already sorted)
    similar_items = item_similarity_topk[idx][:top_n]
    similar_item_ids = [item_df.iloc[i]["item_id"] for i, _ in similar_items]

    return book_meta[book_meta["item_id"].isin(similar_item_ids)][
        ["title", "author", "year", "publisher", "avg_rating"]
    ]

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Book Recommender", layout="wide")
st.title("📚 Book Recommendation System")

mode = st.radio(
    "Choose recommendation type:",
    ["Collaborative Filtering", "Content-Based"],
    horizontal=True
)

if mode == "Collaborative Filtering":
    user_name = st.selectbox("Select User", users_df["name"])
    user_id = users_df[users_df["name"] == user_name]["user_id"].iloc[0]

    if st.button("Get Recommendations"):
        recs = recommend_collaborative(user_id)
        st.dataframe(recs, use_container_width=True)

else:
    book_title = st.selectbox("Select a Book", item_df["title"].values)

    if st.button("Get Similar Books"):
        recs = recommend_content(book_title)
        st.dataframe(recs, use_container_width=True)
