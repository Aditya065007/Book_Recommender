# Book Recommendation System

A Streamlit dashboard offering two classic recommendation approaches over a book ratings dataset: collaborative filtering (based on a selected reader's rating history) and content-based filtering (based on similarity to a chosen book).

## Features

- **Collaborative filtering ("Recommended by a critic")** — predicts ratings for books a selected user hasn't rated yet using an SVD matrix-factorization model, and returns the top-N highest-predicted books
- **Content-based filtering ("Recommend Similar Books")** — returns the nearest neighbors to a selected book using a precomputed item-similarity matrix
- **Adjustable output** — recommendation count is configurable (5–15) in both modes
- **Readable results** — recommendations are displayed with title, author, year, publisher, and average rating, styled as cards

## Tech Stack

- **scikit-surprise** — SVD-based collaborative filtering
- **pandas / pickle** — data and precomputed-model loading
- **Streamlit** — UI, with cached data loading via `@st.cache_data`

## How It Works

- **Collaborative filtering:** for a selected user, the SVD model predicts a rating for every book they haven't already rated; predictions are sorted and the top N are returned
- **Content-based filtering:** for a selected book, a precomputed item-item similarity matrix (built during training, not at runtime) is used to find its nearest neighbors

Both modes join predictions back to a book metadata table to display title, author, year, publisher, and rating.

```

## Notes

- Built on a public book-ratings dataset (~50K users, ~100MB of ratings data) — this is a standard collaborative/content-based filtering exercise, not a novel modeling approach, and is intended as a portfolio/learning project demonstrating both major recommender system paradigms side by side rather than as a production-grade system
- No offline evaluation (RMSE, precision@k, etc.) is included in this app
