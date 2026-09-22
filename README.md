# Book Recommendation System

**Live App:** [Add your Streamlit link here]

A Streamlit dashboard offering two classic, genuinely different approaches to recommending books: one based on what similar readers liked, and one based on how similar a book is to another book. Most real-world recommendation systems (Netflix, Amazon, Spotify) use some blend of both ideas — this app demonstrates each one separately so the difference is easy to see.

## Features

- **Collaborative filtering ("Recommended by a critic")** — predicts ratings for books a selected user hasn't rated yet, based on patterns in how users with similar taste have rated other books, and returns the top-N highest-predicted books
- **Content-based filtering ("Recommend Similar Books")** — returns the books most similar to a chosen book, based on the books' own attributes rather than any user's rating history
- **Adjustable output** — recommendation count is configurable (5–15) in both modes
- **Readable results** — recommendations are displayed with title, author, year, publisher, and average rating, styled as cards

## Tech Stack, and why each piece is here

- **scikit-surprise** — a Python library built specifically for recommendation systems. It provides the SVD (Singular Value Decomposition) algorithm used here for collaborative filtering, which is a well-established technique for the "which users like which items" problem — famously the approach that won the original Netflix Prize competition.
- **pandas** — used to load and join the tabular data: ratings, book metadata, and the user-name-to-ID lookup table.
- **pickle** — Python's standard way of saving a trained model or precomputed object to disk and loading it back later, so the app doesn't need to retrain anything every time it starts.
- **Streamlit** — builds the dashboard interface, with `@st.cache_data` used to load the (fairly large) dataset once and keep it in memory rather than reloading it on every interaction.

## How It Works, step by step

**Collaborative filtering (the "critic" mode):**
1. The idea behind collaborative filtering is that if two readers have historically rated many books similarly, they're likely to agree on books neither has read yet.
2. The SVD model was trained beforehand on the full ratings dataset, learning hidden patterns in who rates what highly.
3. When you select a user, the app looks at every book that user *hasn't* rated yet, and asks the trained model to predict what rating they *would* give it.
4. Those predicted ratings are sorted from highest to lowest, and the top N books are shown as recommendations.

**Content-based filtering (the "similar books" mode):**
1. This mode ignores user rating history entirely and instead asks: "based on the book's own attributes, which other books are most like this one?"
2. This relies on a similarity matrix that was precomputed ahead of time — essentially a table that stores, for every pair of books, a score for how alike they are.
3. When you select a book, the app looks up its row in that matrix, sorts all other books by similarity score, and returns the top N.

Both modes finish the same way: the recommended book IDs are matched against a metadata table to pull in the title, author, year, publisher, and average rating for display.


