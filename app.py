import os
import sys
import pickle
import streamlit as st
import numpy as np
import logging
import logging as logger

from books_recommender.logger.log import logger
from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException


# ---------------- STREAMLIT CACHE ----------------
@st.cache_resource
def load_pickle(path):
    return pickle.load(open(path, "rb"))


class Recommendation:
    def __init__(self, app_config=AppConfiguration()):
        try:
            self.recommendation_config = app_config.get_recommendation_config()

            # Load once (performance fix)
            self.model = load_pickle(self.recommendation_config.trained_model_path)
            self.book_pivot = load_pickle(
                self.recommendation_config.book_pivot_serialized_objects
            )
            self.final_rating = load_pickle(
                self.recommendation_config.final_rating_serialized_objects
            )

        except Exception as e:
            raise AppException(e, sys) from e

    # ---------------- POSTER FETCH ----------------
    def fetch_poster(self, suggestion):
        try:
            poster_url = []

            for book_id in suggestion[0]:
                book_name = self.book_pivot.index[book_id]
                matches = np.where(self.final_rating["title"] == book_name)[0]

                if len(matches) > 0:
                    poster_url.append(
                        self.final_rating.iloc[matches[0]]["image_url"]
                    )
                else:
                    poster_url.append(
                        "https://via.placeholder.com/150?text=No+Image"
                    )

            return poster_url

        except Exception as e:
            raise AppException(e, sys) from e

    # ---------------- RECOMMEND BOOK ----------------
    def recommend_book(self, book_name):
        try:
            books_list = []

            book_id = np.where(self.book_pivot.index == book_name)[0][0]

            _, suggestion = self.model.kneighbors(
                self.book_pivot.iloc[book_id].values.reshape(1, -1),
                n_neighbors=6,
            )

            poster_url = self.fetch_poster(suggestion)

            # Skip the first (same book)
            for i in suggestion[0][1:]:
                books_list.append(self.book_pivot.index[i])

            return books_list, poster_url[1:]

        except Exception as e:
            raise AppException(e, sys) from e

    # ---------------- TRAIN ENGINE ----------------
    def train_engine(self):
        try:
            with st.spinner("Training model... Please wait"):
                obj = TrainingPipeline()
                obj.start_training_pipeline()

            st.success("Training Completed Successfully!")
            logging.info("Training completed successfully")

        except Exception as e:
            raise AppException(e, sys) from e

    # ---------------- STREAMLIT UI ----------------
    def recommendations_engine(self, selected_books):
        try:
            recommended_books, poster_url = self.recommend_book(selected_books)

            cols = st.columns(5)

            for i, col in enumerate(cols):
                if i < len(recommended_books):
                    col.text(recommended_books[i])
                    col.image(poster_url[i])

        except Exception as e:
            raise AppException(e, sys) from e


# ---------------- MAIN APP ----------------
if __name__ == "__main__":
    st.set_page_config(page_title="Books Recommender", layout="wide")

    st.header("📚 End to End Books Recommender System")
    st.write("Collaborative Filtering Based Recommendation System")

    obj = Recommendation()

    # TRAINING BUTTON
    if st.button("Train Recommender System"):
        obj.train_engine()

    # LOAD BOOK LIST
    book_names = load_pickle(
        os.path.join(os.getcwd(), "templates", "book_names.pkl")
    )

    selected_books = st.selectbox(
        "Type or select a book from the dropdown",
        book_names,
    )

    # SHOW RECOMMENDATION
    if st.button("Show Recommendation"):
        obj.recommendations_engine(selected_books)
