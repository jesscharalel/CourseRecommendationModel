"""
models.py
---------
Three recommender classes:
  - ContentBasedRecommender  : TF-IDF cosine similarity on course tags
  - CollaborativeRecommender : SVD matrix factorisation (NumPy/SciPy)
"""
 
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
 
from data import COURSES
 
#make sure there are no repeats with cross-listed classes, and list classes with multiple departments
# ─────────────────────────────────────────────
# 1. CONTENT-BASED FILTERING
# ─────────────────────────────────────────────
 
class ContentBasedRecommender:
    """
    Recommends courses by computing TF-IDF cosine similarity
    between course tag strings.
    """
 
    def __init__(self, courses: pd.DataFrame):
        """
        Builds TF-IDF matrix and pairwise similarity matrix.
 
        Arguments:
            courses : DataFrame with at minimum columns
                      [course_id, tags, title, category, difficulty, rating]
        """
        self.courses = courses.copy().set_index("course_id")
        tfidf        = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(self.courses["tags"])
        self.sim_matrix = cosine_similarity(tfidf_matrix)
        self.idx_map    = {cid: i for i, cid in enumerate(self.courses.index)}
 
    def recommend(self, course_id: int, top_n: int = 5) -> pd.DataFrame:
        """
        Returns the top_n courses most similar to course_id.
 
        Arguments:
            course_id : ID of the seed course
            top_n     : number of results to return
        Returns:
            DataFrame with columns [course_id, title, category,
                                    difficulty, rating, content_score]
        """
        idx    = self.idx_map[course_id]
        sims   = list(enumerate(self.sim_matrix[idx]))
        sims   = sorted(sims, key=lambda x: x[1], reverse=True)[1:]
        top    = [self.courses.index[i] for i, _ in sims[:top_n]]
        scores = [round(s, 4) for _, s in sims[:top_n]]
        result = self.courses.loc[top, ["title", "category",
                                        "difficulty", "rating"]].copy()
        result["content_score"] = scores
        return result.reset_index()
 
 
# ─────────────────────────────────────────────
# 2. COLLABORATIVE FILTERING  (SVD)
# ─────────────────────────────────────────────
 
class CollaborativeRecommender:
    """
    Recommends courses using truncated SVD matrix factorisation.
    Trains on the provided ratings DataFrame; call evaluate_rmse()
    with a separate test set for an honest performance estimate.
    """
 
    def __init__(self, ratings: pd.DataFrame, k: int = 20):
        """
        Builds and decomposes the user-course rating matrix.
 
        Arguments:
            ratings : DataFrame with columns [user_id, course_id, rating]
            k       : number of latent factors for SVD (default 20)
        """
        self.ratings  = ratings
        self.all_cids = sorted(ratings["course_id"].unique())
        self.all_uids = sorted(ratings["user_id"].unique())
        self.uid2i    = {u: i for i, u in enumerate(self.all_uids)}
        self.cid2j    = {c: j for j, c in enumerate(self.all_cids)}
 
        n_u, n_c = len(self.all_uids), len(self.all_cids)
        R = np.zeros((n_u, n_c))
        for _, row in ratings.iterrows():
            R[self.uid2i[row.user_id],
              self.cid2j[row.course_id]] = row.rating
 
        # Mean-centre each user row before decomposition
        counts = (R != 0).sum(axis=1).clip(min=1)
        self.row_means = R.sum(axis=1) / counts
        R_centred = R.copy()
        for i in range(n_u):
            R_centred[i, R[i] != 0] -= self.row_means[i]
 
        k = min(k, min(R.shape) - 1)
        U, sigma, Vt = svds(R_centred, k=k)
        self.R_pred = np.clip(
            np.dot(np.dot(U, np.diag(sigma)), Vt)
            + self.row_means[:, np.newaxis],
            1, 5,
        )
 
    def predict_rating(self, user_id: int, course_id: int) -> float:
        """
        Predicts the rating user_id would give course_id.
 
        Returns 3.0 (neutral default) for users or courses not
        seen during training.
        """
        if user_id not in self.uid2i or course_id not in self.cid2j:
            return 3.0
        return float(self.R_pred[self.uid2i[user_id],
                                  self.cid2j[course_id]])
 
    def evaluate_rmse(self, test_ratings: pd.DataFrame) -> float:
        """
        Computes RMSE on a held-out test set.
 
        Arguments:
            test_ratings : DataFrame with columns [user_id, course_id, rating]
                           Must NOT overlap with training data.
        Returns:
            RMSE as a float
        """
        errors = [
            (self.predict_rating(r.user_id, r.course_id) - r.rating) ** 2
            for _, r in test_ratings.iterrows()
        ]
        return round(float(np.sqrt(np.mean(errors))), 4)
 
    def recommend(self, user_id: int, top_n: int = 5,
                  already_rated: set = None) -> pd.DataFrame:
        """
        Returns top_n course recommendations for user_id.
 
        Arguments:
            user_id       : target user
            top_n         : number of results
            already_rated : set of course_ids to exclude
        Returns:
            DataFrame with columns [course_id, title, category,
                                    difficulty, rating, predicted_rating]
        """
        already_rated = already_rated or set()
        candidates = [
            (cid, self.predict_rating(user_id, cid))
            for cid in self.all_cids if cid not in already_rated
        ]
        #["Department", "ID", "Title", "category", "difficulty", "rating"]
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_cids   = [cid for cid, _ in candidates[:top_n]]
        top_scores = [round(s, 4) for _, s in candidates[:top_n]]
        result = COURSES.set_index("course_id").loc[
            top_cids, ["title", "category", "difficulty", "rating"]
        ].copy()
        result["predicted_rating"] = top_scores
        return result.reset_index()
 