"""
data.py
-------
Sample course and rating data, plus train/test split utilities.
Replace COURSES and RATINGS with your own data — keep the same column names.
"""
 #MAKE SURE THERE ARE NO DUPLICATES IN DATAFRAME AS THAT IS MESSING UP RECOMMENDATIONS AND COSINE SIMILARITY
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

 
def load_courses_from_file(filepath: str) -> pd.DataFrame:
    """
    Loads and parses the course text file into a DataFrame
    compatible with the recommender system.

    Arguments:
        filepath : path to CourseText.txt
    Returns:
        DataFrame with columns [course_id, department, id_num,
                                title, tags, category, difficulty, rating]
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    df = pd.DataFrame(lines, columns=["raw"])
    df[['Department', 'ID']] = df['raw'].str.split("-", n=1, expand=True)
    df[['ID', 'Title']]      = df['ID'].str.split(" ", n=1, expand=True)
    df['Title']              = df['Title'].str.strip("\n")
    df['Department']         = df['Department'].str.strip()
    df['ID']                 = df['ID'].str.strip()
    df = df.drop(columns=['raw']).reset_index(drop=True)

    df = df.rename(columns={'Title': 'title'})
    df['department'] = df['Department']
    df['id_num'] = df['ID']

    df = df.drop(columns=['Department', 'ID'])

    # ─────────────────────────────────────────────
    # 4. Stable course_id (CRITICAL FIX)
    # ─────────────────────────────────────────────
    df['course_key'] = df['department'].astype(str) + "_" + df['id_num'].astype(str)
    df['course_id'] = pd.factorize(df['course_key'])[0] + 1
    df = df.drop(columns=['course_key'])

    # Use Department as category
    df['category'] = df['department']

    # Generate tags from title + department words for content-based filtering
    df['tags'] = (
        df['title'].str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True)
        + " "
        + df['department'].str.lower()
    )

    # Placeholder difficulty and rating — replace with real data if you have it
    df['difficulty'] = 'unknown'
    df['rating']     = 3.0

    return df[
        ['course_id', 'title', 'department', 'id_num',
               'tags', 'category', 'difficulty', 'rating']
        
    ]
    


# ── Load real courses ────────────────────────────────────────
COURSES = load_courses_from_file("CourseText.txt")

# ── Rename 'title' column to match recommender expectations ──
# (already named 'title' above)

# ── Synthetic ratings until you have real ones ───────────────
np.random.seed(42)
_N_USERS, _N_RATINGS = 50, 500
RATINGS = pd.DataFrame({
    "user_id":   np.random.randint(1, _N_USERS + 1, _N_RATINGS),
    "course_id": np.random.randint(1, len(COURSES) + 1, _N_RATINGS),
    "rating":    np.random.choice([1, 2, 3, 4, 5], _N_RATINGS,
                                  p=[0.05, 0.10, 0.20, 0.35, 0.30]),
}).drop_duplicates(["user_id", "course_id"]).reset_index(drop=True)


def split_ratings(ratings, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(ratings, test_size=test_size,
                                   random_state=random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True)