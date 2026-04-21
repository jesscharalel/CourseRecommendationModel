"""
data.py
-------
Sample course and rating data, plus train/test split utilities.
Replace COURSES and RATINGS with your own data — keep the same column names.
"""
#MAKE SURE THERE ARE NO DUPLICATES IN DATAFRAME AS THAT IS MESSING UP RECOMMENDATIONS AND COSINE SIMILARITY
#train and test on departments
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

GENERIC_WORDS = {
    'introduction', 'special', 'topics', 'seminar', 'directed',
    'study', 'research', 'independent', 'advanced', 'topics',
    'honors', 'thesis', 'internship', 'lab', 'laboratory',
    'workshop', 'practicum', 'selected', 'readings', 'junior',
    'senior', 'graduate', 'undergraduate', 'special', 'studies'
}

def make_tags(title: str, department: str) -> str:
    """
    Builds a tag string from title words, filtering out generic
    words that appear in hundreds of courses and would dilute
    cosine similarity toward meaningless matches.
    """
    words = title.lower().replace(r'[^a-z0-9 ]', ' ').split()
    meaningful = [w for w in words if w not in GENERIC_WORDS and len(w) > 2]
    # repeat dept name twice to give it more weight
    dept = department.lower()
    return " ".join(meaningful) + f" {dept} {dept}"



def load_courses_from_file(filepath: str) -> pd.DataFrame:
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
    df['id_num']     = df['ID']
    df = df.drop(columns=['Department', 'ID'])

    df['category']   = df['department']
    df['difficulty'] = 'unknown'
    df['rating']     = 3.0
    df['tags'] = df.apply(lambda r: make_tags(r['title'], r['department']), axis=1)

    # ── ADD DEDUPLICATION HERE ───────────────────────────────────
    print(f"  Raw rows before dedup: {len(df)}")

    deduped = (
        df.groupby("title", as_index=False)
          .agg(
              department = ("department", lambda x: " / ".join(sorted(x.unique()))),
              id_num     = ("id_num",     "first"),
              tags       = ("tags",       "first"),
              category   = ("category",   "first"),
              difficulty = ("difficulty", "first"),
              rating     = ("rating",     "first"),
          )
    )

    deduped.insert(0, "course_id", range(1, len(deduped) + 1))
    print(f"  Rows after dedup:      {len(deduped)}  "
          f"({len(df) - len(deduped)} cross-listed duplicates removed)")
    # ────────────────────────────────────────────────────────────

    return deduped[['course_id', 'title', 'department', 'id_num',
                    'tags', 'category', 'difficulty', 'rating']]


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

def audit_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds courses with identical titles listed under multiple departments.
    Prints a summary and returns the duplicate groups.
    """
    dupes = (
        df.groupby("title")
          .filter(lambda x: len(x) > 1)
          .sort_values("title")
    )
    print(f"  {dupes['title'].nunique()} unique titles appear "
          f"in multiple departments ({len(dupes)} total rows)")
    return dupes

