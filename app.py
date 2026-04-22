"""
app.py
------
Backend for the Course Recommendation System.

To run:
    make sure you have flask, so do pip install flask
    then you can run python app.py in the terminal
    let it load, then it will say running on "LINK"
    click the link, when you want to exit leave tab and do CTRL+C to quit

"""

import os
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data        import COURSES, RATINGS, split_ratings
from recommender import ContentBasedRecommender, CollaborativeRecommender
from evaluate    import evaluate_rmse, cross_validate_cf, compare_k_values

app = Flask(__name__)

# Train models once on startup 
print("Loading data and training models…")
train_ratings, test_ratings = split_ratings(RATINGS, test_size=0.2)
cb = ContentBasedRecommender(COURSES)
cf = CollaborativeRecommender(train_ratings, k=20)
print(f"Ready — {len(COURSES)} courses | {RATINGS['user_id'].nunique()} users")


# Frontend 
@app.route("/")
def index():
    return send_from_directory(".", "UI.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)


# API: stats
@app.route("/api/stats")
def stats():
    return jsonify({
        "num_courses": len(COURSES),
        "num_ratings": len(RATINGS),
        "num_users":   int(RATINGS["user_id"].nunique()),
    })


# Search courses
@app.route("/api/search")
def search():
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify([])
    
    # cases do not need to match between search and list
    mask = (
        COURSES["title"].str.lower().str.contains(q, na=False) |
        COURSES["department"].str.lower().str.contains(q, na=False)
    )
    
    results = COURSES[mask][["course_id", "title", "department"]].head(10)
    
    return jsonify(results.to_dict(orient="records"))

# API: recommend by course
@app.route("/api/recommend")
def recommend():
    #  int conversion
    try:
        course_id = int(request.args.get("course_id"))
    except:
        return jsonify({"error": "Invalid ID"}), 400

    # Find the seed course in your DataFrame
    row = COURSES[COURSES["course_id"] == course_id]
    if row.empty:
        return jsonify({"error": "Not found"}), 404

    seed_row = row.iloc[0]
    seed_title = seed_row["title"]
    seed_dept = seed_row["department"]

    cb_recs_df = cb.recommend(course_id=course_id, top_n=5)
    cb_recs = cb_recs_df.to_dict(orient="records")
    depts = [d.strip() for d in seed_dept.split("/")]
    cluster_df = COURSES[
        COURSES["department"].apply(lambda d: any(dep in d for dep in depts)) &
        (COURSES["course_id"] != course_id)
    ][["course_id", "title", "department"]].head(8)
    
    cluster_recs = cluster_df.to_dict(orient="records")

    return jsonify({
        "seed_id": course_id,
        "seed_title": seed_title,
        "seed_dept": seed_dept,
        "content_based": cb_recs,
        "dept_cluster": cluster_recs
    })

# evaluating
@app.route("/api/evaluate")
def evaluate():
    train_rmse = float(evaluate_rmse(cf, train_ratings))
    test_rmse  = float(evaluate_rmse(cf, test_ratings))

    fold_rmses_raw = cross_validate_cf(RATINGS, k_factors=20, n_folds=5, verbose=False)
    fold_rmses = [{"fold": i + 1, "rmse": round(float(r), 4)}
                  for i, r in enumerate(fold_rmses_raw)]

    k_df   = compare_k_values(RATINGS, train_ratings, test_ratings, k_values=[5, 10, 20, 30])
    k_rows = k_df.rename(columns={"k_factors": "k"}).round(4).to_dict(orient="records")

    return jsonify({
        "train_rmse":       round(train_rmse, 4),
        "test_rmse":        round(test_rmse,  4),
        "cross_validation": {
            "folds": fold_rmses,
            "mean":  round(float(np.mean(fold_rmses_raw)), 4),
            "std":   round(float(np.std(fold_rmses_raw)),  4),
        },
        "k_comparison": k_rows,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)