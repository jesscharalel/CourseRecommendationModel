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
from flask import Flask, jsonify, request, send_from_directory

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data import COURSES
from recommender import ContentBasedRecommender

app = Flask(__name__)

# Train model once
print("Loading data and training content-based model…")
cb = ContentBasedRecommender(COURSES)
print(f"Ready — {len(COURSES)} courses")


# Frontend
@app.route("/")
def index():
    return send_from_directory(".", "UI.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)


# Stats
@app.route("/api/stats")
def stats():
    return jsonify({
        "num_courses": len(COURSES)
    })


# Search courses
@app.route("/api/search")
def search():
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify([])
    
    mask = (
        COURSES["title"].str.lower().str.contains(q, na=False) |
        COURSES["department"].str.lower().str.contains(q, na=False)
    )
    
    results = COURSES[mask][["course_id", "title", "department"]].head(10)
    return jsonify(results.to_dict(orient="records"))


# Recommend (Content-Based ONLY)
@app.route("/api/recommend")
def recommend():
    try:
        course_id = int(request.args.get("course_id"))
    except:
        return jsonify({"error": "Invalid ID"}), 400

    row = COURSES[COURSES["course_id"] == course_id]
    if row.empty:
        return jsonify({"error": "Not found"}), 404

    seed_row = row.iloc[0]

    cb_recs_df = cb.recommend(course_id=course_id, top_n=5)
    cb_recs = cb_recs_df.to_dict(orient="records")

    return jsonify({
        "seed_id": course_id,
        "seed_title": seed_row["title"],
        "seed_dept": seed_row["department"],
        "content_based": cb_recs
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
