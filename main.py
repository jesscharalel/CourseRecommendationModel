"""
main.py
-------
Entry point for the course recommendation system.
 
Run with:
    python main.py
 
What it does:
  1. Loads data and performs an 80/20 train/test split
  2. Trains all three recommenders on TRAINING data only
  3. Evaluates the CF model on the held-out TEST set
  4. Runs 5-fold cross-validation
  5. Compares different k_factor values (train vs test RMSE)
  6. Prints example recommendations for a sample user
"""
 
from data    import COURSES, RATINGS, split_ratings
from recommender  import ContentBasedRecommender, CollaborativeRecommender
from evaluate import evaluate_rmse, cross_validate_cf, compare_k_values
 
SEP  = "─" * 62
SEP2 = "=" * 62
 
 
def main():
    print(SEP2)
    print("   Course Recommendation System")
    print(SEP2)
    print(f"\nFull dataset: {len(COURSES)} courses | "
          f"{len(RATINGS)} ratings | "
          f"{RATINGS['user_id'].nunique()} users")
 
    # ── 1. Train / Test Split ────────────────────────────────────
    print(f"\n{SEP}")
    print("Step 1 — Train / Test Split  (80% / 20%)")
    print(SEP)
    train_ratings, test_ratings = split_ratings(RATINGS, test_size=0.2)
    print(f"  Training ratings : {len(train_ratings)}")
    print(f"  Test ratings     : {len(test_ratings)}")
 
    # ── 2. Build Models (trained on TRAIN only) ──────────────────
    print(f"\n{SEP}")
    print("Step 2 — Training Models on Training Set")
    print(SEP)
 
    print("  Building content-based model …")
    cb = ContentBasedRecommender(COURSES)
 
    print("  Training collaborative filtering model (SVD, k=20) …")
    cf = CollaborativeRecommender(train_ratings, k=20)
 
    # ── 3. Honest Test-Set Evaluation ────────────────────────────
    print(f"\n{SEP}")
    print("Step 3 — Evaluation on Held-Out Test Set")
    print(SEP)
    test_rmse  = evaluate_rmse(cf, test_ratings)
    train_rmse = evaluate_rmse(cf, train_ratings)
    print(f"  Train RMSE : {train_rmse}  (expected to be low — model saw this data)")
    print(f"  Test  RMSE : {test_rmse}   (honest estimate — model never saw this)")
 
    # ── 4. Cross-Validation ──────────────────────────────────────
    cross_validate_cf(RATINGS, k_factors=20, n_folds=5)
 
    # ── 5. Compare k_factor Values ───────────────────────────────
    compare_k_values(RATINGS, train_ratings, test_ratings,
                     k_values=[5, 10, 20, 30])
 
    # ── 6. Example Recommendations ───────────────────────────────
    print(f"\n{SEP}")
    print("Step 6 — Example Recommendations for User 7")
    print(SEP)
 
    user_id       = 7
    already_rated = set(train_ratings[
        train_ratings["user_id"] == user_id
    ]["course_id"])
    print(f"  User {user_id} rated course IDs: {sorted(already_rated)}\n")
 
    print("  Content-Based (similar to 'The Deep History of Climate Adaptation', id=46):")
    cb_recs = cb.recommend(course_id=46, top_n=5)
    print(cb_recs[["course_id", "title", "content_score"]].to_string(index=False))
 
    print(f"\n  Collaborative Filtering:")
    cf_recs = cf.recommend(user_id=user_id, top_n=5, already_rated=already_rated)
    print(cf_recs[["course_id", "title", "predicted_rating"]].to_string(index=False))

 
    print(f"\n{SEP2}")
    print("  ✓ Done.")
    print(SEP2)
 
 
if __name__ == "__main__":
    main()