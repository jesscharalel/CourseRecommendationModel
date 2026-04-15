"""
evaluate.py
-----------
Honest evaluation utilities:
  - evaluate_rmse()        : test-set RMSE for a trained CF model
  - cross_validate_cf()    : k-fold cross-validation
  - compare_k_values()     : sweep over different k (neighbour counts)
"""
 
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
 
from recommender import CollaborativeRecommender
 
 
def evaluate_rmse(cf: CollaborativeRecommender,
                  test_ratings: pd.DataFrame) -> float:
    """
    Computes RMSE on a held-out test set.
 
    The test set must NOT have been used to train the model —
    this is the honest, out-of-sample error estimate.
 
    Arguments:
        cf           : a trained CollaborativeRecommender
        test_ratings : DataFrame with columns [user_id, course_id, rating]
    Returns:
        RMSE as a float
    """
    return cf.evaluate_rmse(test_ratings)
 
 
def cross_validate_cf(ratings: pd.DataFrame,
                      k_factors: int = 20,
                      n_folds: int = 5,
                      verbose: bool = True) -> list:
    """
    Runs k-fold cross-validation on the collaborative filtering model.
 
    Splits ratings into n_folds folds. For each fold, trains on the
    remaining folds and evaluates RMSE on the held-out fold. Reports
    per-fold and average RMSE.
 
    Arguments:
        ratings   : full ratings DataFrame [user_id, course_id, rating]
        k_factors : number of SVD latent factors (default 20)
        n_folds   : number of cross-validation folds (default 5)
        verbose   : whether to print fold-by-fold results
    Returns:
        list of per-fold RMSE values
    """
    kf    = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    rmses = []
 
    if verbose:
        print(f"\n{n_folds}-Fold Cross-Validation (k_factors={k_factors})")
        print("─" * 40)
 
    for fold, (train_idx, test_idx) in enumerate(kf.split(ratings)):
        train = ratings.iloc[train_idx].reset_index(drop=True)
        test  = ratings.iloc[test_idx].reset_index(drop=True)
 
        cf         = CollaborativeRecommender(train, k=k_factors)
        fold_rmse  = cf.evaluate_rmse(test)
        rmses.append(fold_rmse)
 
        if verbose:
            print(f"  Fold {fold + 1}/{n_folds}:  RMSE = {fold_rmse}")
 
    avg = round(float(np.mean(rmses)), 4)
    if verbose:
        print(f"  {'─'*30}")
        print(f"  Average RMSE:   {avg}")
 
    return rmses
 
 
def compare_k_values(ratings: pd.DataFrame,
                     train_ratings: pd.DataFrame,
                     test_ratings: pd.DataFrame,
                     k_values: list = None) -> pd.DataFrame:
    """
    Trains a CF model for each value of k_factors and reports
    train RMSE vs test RMSE, making overfitting visible.
 
    Arguments:
        ratings       : full ratings (used only for context)
        train_ratings : training split
        test_ratings  : test split
        k_values      : list of latent factor counts to try
                        (default [5, 10, 20, 30, 50])
    Returns:
        DataFrame with columns [k_factors, train_rmse, test_rmse]
    """
    if k_values is None:
        k_values = [5, 10, 20, 30, 50]
 
    print("\nComparing k_factors values")
    print("─" * 45)
    print(f"  {'k_factors':>10}  {'train_rmse':>12}  {'test_rmse':>12}")
    print(f"  {'─'*10}  {'─'*12}  {'─'*12}")
 
    rows = []
    for k in k_values:
        cf         = CollaborativeRecommender(train_ratings, k=k)
        train_rmse = cf.evaluate_rmse(train_ratings)
        test_rmse  = cf.evaluate_rmse(test_ratings)
        rows.append({"k_factors": k,
                     "train_rmse": train_rmse,
                     "test_rmse": test_rmse})
        print(f"  {k:>10}  {train_rmse:>12}  {test_rmse:>12}")
 
    return pd.DataFrame(rows)