# backend/evaluation.py
"""
Evaluation script for the TF-IDF retriever.

Expected input: CSV with columns:
  - user_input
  - correct_answer

Usage:
  python backend/evaluation.py --input data/test_queries.csv --threshold 0.2
"""

import argparse
import pandas as pd
import re
from utils import retrieve  # assumes backend/ is your working folder or package importable
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()

def is_match(true_ans: str, pred_ans: str) -> bool:
    """
    Simple overlap-based match.
    Returns True if normalized true answer is substring of normalized pred, or vice-versa.
    You can replace with stricter or softer similarity metrics later.
    """
    t = normalize(true_ans)
    p = normalize(pred_ans)
    if not t or not p:
        return False
    return (t in p) or (p in t)

def evaluate(df: pd.DataFrame, threshold: float = 0.2):

    # Basic checks

    if "user_input" not in df.columns or "correct_answer" not in df.columns:
        raise ValueError("Input CSV must contain 'user_input' and 'correct_answer' columns.")

    y_true_texts = df["correct_answer"].fillna("").astype(str).tolist()
    queries = df["user_input"].fillna("").astype(str).tolist()

    y_pred_texts = []
    y_pred_score = []
    predicted_positive_flags = []  # whether model returned something above threshold
    matched_flags = []  # whether predicted answer matches the true answer (our definition)

    for q, true_ans in zip(queries, y_true_texts):
        ans, score, _source = retrieve(q, top_k=1, min_score=0.0)  # retrieve() returns (answer or None, score, source)
        if ans is None:
            ans = ""
        y_pred_texts.append(ans)
        y_pred_score.append(float(score))

        pred_positive = float(score) >= threshold
        predicted_positive_flags.append(int(pred_positive))

        matched = is_match(true_ans, ans)
        matched_flags.append(int(matched))

    # Exact-match style accuracy (using our is_match logic)
    accuracy = float(np.mean(matched_flags))

    # Compute precision/recall/f1 for the retrieval decision:
    # - predicted_positive_flags are what the model says it 'confidently' returned
    # - matched_flags indicate true positives among those predictions
    
    tp = sum(1 for p, m in zip(predicted_positive_flags, matched_flags) if p == 1 and m == 1)
    predicted_pos_count = sum(predicted_positive_flags)
    actual_pos_count = len(matched_flags)  # assuming every row has a true answer we want to retrieve

    precision = tp / predicted_pos_count if predicted_pos_count > 0 else 0.0
    recall = tp / actual_pos_count if actual_pos_count > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Extra stats
    avg_score_matched = np.mean([s for s, m in zip(y_pred_score, matched_flags) if m]) if any(matched_flags) else 0.0
    avg_score_unmatched = np.mean([s for s, m in zip(y_pred_score, matched_flags) if not m]) if any(1 - np.array(matched_flags)) else 0.0

    results = {
        "n_examples": len(queries),
        "accuracy_match": accuracy,
        "predicted_positive_count": int(predicted_pos_count),
        "true_positives (TP)": int(tp),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_score_matched": float(avg_score_matched),
        "avg_score_unmatched": float(avg_score_unmatched)
    }

    return results, y_pred_texts, y_pred_score, matched_flags, predicted_positive_flags

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/test_queries.csv", help="Path to test CSV")
    parser.add_argument("--threshold", type=float, default=0.2, help="Score threshold to consider a prediction 'positive'")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    results, preds, scores, matched_flags, pred_pos = evaluate(df, threshold=args.threshold)

    print("Evaluation Results:")
    print("-------------------")
    print(f"Examples: {results['n_examples']}")
    print(f"Accuracy (match fraction): {results['accuracy_match']:.4f}")
    print(f"Predicted positives (score >= {args.threshold}): {results['predicted_positive_count']}")
    print(f"True Positives (TP): {results['true_positives (TP)']}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Avg score (matched examples): {results['avg_score_matched']:.4f}")
    print(f"Avg score (unmatched examples): {results['avg_score_unmatched']:.4f}")

    # Optional: save a results CSV for manual inspection
    out_df = pd.DataFrame({
        "user_input": df["user_input"].astype(str),
        "true_answer": df["correct_answer"].astype(str),
        "predicted_answer": preds,
        "pred_score": scores,
        "is_matched": matched_flags,
        "pred_positive": pred_pos
    })
    out_df.to_csv("backend/eval_predictions.csv", index=False)
    print("\nSaved detailed predictions to backend/eval_predictions.csv")

if __name__ == "__main__":
    main()
