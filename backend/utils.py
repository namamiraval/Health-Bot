# Utilities: text cleaning, TF-IDF retriever build/load, and retrieval helper
import json
import os
import re
import pickle
from typing import Tuple, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(__file__)
FAQ_PATH = os.path.join(BASE_DIR, "sample_data", "curated_faq.json")
INDEX_PATH = os.path.join(BASE_DIR, "tfidf_index.pkl")
VECT_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

def clean_text(s: str) -> str:
    """Basic normalisation: lowercase, remove non-alphanum (keep spaces), collapse spaces."""
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_faq(path: str = FAQ_PATH) -> List[dict]:
    """Load FAQ JSON file into a list of dicts with 'question' and 'answer' keys."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # normalize questions
    for qa in data:
        qa["question_clean"] = clean_text(qa.get("question", ""))
        qa["answer"] = qa.get("answer", "")
    return data

def build_tfidf_index(faq_path: str = FAQ_PATH, save: bool = True) -> Tuple[TfidfVectorizer, np.ndarray, List[dict]]:
    """
    Build and optionally save a TF-IDF vectorizer and matrix from the FAQ questions.
    Returns: (vectorizer, tfidf_matrix, faq_list)
    """
    faq_list = load_faq(faq_path)
    questions = [qa.get("question_clean", "") for qa in faq_list]
    if not questions:
        raise ValueError("No FAQ questions found to build TF-IDF index.")
    vect = TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=1)
    tfidf_matrix = vect.fit_transform(questions)
    if save:
        with open(VECT_PATH, "wb") as f:
            pickle.dump(vect, f)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(tfidf_matrix, f)
    return vect, tfidf_matrix, faq_list

def load_tfidf_index() -> Tuple[TfidfVectorizer, np.ndarray, List[dict]]:
    """Load saved vectorizer and matrix. If missing, build from FAQ_path."""
    if os.path.exists(VECT_PATH) and os.path.exists(INDEX_PATH):
        with open(VECT_PATH, "rb") as f:
            vect = pickle.load(f)
        with open(INDEX_PATH, "rb") as f:
            tfidf_matrix = pickle.load(f)
        faq_list = load_faq(FAQ_PATH)
        return vect, tfidf_matrix, faq_list
    # fallback: build
    return build_tfidf_index(FAQ_PATH, save=True)

def retrieve(query: str, top_k: int = 1, min_score: float = 0.2) -> Tuple[str, float, str]:
    """
    Retrieve best matching answer for the query.
    Returns: (answer_text or None, score (0-1), source key)
    source is 'tfidf' when matched, otherwise 'none'
    """
    vect, tfidf_matrix, faq_list = load_tfidf_index()
    q_clean = clean_text(query)
    q_vec = vect.transform([q_clean])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]  # shape (n_faq,)
    # get best index
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    if best_score >= min_score:
        answer = faq_list[best_idx].get("answer", "")
        return answer, best_score, "tfidf"
    return None, 0.0, "none"

if __name__ == "__main__":
    # quick CLI: python utils.py build  -> builds and saves index
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "build":
        print("Building TF-IDF index from", FAQ_PATH)
        build_tfidf_index(save=True)
        print("Saved vectorizer to", VECT_PATH)
        print("Saved index to", INDEX_PATH)
    else:
        print("Usage: python utils.py build")
