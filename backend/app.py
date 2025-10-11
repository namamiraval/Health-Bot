# backend/app.py
# Flask backend for Health-Bot: triage + simple FAQ-retriever placeholder
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
import os

app = Flask(__name__)
CORS(app)

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("healthbot")

# --- Red-flag keywords (expand as needed) ---
RED_FLAGS = [
    "chest pain", "difficulty breathing", "shortness of breath",
    "severe bleeding", "unconscious", "loss of consciousness",
    "sudden weakness", "slurred speech", "suicidal", "self harm", "overdose"
]

# --- Load small FAQ file if available ---
FAQ = []
FAQ_PATH = os.path.join(os.path.dirname(__file__), "sample_data", "curated_faq.json")
if os.path.exists(FAQ_PATH):
    try:
        with open(FAQ_PATH, "r", encoding="utf-8") as f:
            FAQ = json.load(f)
            logger.info(f"Loaded {len(FAQ)} FAQ entries from {FAQ_PATH}")
    except Exception as e:
        logger.warning(f"Failed to load FAQ file: {e}")

def contains_red_flag(text: str) -> bool:
    """
    Simple substring-based red-flag detector.
    Expand with smarter NLP rules or regex as needed.
    """
    txt = (text or "").lower()
    for rf in RED_FLAGS:
        if rf in txt:
            return True
    return False

def naive_faq_lookup(text: str):
    """
    Very simple FAQ lookup: returns first FAQ whose question
    is a substring of the user text. Replace later with
    embedding or TF-IDF retrieval.
    """
    txt = (text or "").lower()
    for qa in FAQ:
        q = qa.get("question", "").lower()
        if q and q in txt:
            return qa.get("answer"), 0.9
    return None, 0.0

@app.route("/health_check", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    user_msg = data.get("message", "").strip()
    user_id = data.get("user_id", "anon")

    logger.info(f"Received message from {user_id}: {user_msg}")

    if not user_msg:
        return jsonify({
            "reply": "I didn't catch that — can you type your message again?",
            "escalate": False,
            "confidence": 0.0,
            "source": "validation"
        }), 200

    # 1) Triage: urgent red-flag detection
    if contains_red_flag(user_msg):
        reply = (
            "I may be mistaken, but your message contains signs of a potentially serious condition. "
            "Please call your local emergency services or go to the nearest hospital right away."
        )
        logger.info(f"Escalation triggered for {user_id}")
        return jsonify({
            "reply": reply,
            "escalate": True,
            "confidence": 1.0,
            "source": "triage"
        }), 200

    # 2) Naive FAQ matching (placeholder)
    answer, conf = naive_faq_lookup(user_msg)
    if answer:
        return jsonify({
            "reply": answer,
            "escalate": False,
            "confidence": conf,
            "source": "faq"
        }), 200

    # 3) Fallback safe response
    fallback = (
        "I'm not certain about that. I can provide general information, but I'm not a medical professional. "
        "If this feels urgent, please contact a healthcare provider."
    )
    return jsonify({
        "reply": fallback,
        "escalate": False,
        "confidence": 0.0,
        "source": "fallback"
    }), 200

if __name__ == "__main__":
    # Use 0.0.0.0 for easier local testing on other devices; change debug for prod
    app.run(host="0.0.0.0", port=5000, debug=True)
