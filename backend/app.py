# backend/app.py
# Flask backend for Health-Bot: triage + TF-IDF FAQ retriever integration
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os

# Try to import the retriever; if import fails, we'll fallback to a minimal reply
try:
    from utils import retrieve, build_tfidf_index  # utils.py lives in same backend/ folder
    RETRIEVER_AVAILABLE = True
except Exception as e:
    RETRIEVER_AVAILABLE = False
    retrieve = None
    build_tfidf_index = None
    missing_retriever_error = str(e)

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

def contains_red_flag(text: str) -> bool:
    txt = (text or "").lower()
    for rf in RED_FLAGS:
        if rf in txt:
            return True
    return False

@app.route("/health_check", methods=["GET"])
def health_check():
    info = {"status": "ok", "retriever": RETRIEVER_AVAILABLE}
    if not RETRIEVER_AVAILABLE:
        info["retriever_error"] = missing_retriever_error
    return jsonify(info), 200

@app.route("/build_index", methods=["POST"])
def build_index():
    """
    Optional helper endpoint to (re)build TF-IDF index from the FAQ JSON.
    Useful if you edit the FAQ and want to rebuild without SSH/terminal access.
    """
    if not RETRIEVER_AVAILABLE or build_tfidf_index is None:
        return jsonify({"status":"error","message":"Retriever not available on server"}), 500
    try:
        # build and save index files (this uses utils.build_tfidf_index)
        build_tfidf_index(save=True)
        return jsonify({"status":"ok","message":"Index built"}), 200
    except Exception as e:
        logger.exception("Failed to build index")
        return jsonify({"status":"error","message":str(e)}), 500

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

    # 2) TF-IDF retriever (if available)
    if RETRIEVER_AVAILABLE and retrieve is not None:
        try:
            answer, score, source = retrieve(user_msg, top_k=1, min_score=0.2)
            if answer:
                logger.info(f"TF-IDF matched (score={score:.3f}) for {user_id}")
                return jsonify({
                    "reply": answer,
                    "escalate": False,
                    "confidence": float(score),
                    "source": source
                }), 200
        except Exception as e:
            logger.warning(f"Retriever error: {e}")

    # 3) Fallback safe response (or a friendly default if retriever missing)
    fallback = (
        "I'm not certain about that. I can provide general information, but I'm not a medical professional. "
        "If this feels urgent, please contact a healthcare provider."
    )

    # If retriever is not available, include a friendly diagnostic hint in logs
    if not RETRIEVER_AVAILABLE:
        logger.warning(f"Retriever unavailable: {missing_retriever_error}")

    return jsonify({
        "reply": fallback,
        "escalate": False,
        "confidence": 0.0,
        "source": "fallback"
    }), 200

if __name__ == "__main__":
    # Use 0.0.0.0 for easier local testing on other devices; change debug for prod
    app.run(host="0.0.0.0", port=5000, debug=True)
