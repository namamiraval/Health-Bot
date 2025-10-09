# Simple Flask backend with triage checks and placeholder for NLU/FAQ
app = Flask(__name__)
CORS(app)


# Simple red-flag keywords (expand these to be comprehensive)
RED_FLAGS = [
'chest pain', 'difficulty breathing', 'shortness of breath',
'severe bleeding', 'unconscious', 'loss of consciousness',
'sudden weakness', 'slurred speech', 'suicidal', 'self harm', 'overdose'
]


# Load a small FAQ file if present
try:
with open('sample_data/curated_faq.json', 'r', encoding='utf-8') as f:
FAQ = json.load(f)
except Exception:
FAQ = []




def contains_red_flag(text: str) -> bool:
txt = text.lower()
for rf in RED_FLAGS:
if rf in txt:
return True
return False




@app.route('/health_check', methods=['GET'])
def health_check():
return jsonify({'status': 'ok'})




@app.route('/chat', methods=['POST'])
def chat():
data = request.get_json() or {}
user_msg = data.get('message', '')


# Basic safety triage
if contains_red_flag(user_msg):
reply = ("I may be wrong, but your message contains signs of a potentially serious condition. "
"Please call local emergency services or visit the nearest hospital immediately.")
return jsonify({
'reply': reply,
'escalate': True,
'confidence': 1.0,
'source': 'triage'
})


# Placeholder: NLU / FAQ retrieval
# TODO: implement embedding-based retrieval or fine-tuned QA model here
# For now, do a naive FAQ keyword match
for qa in FAQ:
q = qa.get('question','').lower()
if q and q in user_msg.lower():
return jsonify({
'reply': qa.get('answer',''),
'escalate': False,
'confidence': 0.9,
'source': 'faq'
})


# Fallback safe response
fallback = ("I'm not certain about that. I can provide general information, but I'm not a medical professional. "
"If you feel it's urgent, please contact a healthcare provider.")
return jsonify({
'reply': fallback,
'escalate': False,
'confidence': 0.0,
'source': 'fallback'
})


if __name__ == '__main__':
# Run in debug mode for local dev; change to production server for deployment
app.run(host='0.0.0.0', port=5000, debug=True)