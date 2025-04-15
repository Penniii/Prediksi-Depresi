from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import requests

app = Flask(__name__)
CORS(app)

# ===========================
# LOAD MODEL DETEKSI DEPRESI
# ===========================
model = joblib.load("model_svm_final.pkl")

phq9_columns = [
    "1. In a semester, how often have you had little interest or pleasure in doing things?",
    "2. In a semester, how often have you been feeling down, depressed or hopeless?",
    "3. In a semester, how often have you had trouble falling or staying asleep, or sleeping too much? ",
    "4. In a semester, how often have you been feeling tired or having little energy? ",
    "5. In a semester, how often have you had poor appetite or overeating? ",
    "6. In a semester, how often have you been feeling bad about yourself - or that you are a failure or have let yourself or your family down? ",
    "7. In a semester, how often have you been having trouble concentrating on things, such as reading the books or watching television? ",
    "8. In a semester, how often have you moved or spoke too slowly for other people to notice? Or you've been moving a lot more than usual because you've been restless? ",
    "9. In a semester, how often have you had thoughts that you would be better off dead, or of hurting yourself? "
]

def get_depression_label(score):
    if score <= 4:
        return "minimal depression"
    elif score <= 9:
        return "mild depression"
    elif score <= 14:
        return "moderate depression"
    elif score <= 19:
        return "moderately severe depression"
    else:
        return "severe depression"

# ===========================
# ENDPOINT PREDIKSI DEPRESI
# ===========================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    phq9_scores = data.get("phq9_scores")

    if not phq9_scores or len(phq9_scores) != 9:
        return jsonify({"error": "Invalid input. Expected 9 PHQ-9 scores."}), 400

    total_score = sum(phq9_scores)
    prediction_label = get_depression_label(total_score)

    return jsonify({
        "total_score": int(total_score),
        "prediction": prediction_label
    })

# ===========================
# ENDPOINT CHATBOT AI
# ===========================
GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "Kamu adalah seorang AI bernama Penyy ramah dan empatik yang membantu mahasiswa memahami dan menghadapi masalah psikologis mereka, khususnya depresi. selalu balas dalam bahasa indonesia. "},
            {"role": "user", "content": user_message}
        ]
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=body)
        response.raise_for_status()
        result = response.json()
        ai_reply = result["choices"][0]["message"]["content"]
    except Exception as e:
        ai_reply = f"Maaf, terjadi kesalahan saat menghubungi AI: {str(e)}"

    return jsonify({"reply": ai_reply})

if __name__ == "__main__":
    app.run(debug=True)
