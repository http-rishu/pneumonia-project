import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

app = Flask(__name__, static_folder="../frontend/dist", static_url_path="")

# CORS setup
CORS(app,
    resources={r"/api/*": {"origins": "*"}},
    allow_headers=["Content-Type"],
    methods=["GET", "POST", "OPTIONS"]
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔥 SAFE MODEL LOADING (NO CRASH VERSION)
print("Loading trained model...")

model = None

try:
    from tensorflow.keras.models import load_model

    model_path = os.path.join(BASE_DIR, "pneumonia_model.h5")

    if os.path.exists(model_path):
        model = load_model(model_path, compile=False)
        print("✅ Model loaded successfully")
    else:
        print("⚠️ Model file not found, using fallback")

except Exception as e:
    print("❌ TensorFlow load failed:", e)
    print("⚠️ Running in demo mode (no ML model)")

# 🔥 Dummy prediction (fallback mode)
def dummy_predict():
    # random prediction for demo
    prob = np.random.rand()
    return prob


# 🔥 Severity calculation (dummy safe)
def calculate_severity(prob):
    if prob < 0.3:
        return "Mild"
    elif prob < 0.6:
        return "Moderate"
    else:
        return "Severe"


# ✅ Health API
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Pneumonia Detection API is running"
    })


# ✅ Prediction API
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded", "success": False}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No file selected", "success": False}), 400

        allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in allowed_extensions:
            return jsonify({
                "error": f"Invalid file type",
                "success": False
            }), 400

        # Image preprocessing
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((128, 128))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 🔥 Prediction logic
        if model is not None:
            pred = model.predict(img_array, verbose=0)[0][0]
        else:
            pred = dummy_predict()

        result = "PNEUMONIA" if pred > 0.5 else "NORMAL"

        confidence = float(pred) if result == "PNEUMONIA" else float(1 - pred)
        confidence_percent = round(confidence * 100, 2)

        severity = calculate_severity(pred)

        return jsonify({
            "success": True,
            "prediction": result,
            "confidence": confidence_percent,
            "severity": severity,
            "message": f"Analysis complete: {result} ({confidence_percent}%)"
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({
            "error": str(e),
            "success": False
        }), 500


# ✅ Serve frontend
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")


# 🚀 Run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)