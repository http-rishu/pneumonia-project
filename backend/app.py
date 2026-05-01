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

print("🚀 Starting Pneumonia API...")

# Disable model (safe deploy)
model = None

# Dummy prediction
def dummy_predict():
    return float(np.random.rand())

# Severity calculation
def calculate_severity(prob):
    if prob < 0.3:
        return "Mild"
    elif prob < 0.6:
        return "Moderate"
    else:
        return "Severe"

# Health API
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Pneumonia Detection API is running 🚀"
    })

# Prediction API
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
                "error": "Invalid file type",
                "success": False
            }), 400

        # Image processing
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((128, 128))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Dummy prediction
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
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

# Serve frontend
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")

# Run server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)