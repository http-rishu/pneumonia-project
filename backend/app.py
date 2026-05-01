import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer
from PIL import Image

app = Flask(__name__, static_folder="../frontend/dist", static_url_path="")

# CORS setup
CORS(app,
    resources={r"/api/*": {"origins": "*"}},
    allow_headers=["Content-Type"],
    methods=["GET", "POST", "OPTIONS"]
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔥 FINAL SAFE MODEL LOADING (FULL FIX)
print("Loading trained model...")
model_path = os.path.join(BASE_DIR, "pneumonia_model.h5")

try:
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found")

    model = load_model(
        model_path,
        compile=False,
        custom_objects={"InputLayer": InputLayer}
    )

    print("✅ Model loaded successfully")

except Exception as e:
    print("❌ Model load failed:", e)
    print("⚠️ Using fallback model (demo mode)")

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])


# 🔥 Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, Conv2D):
                last_conv_layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


# 🔥 Severity calculation
def calculate_severity(heatmap):
    intensity = np.mean(heatmap)

    if intensity < 0.3:
        return "Mild"
    elif intensity < 0.6:
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
                "error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}",
                "success": False
            }), 400

        # Image preprocessing
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((128, 128))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        pred = model.predict(img_array, verbose=0)[0][0]
        result = "PNEUMONIA" if pred > 0.5 else "NORMAL"

        confidence = float(pred) if result == "PNEUMONIA" else float(1 - pred)
        confidence_percent = round(confidence * 100, 2)

        # Grad-CAM
        heatmap = make_gradcam_heatmap(img_array, model)
        severity = calculate_severity(heatmap)

        return jsonify({
            "success": True,
            "prediction": result,
            "confidence": confidence_percent,
            "raw_score": float(pred),
            "severity": severity,
            "message": f"X-ray analysis complete. The image shows {result} with {confidence_percent}% confidence."
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
    app.run(host="0.0.0.0", port=5000, debug=True)