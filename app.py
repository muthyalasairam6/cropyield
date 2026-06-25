"""
Fertilizer Prediction AI Agent - Flask Web Application
Serves a premium web UI and REST API for fertilizer predictions.
"""

from flask import Flask, render_template, request, jsonify
from predict import FertilizerPredictor
import os
import json

app = Flask(__name__)

# Initialize predictor (will fail gracefully if model not trained yet)
predictor = None


def get_predictor():
    """Lazy-load the predictor."""
    global predictor
    if predictor is None:
        try:
            predictor = FertilizerPredictor()
        except FileNotFoundError:
            return None
    return predictor


@app.route("/")
def index():
    """Serve the main web UI."""
    pred = get_predictor()
    metadata = pred.get_metadata() if pred else None
    return render_template("index.html", metadata=metadata)


@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for fertilizer prediction."""
    pred = get_predictor()
    if pred is None:
        return jsonify({
            "error": "Model not trained yet. Run 'python train_model.py' first."
        }), 500

    try:
        data = request.get_json()

        result = pred.predict(
            temperature=int(data.get("temperature", 25)),
            humidity=int(data.get("humidity", 50)),
            moisture=int(data.get("moisture", 40)),
            soil_type=data.get("soil_type", "Sandy"),
            crop_type=data.get("crop_type", "Maize"),
            nitrogen=int(data.get("nitrogen", 20)),
            potassium=int(data.get("potassium", 10)),
            phosphorous=int(data.get("phosphorous", 10)),
        )

        if "error" in result:
            return jsonify(result), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/metadata", methods=["GET"])
def metadata():
    """Return model metadata and performance metrics."""
    pred = get_predictor()
    if pred is None:
        return jsonify({"error": "Model not trained yet."}), 500
    return jsonify(pred.get_metadata())


if __name__ == "__main__":
    print("\n[*] Fertilizer Prediction AI Agent")
    print("=" * 40)

    pred = get_predictor()
    if pred is None:
        print("[!] No trained model found!")
        print("   Run 'python train_model.py' first, then restart the app.")
        print("   Starting server anyway...\n")
    else:
        print(f"[OK] Model loaded: {pred.metadata['best_model']}")
        print(f"   Fertilizers: {pred.metadata['fertilizer_names']}")
        print(f"   Soil types:  {pred.metadata['soil_types']}")
        print(f"   Crop types:  {pred.metadata['crop_types']}\n")

    app.run(debug=True, host="127.0.0.1", port=5000)
