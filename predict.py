"""
Fertilizer Prediction AI Agent - Prediction Module
Loads the trained model and provides fertilizer recommendations.
"""

import os
import pickle
import json
import numpy as np


class FertilizerPredictor:
    """AI Agent for predicting the best fertilizer based on soil and crop conditions."""

    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "models")

        self.model_dir = model_dir
        self.model = None
        self.encoders = None
        self.metadata = None
        self._load()

    def _load(self):
        """Load the trained model, encoders, and metadata."""
        model_path = os.path.join(self.model_dir, "best_model.pkl")
        encoders_path = os.path.join(self.model_dir, "encoders.pkl")
        metadata_path = os.path.join(self.model_dir, "metadata.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "Model not found. Run 'python train_model.py' first to train the model."
            )

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(encoders_path, "rb") as f:
            self.encoders = pickle.load(f)

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

    def predict(self, temperature, humidity, moisture, soil_type, crop_type,
                nitrogen, potassium, phosphorous):
        """
        Predict the best fertilizer for given conditions.

        Parameters:
            temperature (int): Temperature in Celsius
            humidity (int): Humidity percentage
            moisture (int): Moisture percentage
            soil_type (str): One of Sandy, Loamy, Black, Red, Clayey
            crop_type (str): One of Maize, Sugarcane, Cotton, Tobacco, Paddy, etc.
            nitrogen (int): Nitrogen content
            potassium (int): Potassium content
            phosphorous (int): Phosphorous content

        Returns:
            dict: Prediction result with fertilizer name and confidence
        """
        le_soil = self.encoders["le_soil"]
        le_crop = self.encoders["le_crop"]
        le_fertilizer = self.encoders["le_fertilizer"]
        scaler = self.encoders["scaler"]

        # Validate inputs
        if soil_type not in le_soil.classes_:
            return {
                "error": f"Invalid soil type '{soil_type}'. "
                         f"Valid options: {list(le_soil.classes_)}"
            }
        if crop_type not in le_crop.classes_:
            return {
                "error": f"Invalid crop type '{crop_type}'. "
                         f"Valid options: {list(le_crop.classes_)}"
            }

        # Encode categorical features
        soil_encoded = le_soil.transform([soil_type])[0]
        crop_encoded = le_crop.transform([crop_type])[0]

        # Build feature array in same order as training
        features = np.array([[
            temperature, humidity, moisture,
            soil_encoded, crop_encoded,
            nitrogen, potassium, phosphorous
        ]])

        # Scale
        features_scaled = scaler.transform(features)

        # Predict
        prediction = self.model.predict(features_scaled)
        fertilizer_name = le_fertilizer.inverse_transform(prediction)[0]

        # Get confidence if model supports predict_proba
        confidence = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features_scaled)[0]
            confidence = round(float(max(proba)) * 100, 2)

        # Fertilizer usage tips
        tips = self._get_fertilizer_tips(fertilizer_name)

        return {
            "fertilizer": fertilizer_name,
            "confidence": confidence,
            "model_used": self.metadata.get("best_model", "Unknown"),
            "tips": tips,
            "input_summary": {
                "temperature": temperature,
                "humidity": humidity,
                "moisture": moisture,
                "soil_type": soil_type,
                "crop_type": crop_type,
                "nitrogen": nitrogen,
                "potassium": potassium,
                "phosphorous": phosphorous,
            }
        }

    def _get_fertilizer_tips(self, fertilizer_name):
        """Return usage tips for the recommended fertilizer."""
        tips = {
            "Urea": "High nitrogen fertilizer (46% N). Best for leafy growth. Apply in split doses. Avoid application before heavy rain.",
            "DAP": "Di-ammonium Phosphate (18% N, 46% P). Excellent for root development. Apply at sowing time for best results.",
            "14-35-14": "Balanced NPK fertilizer. Good for flowering and fruiting stages. Mix with soil before planting.",
            "28-28": "High NP fertilizer. Ideal for crops needing strong vegetative and root growth simultaneously.",
            "17-17-17": "Perfectly balanced NPK. Versatile all-purpose fertilizer suitable for most crops and growth stages.",
            "20-20": "Balanced NP fertilizer. Good for general crop nutrition. Apply during early growth stages.",
            "10-26-26": "High PK fertilizer. Excellent for tuber crops and fruit development. Apply during reproductive stage.",
        }
        return tips.get(fertilizer_name, "Follow manufacturer guidelines for application rates.")

    def get_metadata(self):
        """Return model metadata including performance metrics."""
        return self.metadata


if __name__ == "__main__":
    # Quick test
    predictor = FertilizerPredictor()
    result = predictor.predict(
        temperature=30,
        humidity=55,
        moisture=40,
        soil_type="Sandy",
        crop_type="Maize",
        nitrogen=35,
        potassium=5,
        phosphorous=5
    )
    print("\n[*] Fertilizer Prediction Result:")
    print(f"   Recommended: {result['fertilizer']}")
    if result.get('confidence'):
        print(f"   Confidence:  {result['confidence']}%")
    print(f"   Model Used:  {result['model_used']}")
    print(f"   Tip: {result['tips']}")
