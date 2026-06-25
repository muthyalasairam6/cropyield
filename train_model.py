"""
Fertilizer Prediction AI Agent - Model Training Pipeline
Trains multiple ML models on fertilizer prediction data and saves the best one.
Based on the Colab notebook approach using RandomForest, SVM, KNN, and MLP.
"""

import pandas as pd
import numpy as np
import os
import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


def generate_fertilizer_dataset(n_samples=500):
    """
    Generate a realistic synthetic fertilizer prediction dataset.
    Columns: Temperature, Humidity, Moisture, Soil Type, Crop Type,
             Nitrogen, Potassium, Phosphorous, Fertilizer Name
    """
    np.random.seed(42)

    soil_types = ["Sandy", "Loamy", "Black", "Red", "Clayey"]
    crop_types = ["Maize", "Sugarcane", "Cotton", "Tobacco", "Paddy",
                  "Barley", "Wheat", "Millets", "Oil seeds", "Ground Nuts"]
    fertilizer_names = ["Urea", "DAP", "14-35-14", "28-28", "17-17-17",
                        "20-20", "10-26-26"]

    # Define realistic parameter ranges for each fertilizer
    fertilizer_profiles = {
        "Urea": {
            "temp": (25, 35), "humidity": (50, 70), "moisture": (30, 50),
            "nitrogen": (30, 50), "potassium": (0, 10), "phosphorous": (0, 10),
            "soils": ["Sandy", "Loamy", "Clayey"],
            "crops": ["Maize", "Paddy", "Wheat", "Sugarcane"]
        },
        "DAP": {
            "temp": (25, 35), "humidity": (45, 65), "moisture": (35, 55),
            "nitrogen": (8, 20), "potassium": (0, 10), "phosphorous": (30, 50),
            "soils": ["Loamy", "Black", "Red"],
            "crops": ["Sugarcane", "Cotton", "Wheat", "Barley"]
        },
        "14-35-14": {
            "temp": (28, 38), "humidity": (55, 75), "moisture": (50, 70),
            "nitrogen": (5, 15), "potassium": (5, 15), "phosphorous": (25, 40),
            "soils": ["Black", "Red", "Clayey"],
            "crops": ["Cotton", "Tobacco", "Oil seeds"]
        },
        "28-28": {
            "temp": (28, 38), "humidity": (55, 70), "moisture": (30, 45),
            "nitrogen": (18, 30), "potassium": (0, 8), "phosphorous": (15, 25),
            "soils": ["Red", "Loamy", "Sandy"],
            "crops": ["Tobacco", "Maize", "Ground Nuts"]
        },
        "17-17-17": {
            "temp": (20, 32), "humidity": (60, 80), "moisture": (40, 60),
            "nitrogen": (12, 22), "potassium": (12, 22), "phosphorous": (12, 22),
            "soils": ["Loamy", "Black", "Clayey"],
            "crops": ["Barley", "Millets", "Wheat"]
        },
        "20-20": {
            "temp": (22, 34), "humidity": (50, 70), "moisture": (35, 55),
            "nitrogen": (15, 25), "potassium": (15, 25), "phosphorous": (0, 10),
            "soils": ["Sandy", "Red", "Loamy"],
            "crops": ["Paddy", "Millets", "Ground Nuts"]
        },
        "10-26-26": {
            "temp": (24, 36), "humidity": (55, 75), "moisture": (45, 65),
            "nitrogen": (5, 15), "potassium": (20, 35), "phosphorous": (20, 35),
            "soils": ["Clayey", "Black", "Red"],
            "crops": ["Oil seeds", "Ground Nuts", "Cotton"]
        },
    }

    samples_per_class = n_samples // len(fertilizer_names)
    data = []

    for fert_name, profile in fertilizer_profiles.items():
        for _ in range(samples_per_class):
            temp = np.random.randint(*profile["temp"])
            humidity = np.random.randint(*profile["humidity"])
            moisture = np.random.randint(*profile["moisture"])
            nitrogen = np.random.randint(*profile["nitrogen"])
            potassium = np.random.randint(*profile["potassium"])
            phosphorous = np.random.randint(*profile["phosphorous"])
            soil = np.random.choice(profile["soils"])
            crop = np.random.choice(profile["crops"])

            data.append({
                "Temparature": temp,
                "Humidity": humidity,
                "Moisture": moisture,
                "Soil Type": soil,
                "Crop Type": crop,
                "Nitrogen": nitrogen,
                "Potassium": potassium,
                "Phosphorous": phosphorous,
                "Fertilizer Name": fert_name
            })

    df = pd.DataFrame(data)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def train_and_evaluate():
    """Train multiple models and save the best one."""

    print("=" * 60)
    print("[*] Fertilizer Prediction AI Agent - Training Pipeline")
    print("=" * 60)

    # --- Step 1: Load or Generate Data ---
    csv_path = os.path.join(os.path.dirname(__file__), "Fertilizer Prediction.csv")
    if os.path.exists(csv_path):
        print(f"\n[+] Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print("\n[+] Generating synthetic fertilizer dataset...")
        df = generate_fertilizer_dataset(n_samples=700)
        df.to_csv(csv_path, index=False)
        print(f"   Saved dataset to: {csv_path}")

    print(f"\n[i] Dataset shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   Fertilizer distribution:")
    print(df["Fertilizer Name"].value_counts().to_string())
    print(f"\n   Null values: {df.isnull().sum().sum()}")

    # --- Step 2: Preprocessing ---
    print("\n[*] Preprocessing data...")

    # Encode categorical columns
    le_soil = LabelEncoder()
    le_crop = LabelEncoder()
    le_fertilizer = LabelEncoder()

    df["Soil Type"] = le_soil.fit_transform(df["Soil Type"])
    df["Crop Type"] = le_crop.fit_transform(df["Crop Type"])
    df["Fertilizer Name"] = le_fertilizer.fit_transform(df["Fertilizer Name"])

    # Features and target
    X = df.drop("Fertilizer Name", axis=1)
    y = df["Fertilizer Name"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set:     {X_test.shape[0]} samples")
    print(f"   Features:     {X_train.shape[1]}")
    print(f"   Classes:      {len(le_fertilizer.classes_)}")

    # --- Step 3: Train Models ---
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42
        ),
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=500,
            random_state=42,
            early_stopping=True
        ),
    }

    results = {}
    best_accuracy = 0
    best_model_name = ""
    best_model = None

    print("\n" + "=" * 60)
    print("[*] Training Models...")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n[>] Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
        }

        print(f"   [OK] Accuracy:  {acc*100:.2f}%")
        print(f"   [--] Precision: {prec*100:.2f}%")
        print(f"   [--] Recall:    {rec*100:.2f}%")
        print(f"   [--] F1 Score:  {f1*100:.2f}%")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model

    # --- Step 4: Save Best Model & Artifacts ---
    print("\n" + "=" * 60)
    print(f"[BEST] Best Model: {best_model_name} ({best_accuracy*100:.2f}%)")
    print("=" * 60)

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\n[+] Saved model to: {model_path}")

    # Save encoders and scaler
    encoders_path = os.path.join(model_dir, "encoders.pkl")
    with open(encoders_path, "wb") as f:
        pickle.dump({
            "le_soil": le_soil,
            "le_crop": le_crop,
            "le_fertilizer": le_fertilizer,
            "scaler": scaler,
        }, f)
    print(f"[+] Saved encoders to: {encoders_path}")

    # Save metadata
    metadata = {
        "best_model": best_model_name,
        "feature_columns": list(X.columns),
        "soil_types": list(le_soil.classes_),
        "crop_types": list(le_crop.classes_),
        "fertilizer_names": list(le_fertilizer.classes_),
        "results": results,
    }
    metadata_path = os.path.join(model_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Saved metadata to: {metadata_path}")

    print("\n[OK] Training complete! All artifacts saved to ./models/")
    return results


if __name__ == "__main__":
    train_and_evaluate()
