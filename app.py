from flask import Flask, request, jsonify
import pandas as pd
import joblib
from prometheus_client import Counter, generate_latest

app = Flask(__name__)

# Counter to track the number of predictions
prediction_counter = Counter("model_predictions", "Total Predictions Made")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_features = pd.DataFrame(data["features"])

        # Load all preprocessing objects
        loaded_scaler = joblib.load("scaler.pkl")
        loaded_selector = joblib.load("selector.pkl")
        loaded_model = joblib.load("model.pkl")

        # Apply same preprocessing as training
        input_scaled = loaded_scaler.transform(input_features)
        input_selected = loaded_selector.transform(input_scaled)

        predictions = loaded_model.predict(input_selected)

        # Increment counter
        prediction_counter.inc()

        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/metrics", methods=["GET"])
def metrics():
    return generate_latest()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
