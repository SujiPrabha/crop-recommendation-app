from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")

# Prepare model
X = data[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y = data["label"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)


@app.route("/")
def home():
    return "Crop Recommendation API is running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        received = request.get_json()

        values = received["data"]

        N = float(values[0])
        P = float(values[1])
        K = float(values[2])
        temperature = float(values[3])
        humidity = float(values[4])
        ph = float(values[5])
        rainfall = float(values[6])

        input_data = [[
            N, P, K, temperature, humidity, ph, rainfall
        ]]

        crop = model.predict(input_data)[0]

        result = {
            "crop": crop,
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
