from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('./earthquake_model.pkl')

# Function to estimate distance based on magnitude
def estimate_distance(magnitude: float) -> float:
    if magnitude >= 7:
        return 500
    elif magnitude >= 5:
        return 200
    else:
        return 100

@app.route("/", methods=["GET"])
def welcome():
    return jsonify({"message": "Welcome to my Flask app!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON request
        data = request.get_json()

        # Check if 'parameters' key exists and has 12 values
        if "parameters" not in data or len(data["parameters"]) != 12:
            return jsonify({"error": "12 input parameters are required."}), 400

        # Convert input to numpy array and make predictions
        input_array = np.array([data["parameters"]])
        magnitude = model.predict(input_array)[0]
        distance = estimate_distance(magnitude)

        # Return the response
        return jsonify({"magnitude": magnitude, "distance": distance})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
