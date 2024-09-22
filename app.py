from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import json

app = Flask(__name__)
CORS(app)

# Load the model and necessary artifacts
model = pickle.load(open('bengaluru_house_price_model.pkl', 'rb'))

# Load the data columns (this includes location columns for one-hot encoding)
with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

@app.route("/")
def home():
    return render_template("welcome.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Extract JSON data
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Extract form values from JSON
        location = data.get('location', None)
        total_sqft = float(data.get('total_sqft', 0))
        bath = int(data.get('bath', 0))
        bhk = int(data.get('bhk', 0))

        if not location or not total_sqft or not bath or not bhk:
            return jsonify({"error": "Missing required fields"}), 400

        # Prepare the input array
        x = np.zeros(len(data_columns))
        x[0] = total_sqft
        x[1] = bath
        x[2] = bhk

        # One-hot encoding for location (set the correct location column to 1)
        if location.lower() in data_columns:
            loc_index = data_columns.index(location.lower())
            x[loc_index] = 1

        # Make the prediction
        prediction = model.predict([x])[0]

        # Return the result as JSON
        return jsonify({
            "Price": round(float(prediction), 7)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)
