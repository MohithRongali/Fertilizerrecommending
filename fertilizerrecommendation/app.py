from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained scikit-learn model, label encoder, and scaler
model = joblib.load(r"D:\fertilizerrecommendation\models\fertilizer_model.pkl")  # .pkl scikit-learn model
label_encoder = joblib.load(r"D:\fertilizerrecommendation\models\label_encoder.pkl")
scaler = joblib.load(r"D:\fertilizerrecommendation\models\scaler.pkl")

csv_file = r"D:\fertilizerrecommendation\data\f2.csv"

def encode_label(value, encoder):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        encoder.classes_ = np.append(encoder.classes_, value)
        return encoder.transform([value])[0]

def predict_fertilizer(soil_type, crop_type, nitrogen, phosphorous, potassium):
    # Prepare input feature vector (you must adjust to your actual feature set)
    # Example: You might have one-hot encoded soil and crop types, here simplified:
    input_features = np.array([[soil_type, crop_type, nitrogen, phosphorous, potassium]], dtype=object)

    # You need to one-hot encode categorical inputs exactly like training, or pre-process them accordingly
    # For simplicity, let's assume you did label encoding for soil_type and crop_type:
    soil_encoded = encode_label(soil_type, label_encoder)
    crop_encoded = encode_label(crop_type, label_encoder)

    # Build feature vector accordingly (adjust this to your model's expected input)
    # For example, if your model uses [soil_encoded, crop_encoded, nitrogen, phosphorous, potassium]:
    input_vector = np.array([[soil_encoded, crop_encoded, nitrogen, phosphorous, potassium]])

    # Scale numerical features (assuming scaler was fitted on all features or numerical only)
    input_scaled = scaler.transform(input_vector)

    # Predict fertilizer class (returns label encoded integer)
    prediction_encoded = model.predict(input_scaled)[0]

    # Decode prediction back to fertilizer name
    fertilizer_name = label_encoder.inverse_transform([prediction_encoded])[0]

    return fertilizer_name

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        soil_type = request.form['soil_type'].strip().lower()
        crop_type = request.form['crop_type'].strip().lower()
        nitrogen = float(request.form['nitrogen'])
        potassium = float(request.form['potassium'])
        phosphorous = float(request.form['phosphorous'])

        # Load CSV for existing data check
        df = pd.read_csv(csv_file)
        df['Nitrogen'] = df['Nitrogen'].astype(float)
        df['Potassium'] = df['Potassium'].astype(float)
        df['Phosphorous'] = df['Phosphorous'].astype(float)
        df['Soil_Type'] = df['Soil_Type'].str.strip().str.lower()
        df['Crop_Type'] = df['Crop_Type'].str.strip().str.lower()

        match = df[
            (df['Soil_Type'] == soil_type) &
            (df['Crop_Type'] == crop_type) &
            (df['Nitrogen'] == nitrogen) &
            (df['Potassium'] == potassium) &
            (df['Phosphorous'] == phosphorous)
        ]

        if not match.empty:
            fertilizer = match.iloc[0]['Fertilizer']
        else:
            fertilizer = predict_fertilizer(soil_type, crop_type, nitrogen, phosphorous, potassium)

            new_data = pd.DataFrame([[soil_type, crop_type, nitrogen, phosphorous, potassium, fertilizer]],
                                    columns=['Soil_Type', 'Crop_Type', 'Nitrogen', 'Phosphorous', 'Potassium', 'Fertilizer'])
            df = pd.concat([df, new_data], ignore_index=True)
            df.to_csv(csv_file, index=False)

        return jsonify({"prediction": fertilizer})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
