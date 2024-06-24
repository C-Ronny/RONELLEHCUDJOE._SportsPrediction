import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), 'model_file.pkl')
model = joblib.load('/Users/ronny/Downloads/FIFA_PROJECT/model_file.pkl')

# Load the scaler
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
scaler = joblib.load('/Users/ronny/Downloads/FIFA_PROJECT/scaler.pkl')

# Define the expected feature names used during training
expected_features = ['movement_reactions', 'entality_composure', 'passing', 
                     'dribbling', 'physic', 'attacking_short_passing', 
                     'entality_vision', 'kill_long_passing', 'hooting', 
                     'power_shot_power', 'age']

def main():
    st.title("FIFA Player Rating Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Player Rating Predictor App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Create input fields for each feature with labels
    inputs = {}
    for feature in expected_features:
        inputs[feature] = st.number_input(f"Enter {feature.replace('_', ' ').capitalize()} : ", value=0.0)

    # Create a button to predict the rating
    if st.button("Predict Rating"):
        # Create a pandas dataframe from the input values
        input_data = pd.DataFrame([inputs], columns=expected_features)

        # Scale the input data using the loaded scaler
        scaled_data = scaler.transform(input_data)

        # Make a prediction using the loaded model
        prediction = model.predict(scaled_data)[0]

        # Display the predicted rating
        st.write("Predicted Rating:", round(prediction, 2))

if __name__ == "__main__":
    main()
