import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
model_path = 'model_file.pkl'
model = joblib.load(model_path)

# Load the scaler
scaler_path = 'scaler.pkl'
scaler = joblib.load(scaler_path)

# Define the expected feature names used during training
expected_features = ['movement_reactions', 'entality_composure', 'passing', 
                     'dribbling', 'physic', 'attacking_short_passing', 
                     'entality_vision', 'kill_long_passing', 'hooting', 
                     'power_shot_power', 'age']

def main():
    st.title("FIFA Rating Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Predict a Players Rating</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Create two columns for input fields
    col1, col2 = st.columns(2)

    # Create input fields for each feature with labels
    inputs = {}
    for i, feature in enumerate(expected_features):
        if i < len(expected_features) // 2:
            inputs[feature] = col1.number_input(f"Enter {feature.replace('_', ').capitalize()} : ", value=0.0)
        else:
            inputs[feature] = col2.number_input(f"Enter {feature.replace('_', ').capitalize()} : ", value=0.0)

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
