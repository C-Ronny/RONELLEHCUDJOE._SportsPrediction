import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained model and scaler from GitHub repository
model_path = 'https://raw.githubusercontent.com/C-Ronny/RONELLEHCUDJOE._SportsPrediction/main/model_file.pkl'
scaler_path = 'https://raw.githubusercontent.com/C-Ronny/RONELLEHCUDJOE._SportsPrediction/main/scaler.pkl'

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Define the expected feature names used during training
expected_features = ['movement_reactions', 'mentality_composure', 'passing', 
                     'dribbling', 'physic', 'attacking_short_passing', 
                     'mentality_vision', 'skill_long_passing', 'shooting', 
                     'power_shot_power', 'age']

def main():
    st.title("FIFA Player Rating Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Player Rating Predictor App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Create input fields for user input
    inputs = {}
    for feature in expected_features:
        inputs[feature] = st.number_input(feature, value=0.0)

    # Create a button to make predictions
    if st.button("Predict"):
        # Create a pandas dataframe from user input
        input_df = pd.DataFrame(inputs, index=[0])

        # Scale the input data using the loaded scaler
        scaled_input = scaler.transform(input_df)

        # Make predictions using the loaded model
        prediction = model.predict(scaled_input)

        # Display the predicted rating
        st.write("Predicted Rating:", prediction[0])

if __name__ == "__main__":
    main()
