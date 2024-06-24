import streamlit as st
import pandas as pd
import joblib
import io
import requests

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

    model_url = 'https://raw.githubusercontent.com/C-Ronny/RONELLEHCUDJOE._SportsPrediction/main/model_file.pkl'
    scaler_url = 'https://raw.githubusercontent.com/C-Ronny/RONELLEHCUDJOE._SportsPrediction/main/scaler.pkl'
    
    model_response = requests.get(model_url)
    scaler_response = requests.get(scaler_url)
    
    model = joblib.load(io.BytesIO(model_response.content))
    scaler = joblib.load(io.BytesIO(scaler_response.content))

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
