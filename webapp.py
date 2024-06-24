import streamlit as st
import pandas as pd
import joblib
import io
import requests

# Define the expected feature names used during training
expected_features = ['movement_reactions', 'mentality_composure', 'passing', 
                     'dribbling', 'physic', 'attacking_short_passing', 
                     'mentality_vision', 'skill_long_passing', 'shooting', 
                     'power_shot_power', 'age']

def load_file_from_github(url):
    response = requests.get(url)
    response.raise_for_status()  # Check for HTTP request errors
    return io.BytesIO(response.content)

def main():
    st.title("FIFA Player Rating Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Player Rating Predictor App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # URLs to download the model and scaler from GitHub
    model_url = 'https://raw.githubusercontent.com/C-Ronny/RONELLEHCUDJOE._SportsPrediction/main/model.pkl'
    scaler_url = 'https://raw.githubusercontent.com/C-Ronny/RONELLEHCUDJOE._SportsPrediction/main/scaler.pkl'

    try:
        # Load the model and scaler from GitHub
        model = joblib.load(load_file_from_github(model_url))
        scaler = joblib.load(load_file_from_github(scaler_url))
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return

    # Create input fields for user input
    inputs = {}
    for feature in expected_features:
        inputs[feature] = st.number_input(feature, value=0.0)

    # Create a button to make predictions
    if st.button("Predict"):
        try:
            # Create a pandas dataframe from user input
            input_df = pd.DataFrame(inputs, index=[0])

            # Scale the input data using the loaded scaler
            scaled_input = scaler.transform(input_df)

            # Make predictions using the loaded model
            prediction = model.predict(scaled_input)

            # Display the predicted rating
            st.write("Predicted Rating:", prediction[0])
        except Exception as e:
            st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()
