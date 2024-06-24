import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model_file.pkl')

# Define the expected feature names used during training
expected_features = ['movement_reactions', 'mentality_composure', 'passing', 'dribbling', 'physic',
                     'attacking_short_passing', 'mentality_vision', 'skill_long_passing', 'shooting',
                     'power_shot_power', 'age']

def main():
    st.title("⚽ FIFA Player Rating Predictor 🏆")
    
    html_temp = """
    <div style="background-color:#025246;padding:10px;border-radius:10px;">
    <h2 style="color:white;text-align:center;">Player Rating Predictor App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    st.markdown("### Enter player attributes to predict the overall rating 🌟")

    # Input fields for user to input player features
    movement_reactions = st.number_input("Movement Reactions ⚡", min_value=0.0)
    mentality_composure = st.number_input("Mentality Composure 🧠", min_value=0.0)
    passing = st.number_input("Passing 🎯", min_value=0.0)
    dribbling = st.number_input("Dribbling 🕺", min_value=0.0)
    physic = st.number_input("Physic 💪", min_value=0.0)
    attacking_short_passing = st.number_input("Attacking Short Passing 🔄", min_value=0.0)
    mentality_vision = st.number_input("Mentality Vision 👁️", min_value=0.0)
    skill_long_passing = st.number_input("Skill Long Passing 🦵", min_value=0.0)
    shooting = st.number_input("Shooting 🎯", min_value=0.0)
    power_shot_power = st.number_input("Power Shot Power 💥", min_value=0.0)
    age = st.number_input("Age 📅", min_value=0.0)

    # When the user clicks the predict button
    if st.button("Predict 🧙‍♂️"):
        # Create a DataFrame with user inputs
        features = {
            'movement_reactions': movement_reactions,
            'mentality_composure': mentality_composure,
            'passing': passing,
            'dribbling': dribbling,
            'physic': physic,
            'attacking_short_passing': attacking_short_passing,
            'mentality_vision': mentality_vision,
            'skill_long_passing': skill_long_passing,
            'shooting': shooting,
            'power_shot_power': power_shot_power,
            'age': age
        }
        
        # Ensure the columns are in the correct order
        df = pd.DataFrame([features], columns=expected_features)

        # Perform prediction using the loaded model
        prediction = model.predict(df)
        output = prediction[0]

        # Display the predicted player rating
        st.success(f'Predicted Player Rating (Overall): {output:.2f} ⭐')

if _name_ == '_main_':
    main()
