import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create Streamlit app
st.title("Titanic Survival Prediction")

# User input
st.subheader("Enter features:")
age = st.number_input("Age")
sex = st.selectbox("Sex", ["Male", "Female"])
embarked = st.selectbox("Embarked", ["S", "C", "Q"])
fare = st.number_input("Fare")
parch = st.number_input("Number of Parents/Children")
passenger_id = st.number_input("Passenger ID")
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sibsp = st.number_input("Number of Siblings/Spouses")

# Predict button
if st.button("Predict"):
    # Map sex and embarked to numerical values
    sex_map = {"Male": 0, "Female": 1}
    embarked_map = {"S": 0, "C": 1, "Q": 2}
    pclass_map = {1: 0, 2: 1, 3: 2}
    
    # Create input dataframe with exact feature names and order
    input_df = pd.DataFrame({
        'Age': [age],
        'Sex': [sex_map[sex]],
        'Embarked': [embarked_map[embarked]],
        'Fare': [fare],
        'Parch': [parch],
        'PassengerId': [passenger_id],
        'Pclass': [pclass_map[pclass]],
        'SibSp': [sibsp]
    }, columns=model.feature_names_in_)  # Ensure correct order
    
    # Make prediction
    prediction = model.predict(input_df)
    
    # Display result
    st.subheader("Prediction:")
    st.write("Survived" if prediction[0] == 1 else "Did not survive")