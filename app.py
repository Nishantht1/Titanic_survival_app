import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('titanic_model.pkl')

st.title("🚢 Titanic Survival Predictor")

st.markdown("Enter passenger details below:")

# Collect user input
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.slider("Age", 0, 100, 30)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.slider("Fare Paid", 0.0, 500.0, 50.0)
embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

# Format input to match model
input_data = pd.DataFrame([{
    'PassengerId': 999,  # dummy ID
    'Pclass': pclass,
    'Sex': 0 if sex == 'Male' else 1,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked_Q': 1 if embarked == 'Q' else 0,
    'Embarked_S': 1 if embarked == 'S' else 0
}])

# Predict when button is clicked
if st.button("Predict Survival"):
    pred = model.predict(input_data)[0]
    result = "✅ Survived" if pred == 1 else "❌ Did Not Survive"
    st.subheader(f"Prediction: {result}")
