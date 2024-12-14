import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load and preprocess the dataset
@st.cache
def load_and_train_model():
    # Load the dataset
    s = pd.read_csv('finaldataset.csv')  # Replace with your dataset path

    # Define the clean_sm function
    def clean_sm(x):
        return np.where(x == 1, 1, 0)

    # Preprocess the data
    s['sm_li'] = clean_sm(s['web1h'])
    s['income'] = np.where(s['income'] > 9, np.nan, s['income'])
    s['educ2'] = np.where(s['educ2'] > 8, np.nan, s['educ2'])
    s['parent'] = clean_sm(s['par'])
    s['married'] = clean_sm(np.where(s['marital'] == 1, 1, 0))
    s['female'] = clean_sm(np.where(s['gender'] == 2, 1, 0))
    s['age'] = np.where(s['age'] > 98, np.nan, s['age'])
    s = s.drop(columns=['web1h', 'par', 'marital', 'gender']).dropna()

    # Train the model
    X = s[['income', 'educ2', 'age', 'parent', 'married', 'female']]
    y = s['sm_li']
    model = LogisticRegression(class_weight='balanced', random_state=42)
    model.fit(X, y)

    return model

# Load the trained model
model = load_and_train_model()

# Streamlit app interface
st.title("LinkedIn User Predictor")
st.write("This app predicts if someone is likely to use LinkedIn based on demographic data.")

# User input for the features
income = st.selectbox("Income Level (1-9)", options=list(range(1, 10)))
education = st.selectbox("Education Level (1-8)", options=list(range(1, 9)))
age = st.slider("Age", min_value=18, max_value=98, value=30)
parent = st.radio("Are you a parent?", options=["Yes", "No"])
married = st.radio("Are you married?", options=["Yes", "No"])
gender = st.radio("Gender", options=["Female", "Male"])

# Format the input into a DataFrame for prediction
user_data = pd.DataFrame({
    'income': [income],
    'educ2': [education],
    'age': [age],
    'parent': [1 if parent == "Yes" else 0],
    'married': [1 if married == "Yes" else 0],
    'female': [1 if gender == "Female" else 0]
})

# Make predictions when a button is clicked
if st.button("Predict"):
    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0, 1]

    # Display results
    st.write(f"Prediction: {'LinkedIn User' if prediction == 1 else 'Not a LinkedIn User'}")
    st.write(f"Probability of being a LinkedIn User: {probability:.2f}")