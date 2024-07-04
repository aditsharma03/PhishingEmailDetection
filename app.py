import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the Random Forest model from the file
with open('random_forest_model.pkl', 'rb') as file:
    loaded_rf_model = pickle.load(file)

# Load the TfidfVectorizer from the file
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Function to preprocess input
def preprocess_input(email_text):
    email_vectorized = tfidf_vectorizer.transform([email_text])
    return email_vectorized

# Streamlit UI
st.title("Phishing Email Detection")

email_text = st.text_area("Enter the email text:")

if st.button("Predict"):
    preprocessed_email = preprocess_input(email_text)
    prediction = loaded_rf_model.predict(preprocessed_email)
    
    if prediction[0] == "Phishing Email":
        st.error("The email is classified as a phishing email.")
    else:
        st.success("The email is classified as a safe email.")

