import streamlit as st
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()



# Load the Random Forest model
with open('random_forest_model.pkl', 'rb') as file:
    loaded_rf_model = pickle.load(file)

# Load the TfidfVectorizer
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# preprocess input
def preprocess_text(text):
    # Ensure the text is treated as a string
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters except for common ones like @ and . in email addresses
    text = re.sub(r'[^a-zA-Z0-9@.\s]', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()





    words = word_tokenize(text)

    filtered_sentence = [stemmer.stem(word) for word in words if word not in stop_words]


    return ' '.join(filtered_sentence)




st.title("Phishing Email Detection")

email_text = st.text_area("Enter the email text:")

if st.button("Predict"):
    preprocessed_email = preprocess_text(email_text)
    email_vectorized = tfidf_vectorizer.transform([preprocessed_email])
    prediction = loaded_rf_model.predict(email_vectorized)
    
    if prediction[0] == "Phishing Email":
        st.error("The email is classified as a phishing email.")
    else:
        st.success("The email is classified as a safe email.")

