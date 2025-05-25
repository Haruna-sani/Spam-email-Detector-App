import streamlit as st
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download once (comment after first run)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize stopwords and lemmatizer globally
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if isinstance(text, str):
        text = re.sub('[^a-zA-Z]', ' ', text)  # Keep only letters
        text = text.lower().split()
        text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
        return ' '.join(text)
    return ""

@st.cache_resource
def load_model_and_vectorizer():
    vectorizer = joblib.load('/content/bow_vectorizer.pkl')
    model = joblib.load('/content/xgb_spam_model.pkl')
    return vectorizer, model

def predict_email_spam(email_text, vectorizer, model):
    cleaned = clean_text(email_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return 'SPAM' if prediction == 1 else 'LEGITIMATE'

def main():
    st.title("Email Spam Detection App")
    st.write("Enter the email body text below to check if it's SPAM or LEGITIMATE.")

    email_input = st.text_area("Email Text", height=200)

    if st.button("Predict"):
        if not email_input.strip():
            st.warning("Please enter some email text.")
            return
        
        vectorizer, model = load_model_and_vectorizer()
        result = predict_email_spam(email_input, vectorizer, model)
        
        if result == 'SPAM':
            st.error("⚠️ This email is predicted to be **SPAM**.")
        else:
            st.success("✅ This email is predicted to be **LEGITIMATE**.")

if __name__ == "__main__":
    main()
