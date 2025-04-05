import pickle
import string
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('all')

# Initialize Porter Stemmer
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words and word not in string.punctuation]

    # Apply stemming
    text = [ps.stem(word) for word in text]

    return " ".join(text)

# Load pre-trained model and vectorizer
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: Model files not found! Ensure 'vectorizer.pkl' and 'model.pkl' exist.")

# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip():
        transformed_sms = transform_text(input_sms)
        #st.write("Preprocessed Text:", transformed_sms)  # Debug output
        
        vector_input = tfidf.transform([transformed_sms])
        #st.write("TF-IDF Features:", vector_input)  # Check if features are non-zero
        
        result = model.predict(vector_input)[0]
        st.write("Raw Prediction Score:", model.predict_proba(vector_input))  # Check probabilities
        
        if result == 1:
            st.header("🚨 Spam Message")
        else:
            st.header("✅ Not Spam")
    else:
        st.warning("Please enter a message before predicting.")
