import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/yogaardiansyah/xssML/main/XSS_dataset.csv', encoding='utf-8-sig')
df = df[df.columns[-2:]]  # assuming the last two columns are the relevant ones

# Get Sentences data from data frame
sentences = df['Sentence'].values

# Initialize and fit tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

# Load the model
model = tf.keras.models.load_model('xss_model.h5')

# Streamlit app
st.title("XSS Detector")
st.write("Enter HTML code or a URL to check for XSS vulnerabilities:")

# User input
user_input = st.text_area("HTML Code", height=200)
user_url = st.text_input("URL")

# Function to predict XSS
def predict_xss(html_code):
    # Preprocess the input
    sequences = tokenizer.texts_to_sequences([html_code])
    padded_sequences = pad_sequences(sequences, maxlen=100)  # use the same maxlen used during training
    
    # Predict
    prediction = model.predict(padded_sequences)
    return prediction[0][0]

# Function to get HTML content from a URL
def get_html_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup.get_text()
        else:
            st.error("Failed to retrieve the URL. Please check the URL and try again.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        return None

if st.button("Check for XSS"):
    if user_input:
        prediction = predict_xss(user_input)
        if prediction > 0.5:
            st.error("The input contains potential XSS vulnerabilities!")
        else:
            st.success("The input is clean from XSS vulnerabilities.")
    elif user_url:
        html_content = get_html_from_url(user_url)
        if html_content:
            prediction = predict_xss(html_content)
            if prediction > 0.5:
                st.error("The URL contains potential XSS vulnerabilities!")
            else:
                st.success("The URL is clean from XSS vulnerabilities.")
    else:
        st.warning("Please enter some HTML code or a URL to check.")
