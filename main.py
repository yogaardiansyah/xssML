import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

# Load the tokenizer and model
with open('tokenizer.pkl', 'wb') as handle:
    tokenizer = pickle.load(handle)

model = tf.keras.models.load_model('xss_model.h5')

# Streamlit app
st.title("XSS Detector")
st.write("Enter HTML code to check for XSS vulnerabilities:")

# User input
user_input = st.text_area("HTML Code", height=200)

# Function to predict XSS
def predict_xss(html_code):
    # Preprocess the input
    sequences = tokenizer.texts_to_sequences([html_code])
    padded_sequences = pad_sequences(sequences, maxlen=100)  # use the same maxlen used during training

    # Predict
    prediction = model.predict(padded_sequences)
    return prediction[0][0]

if st.button("Check for XSS"):
    if user_input:
        prediction = predict_xss(user_input)
        if prediction > 0.5:
            st.error("The input contains potential XSS vulnerabilities!")
        else:
            st.success("The input is clean from XSS vulnerabilities.")
    else:
        st.warning("Please enter some HTML code to check.")
