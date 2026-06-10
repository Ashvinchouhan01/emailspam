import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('email_spam_model.joblib')
vectorizer = joblib.load('email_vectorizer.joblib')

st.title('Email Spam Checker')
st.write('Paste your email message below to check if it is spam or not spam.')

# Text area for email content input
email_text = st.text_area('Email Content:', height=250, key="email_text")

# Create two columns for buttons right after the text area
col1, col2 = st.columns([1,1], gap="medium")

with col1:
    if st.button('Check'):
        if email_text.strip():
            features = vectorizer.transform([email_text])
            prediction = model.predict(features)[0]
            if prediction == 1:
                st.error('Result: Spam')
            else:
                st.success('Result: Not Spam')
        else:
            st.warning('Please enter some email content before checking.')

def clear_text():
    st.session_state.email_text = ""

with col2:
    st.button('Clear', on_click=clear_text)
