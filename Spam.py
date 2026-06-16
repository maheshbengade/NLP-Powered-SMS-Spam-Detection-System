import streamlit as st
import pickle

# Page Configuration
st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩"
)

# Load model and vectorizer
model = pickle.load(open('SMS Spam Filtering System.pkl', 'rb'))
tf_idf = pickle.load(open('SMS Spam Filtering System vectorizer.pkl', 'rb'))

# Preprocessing
def preprocess(text):
    return text.lower()

# Title
st.title("📩 SMS Spam Detection System")

# Input box
message = st.text_area(
    "SMS Message",
    placeholder="Type your message here..."
)

# Predict button
if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        processed_msg = preprocess(message)
        test_input = tf_idf.transform([processed_msg])
        prediction = model.predict(test_input)[0]

        st.divider()

        if prediction == 1:
            st.error("🚨 Spam Message")
            st.info("This message appears to be promotional, fraudulent, or suspicious.")
        else:
            st.success("✅ Not Spam Message")
            st.info("This message appears to be a normal personal message.")

# Footer
st.markdown("""
<hr style='border:none; border-top:1px solid #E5E3DC; margin:1rem 0 0;'>
<div style='text-align:center; font-size:11.5px; color:#B0AEA8; padding:0.5rem 0 2rem; letter-spacing:0.02em;'>
    TF-IDF &nbsp;·&nbsp; Multinomial Naive Bayes &nbsp;·&nbsp; Python &nbsp;·&nbsp; Streamlit
</div>
""", unsafe_allow_html=True)
