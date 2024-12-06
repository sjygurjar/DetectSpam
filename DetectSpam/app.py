import streamlit as st # type: ignore
import pickle
import re
from nltk.corpus import stopwords # type: ignore
from nltk.stem.porter import PorterStemmer # type: ignore

# Load the trained model and vectorizer
try:
    model = pickle.load(open("spam_detector.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure 'spam_detector.pkl' and 'vectorizer.pkl' are in the same directory.")
    st.stop()

# Function to preprocess input text
def preprocess_text(text):
    ps = PorterStemmer()
    review = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    review = review.lower().split()  # Convert to lowercase and split into words
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]  # Stem and remove stopwords
    return ' '.join(review)

# Streamlit App
st.title("Email Spam Detection App")
st.write("Upload an email or enter text to classify it as Spam or Ham!")

# Input options
input_method = st.radio("Choose input method:", ("Type text", "Upload file"))

if input_method == "Type text":
    user_input = st.text_area("Enter email content here:")
    if st.button("Classify"):
        if user_input.strip():
            processed_text = preprocess_text(user_input)
            vectorized_input = vectorizer.transform([processed_text])
            prediction = model.predict(vectorized_input)[0]
            st.success("This email is **SPAM**." if prediction == 1 else "This email is **HAM**.")
        else:
            st.error("Please enter some text to classify.")

elif input_method == "Upload file":
    uploaded_file = st.file_uploader("Upload a text file containing email content", type=["txt","eml","csv"])
    if uploaded_file:
        email_content = uploaded_file.read().decode("utf-8")
        st.write("Uploaded Email Content:")
        st.text(email_content)
        if st.button("Classify"):
            processed_text = preprocess_text(email_content)
            vectorized_input = vectorizer.transform([processed_text])
            prediction = model.predict(vectorized_input)[0]
            st.success("This email is **SPAM**." if prediction == 1 else "This email is **HAM**.")

# Function for standalone predictions
def predict_email_spam(email_text):
    """Predict if an email is spam or not."""
    processed_text = preprocess_text(email_text)
    email_vector = vectorizer.transform([processed_text]).toarray()  # Vectorize the input email
    prediction = model.predict(email_vector)  # Predict using the trained model
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Test the function with a sample email
sample_email = "Congratulations! You've won a $1000 gift card. Click here to claim."
if st.checkbox("Run Sample Test"):
    result = predict_email_spam(sample_email)
    st.write(f"The sample email is classified as: **{result}**")
