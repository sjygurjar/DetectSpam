import streamlit as st  # type: ignore
import joblib
import re
import io
import pandas as pd
from nltk.corpus import stopwords  # type: ignore
from nltk.stem.porter import PorterStemmer  # type: ignore
import nltk

nltk.download('stopwords', quiet=True)

# Load the trained model and vectorizer
try:
    model = joblib.load("spam_detector.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure 'spam_detector.pkl' and 'vectorizer.pkl' are in the same directory.")
    st.stop()

# BUG FIX: load stopwords ONCE as a set instead of re-fetching the list
# inside the per-word comprehension on every single classification.
STOP_WORDS = set(stopwords.words('english'))
ps = PorterStemmer()


def preprocess_text(text):
    """Clean and stem input text for the bag-of-words model.

    BUG FIX: keep digits, $ and % instead of stripping every non-letter
    character — prices, phone numbers and percentages are strong spam
    signals that the original regex threw away entirely. Must match the
    preprocessing used in train_model.py or the vectorizer's vocabulary
    won't line up with the input.
    """
    review = re.sub(r'[^a-zA-Z0-9$%]', ' ', text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in STOP_WORDS]
    return ' '.join(review)


def classify(text):
    """Single source of truth for prediction, used by every input path
    below so behavior can't drift between them (the old code duplicated
    this logic three times, inconsistently)."""
    processed_text = preprocess_text(text)
    vectorized_input = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_input)[0]
    confidence = model.predict_proba(vectorized_input)[0][prediction]
    return prediction, confidence


def show_result(prediction, confidence):
    label = "SPAM" if prediction == 1 else "HAM"
    st.success(f"This email is **{label}** (confidence: {confidence:.0%}).")


# Streamlit App
st.title("Email Spam Detection App")
st.write("Upload an email or enter text to classify it as Spam or Ham!")

# Input options
input_method = st.radio("Choose input method:", ("Type text", "Upload file"))

if input_method == "Type text":
    user_input = st.text_area("Enter email content here:")
    if st.button("Classify"):
        if user_input.strip():
            prediction, confidence = classify(user_input)
            show_result(prediction, confidence)
        else:
            st.error("Please enter some text to classify.")

elif input_method == "Upload file":
    uploaded_file = st.file_uploader(
        "Upload a text/email file, or a CSV with a 'message' column",
        type=["txt", "eml", "csv"],
    )
    if uploaded_file:
        # BUG FIX: decode() with no error handling crashed on non-UTF-8
        # files (.eml files are frequently Windows-1252, base64, or
        # quoted-printable). Fall back gracefully instead of raising.
        raw_bytes = uploaded_file.read()
        try:
            file_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            file_text = raw_bytes.decode("utf-8", errors="replace")
            st.warning("File wasn't valid UTF-8 — some characters were replaced while reading it.")

        if uploaded_file.name.lower().endswith(".csv"):
            # BUG FIX: the original code dumped the *entire raw CSV text*
            # (headers, commas, every row) into the classifier as if it
            # were one email. Now we actually parse it and classify each
            # message in a 'message' (or first text-like) column.
            try:
                df = pd.read_csv(io.StringIO(file_text))
            except Exception as e:
                st.error(f"Couldn't parse this as a CSV: {e}")
                df = None

            if df is not None:
                text_col = "message" if "message" in df.columns else df.columns[0]
                st.write(f"Classifying {len(df)} rows from column '{text_col}':")
                results = []
                for msg in df[text_col].astype(str):
                    prediction, confidence = classify(msg)
                    results.append("SPAM" if prediction == 1 else "HAM")
                df["prediction"] = results
                st.dataframe(df[[text_col, "prediction"]])
        else:
            st.write("Uploaded Email Content:")
            st.text(file_text)
            if st.button("Classify"):
                prediction, confidence = classify(file_text)
                show_result(prediction, confidence)

# Optional sample test
sample_email = "Congratulations! You've won a $1000 gift card. Click here to claim."
if st.checkbox("Run Sample Test"):
    prediction, confidence = classify(sample_email)
    label = "Spam" if prediction == 1 else "Not Spam"
    st.write(f"The sample email is classified as: **{label}** (confidence: {confidence:.0%})")
