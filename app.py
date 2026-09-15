import streamlit as st
import joblib
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")


# --------------------------------------------------
# Load Model and TF-IDF Vectorizer
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "fake_news_linear_svc.pkl")

VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")


@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


model, vectorizer = load_model()


# --------------------------------------------------
# Application UI
# --------------------------------------------------

st.title("📰 Fake News Detection System")

st.markdown("""
    ### 🔎 AI-powered news classification

    Enter a news headline and article below to determine
    whether the content is predicted to be **Fake** or **Real**.
    """)

st.divider()


# --------------------------------------------------
# User Input
# --------------------------------------------------

st.subheader("📝 News Article")

title = st.text_input(
    "News Title", placeholder="Enter the news headline...", key="news_title"
)

article_text = st.text_area(
    "Article Content",
    height=300,
    placeholder="Paste the complete news article here...",
    key="article_content",
)


st.caption("Model: LinearSVC | Features: TF-IDF")


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Detect Fake News", use_container_width=True):

    if not title.strip() and not article_text.strip():

        st.warning("⚠️ Please enter a news title or article.")

    else:

        # Combine title and article
        content = title.strip() + " " + article_text.strip()

        # Transform using TF-IDF
        text_tfidf = vectorizer.transform([content])

        # Prediction
        prediction = model.predict(text_tfidf)[0]

        # Decision score
        decision_score = model.decision_function(text_tfidf)[0]

        st.divider()

        st.subheader("📊 Prediction Result")

        if prediction == 1:

            st.success("✅ The article is predicted to be REAL NEWS")

        else:

            st.error("⚠️ The article is predicted to be FAKE NEWS")

        st.metric("Model Decision Score", f"{decision_score:.3f}")

        st.caption(
            "The decision score indicates the direction and "
            "strength of the model's classification. "
            "It is not a calibrated probability."
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "⚠️ This system is a machine-learning prediction tool "
    "and should not be treated as a definitive fact-checking system."
)
