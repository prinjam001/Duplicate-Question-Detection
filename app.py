import streamlit as st
import pickle
import re

from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Duplicate Question Detector",
    page_icon="🔍",
    layout="centered"
)


# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

@st.cache_resource
def load_model():

    with open(
        "duplicate_question_svm.pkl",
        "rb"
    ) as file:
        model = pickle.load(file)

    with open(
        "tfidf_vectorizer.pkl",
        "rb"
    ) as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


model, vectorizer = load_model()


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def clean_text(text):

    # Convert input to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# DUPLICATE QUESTION PREDICTION
# ============================================================

def predict_duplicate(question1, question2):

    # --------------------------------------------------------
    # Step 1: Clean questions
    # --------------------------------------------------------

    q1_clean = clean_text(question1)
    q2_clean = clean_text(question2)


    # --------------------------------------------------------
    # Step 2: Convert questions into TF-IDF vectors
    # --------------------------------------------------------

    q1_vector = vectorizer.transform(
        [q1_clean]
    )

    q2_vector = vectorizer.transform(
        [q2_clean]
    )


    # --------------------------------------------------------
    # Step 3: Calculate TF-IDF difference features
    # --------------------------------------------------------

    q_diff = abs(
        q1_vector - q2_vector
    )


    # --------------------------------------------------------
    # Step 4: Calculate TF-IDF product features
    # --------------------------------------------------------

    q_product = q1_vector.multiply(
        q2_vector
    )


    # --------------------------------------------------------
    # Step 5: Combine pair features
    # --------------------------------------------------------

    q_pair_features = hstack([
        q_diff,
        q_product
    ]).tocsr()


    # --------------------------------------------------------
    # Step 6: Predict using Linear SVM
    # --------------------------------------------------------

    prediction = model.predict(
        q_pair_features
    )[0]


    # --------------------------------------------------------
    # Step 7: Calculate SVM decision score
    # --------------------------------------------------------

    decision_score = model.decision_function(
        q_pair_features
    )[0]


    # --------------------------------------------------------
    # Step 8: Calculate cosine similarity
    # --------------------------------------------------------

    similarity = cosine_similarity(
        q1_vector,
        q2_vector
    )[0][0]


    return prediction, decision_score, similarity


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("🔍 Duplicate Question Detection")

st.write(
    "Enter two questions below to determine whether "
    "they are likely to be duplicates."
)

st.markdown("---")


# ============================================================
# QUESTION INPUTS
# ============================================================

question1 = st.text_area(
    "📝 Question 1",
    placeholder="Enter the first question here...",
    height=120
)


question2 = st.text_area(
    "📝 Question 2",
    placeholder="Enter the second question here...",
    height=120
)


st.markdown("")


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔎 Check Duplicate",
    use_container_width=True
):

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not question1.strip():

        st.warning(
            "Please enter Question 1."
        )

    elif not question2.strip():

        st.warning(
            "Please enter Question 2."
        )

    else:

        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        prediction, decision_score, similarity = (
            predict_duplicate(
                question1,
                question2
            )
        )


        st.markdown("---")


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        if prediction == 1:

            st.success(
                "✅ These questions are likely DUPLICATE."
            )

        else:

            st.error(
                "❌ These questions are likely NOT DUPLICATE."
            )


        # ====================================================
        # DISPLAY METRICS
        # ====================================================

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Cosine Similarity",
                f"{similarity:.2%}"
            )


        with col2:

            st.metric(
                "SVM Decision Score",
                f"{decision_score:.4f}"
            )


        # ====================================================
        # SHOW CLEANED QUESTIONS
        # ====================================================

        with st.expander(
            "View Preprocessed Questions"
        ):

            st.write(
                "**Question 1 after preprocessing:**"
            )

            st.code(
                clean_text(question1)
            )


            st.write(
                "**Question 2 after preprocessing:**"
            )

            st.code(
                clean_text(question2)
            )


# ============================================================
# ABOUT PROJECT
# ============================================================

st.markdown("---")

st.subheader("📌 About This Project")

st.write(
    """
    This application detects duplicate questions using
    Natural Language Processing and classical Machine
    Learning techniques.

    The final model uses:

    • TF-IDF Vectorization
    • TF-IDF Difference Features
    • TF-IDF Product Features
    • Linear Support Vector Machine (SVM)
    • GridSearchCV for hyperparameter tuning

    The tuned model achieved approximately 80.60% test
    accuracy and 89.28% ROC-AUC.

    No Deep Learning is used in this project.
    """
)