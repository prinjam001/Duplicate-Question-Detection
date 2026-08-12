import streamlit as st
import pickle
import re
import pandas as pd

from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader


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

    # Clean questions
    q1_clean = clean_text(question1)
    q2_clean = clean_text(question2)

    # TF-IDF
    q1_vector = vectorizer.transform(
        [q1_clean]
    )

    q2_vector = vectorizer.transform(
        [q2_clean]
    )

    # Difference features
    q_diff = abs(
        q1_vector - q2_vector
    )

    # Product features
    q_product = q1_vector.multiply(
        q2_vector
    )

    # Combine features
    q_pair_features = hstack([
        q_diff,
        q_product
    ]).tocsr()

    # SVM prediction
    prediction = model.predict(
        q_pair_features
    )[0]

    # Decision score
    decision_score = model.decision_function(
        q_pair_features
    )[0]

    # Cosine similarity
    similarity = cosine_similarity(
        q1_vector,
        q2_vector
    )[0][0]

    return prediction, decision_score, similarity


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_pdf):

    reader = PdfReader(uploaded_pdf)

    pdf_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            pdf_text += text + "\n"

    return pdf_text, len(reader.pages)


# ============================================================
# EXTRACT QUESTIONS FROM PDF
# ============================================================

def extract_questions(pdf_text):

    # Normalize line breaks
    pdf_text = pdf_text.replace("\r", "\n")

    # Split text into lines
    lines = pdf_text.split("\n")

    questions = []

    current_question = ""

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Remove question numbering
        cleaned_line = re.sub(
            r"^\s*(?:Q(?:uestion)?\.?\s*)?\d+[\.\):\-]?\s*",
            "",
            line,
            flags=re.IGNORECASE
        )

        # If line contains ?
        if "?" in cleaned_line:

            current_question += " " + cleaned_line

            questions.append(
                current_question.strip()
            )

            current_question = ""

        else:

            current_question += " " + cleaned_line

    # Add remaining text if it looks like a question
    if current_question.strip():

        remaining = current_question.strip()

        if len(remaining.split()) >= 3:

            questions.append(
                remaining
            )

    return questions


# ============================================================
# FIND DUPLICATE QUESTIONS IN PDF
# ============================================================

def find_pdf_duplicates(
    questions,
    similarity_threshold=0.50
):

    results = []

    if len(questions) < 2:

        return pd.DataFrame(
            columns=[
                "Question 1",
                "Question 2",
                "Cosine Similarity",
                "SVM Decision Score",
                "Prediction"
            ]
        )

    # Clean all questions
    cleaned_questions = [
        clean_text(q)
        for q in questions
    ]

    # Convert all questions to TF-IDF
    tfidf_matrix = vectorizer.transform(
        cleaned_questions
    )

    # Calculate all cosine similarities
    similarity_matrix = cosine_similarity(
        tfidf_matrix
    )

    # Compare every pair
    for i in range(len(questions)):

        for j in range(
            i + 1,
            len(questions)
        ):

            similarity = similarity_matrix[i][j]

            # Only send likely candidates to SVM
            if similarity >= similarity_threshold:

                q1_vector = tfidf_matrix[i]
                q2_vector = tfidf_matrix[j]

                # Difference features
                q_diff = abs(
                    q1_vector - q2_vector
                )

                # Product features
                q_product = q1_vector.multiply(
                    q2_vector
                )

                # Pair features
                pair_features = hstack([
                    q_diff,
                    q_product
                ]).tocsr()

                # SVM prediction
                prediction = model.predict(
                    pair_features
                )[0]

                decision_score = (
                    model.decision_function(
                        pair_features
                    )[0]
                )

                if prediction == 1:

                    results.append({

                        "Question 1":
                            questions[i],

                        "Question 2":
                            questions[j],

                        "Cosine Similarity":
                            round(
                                similarity,
                                4
                            ),

                        "SVM Decision Score":
                            round(
                                decision_score,
                                4
                            ),

                        "Prediction":
                            "Duplicate"

                    })

    return pd.DataFrame(results)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title(
    "🔍 Duplicate Question Detection"
)

st.write(
    "Detect duplicate questions using "
    "NLP and Machine Learning."
)


# ============================================================
# MODE SELECTION
# ============================================================

mode = st.radio(

    "Choose an analysis mode:",

    [
        "📝 Compare Two Questions",
        "📄 Analyze Questions from PDF"
    ]
)


# ============================================================
# MODE 1: TWO QUESTIONS
# ============================================================

if mode == "📝 Compare Two Questions":

    st.markdown("---")

    question1 = st.text_area(

        "📝 Question 1",

        placeholder=(
            "Enter the first question here..."
        ),

        height=120
    )

    question2 = st.text_area(

        "📝 Question 2",

        placeholder=(
            "Enter the second question here..."
        ),

        height=120
    )

    st.markdown("")

    if st.button(
        "🔎 Check Duplicate",
        use_container_width=True
    ):

        if not question1.strip():

            st.warning(
                "Please enter Question 1."
            )

        elif not question2.strip():

            st.warning(
                "Please enter Question 2."
            )

        else:

            prediction, decision_score, similarity = (
                predict_duplicate(
                    question1,
                    question2
                )
            )

            st.markdown("---")

            if prediction == 1:

                st.success(
                    "✅ These questions are likely DUPLICATE."
                )

            else:

                st.error(
                    "❌ These questions are likely NOT DUPLICATE."
                )

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
# MODE 2: PDF ANALYSIS
# ============================================================

else:

    st.markdown("---")

    st.subheader(
        "📄 Analyze Questions from PDF"
    )

    st.write(
        "Upload a text-based PDF containing "
        "multiple questions."
    )

    uploaded_pdf = st.file_uploader(

        "Upload PDF",

        type=["pdf"]
    )

    if uploaded_pdf is not None:

        # ----------------------------------------------------
        # Extract PDF text
        # ----------------------------------------------------

        pdf_text, page_count = (
            extract_pdf_text(
                uploaded_pdf
            )
        )

        st.success(
            f"✅ PDF uploaded successfully! "
            f"Number of pages: {page_count}"
        )

        # ----------------------------------------------------
        # Display extracted text
        # ----------------------------------------------------

        with st.expander(
            "📖 View Extracted PDF Text"
        ):

            st.text_area(
                "Extracted Text",
                pdf_text,
                height=300
            )

        # ----------------------------------------------------
        # Extract questions
        # ----------------------------------------------------

        questions = extract_questions(
            pdf_text
        )

        st.subheader(
            "📝 Questions Detected"
        )

        st.write(
            f"Total questions detected: "
            f"**{len(questions)}**"
        )

        if len(questions) > 0:

            for i, question in enumerate(
                questions
            ):

                st.write(
                    f"**Q{i + 1}.** {question}"
                )

        else:

            st.warning(
                "No questions could be detected. "
                "Make sure the PDF contains selectable "
                "text and question marks."
            )

        # ----------------------------------------------------
        # Find duplicates
        # ----------------------------------------------------

        if len(questions) >= 2:

            st.markdown("---")

            similarity_threshold = st.slider(

                "Candidate similarity threshold",

                min_value=0.30,

                max_value=0.90,

                value=0.50,

                step=0.05
            )

            if st.button(
                "🔎 Find Duplicate Questions",
                use_container_width=True
            ):

                with st.spinner(
                    "Analyzing question pairs..."
                ):

                    results = find_pdf_duplicates(

                        questions,

                        similarity_threshold
                    )

                st.markdown("---")

                st.subheader(
                    "📊 Duplicate Question Results"
                )

                if len(results) > 0:

                    st.success(
                        f"Found {len(results)} "
                        f"potential duplicate pairs."
                    )

                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    # Download results
                    csv_data = results.to_csv(
                        index=False
                    )

                    st.download_button(

                        label=(
                            "⬇️ Download Duplicate "
                            "Results"
                        ),

                        data=csv_data,

                        file_name=(
                            "duplicate_questions.csv"
                        ),

                        mime="text/csv",

                        use_container_width=True
                    )

                else:

                    st.info(
                        "No duplicate question pairs "
                        "were found using the current "
                        "similarity threshold."
                    )


# ============================================================
# ABOUT PROJECT
# ============================================================

st.markdown("---")

st.subheader(
    "📌 About This Project"
)

st.write(
    """
This application detects duplicate questions using
Natural Language Processing and classical Machine
Learning techniques.

The model uses:

• TF-IDF Vectorization
• TF-IDF Difference Features
• TF-IDF Product Features
• Linear Support Vector Machine (SVM)
• Cosine Similarity
• GridSearchCV for hyperparameter tuning

The tuned Linear SVM achieved approximately
80.60% test accuracy and 89.28% ROC-AUC.

The application can also analyze questions extracted
from PDF documents.

No Deep Learning is used in this project.
"""
)
)
