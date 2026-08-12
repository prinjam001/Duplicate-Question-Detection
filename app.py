# ============================================================
# DUPLICATE QUESTION DETECTION APP
# ============================================================

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
    layout="wide"
)


# ============================================================
# LOAD MODEL AND TF-IDF VECTORIZER
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
# TWO QUESTION PREDICTION
# ============================================================

def predict_duplicate(question1, question2):

    # Clean questions

    q1_clean = clean_text(question1)

    q2_clean = clean_text(question2)


    # TF-IDF transformation

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


    # SVM decision score

    decision_score = model.decision_function(
        q_pair_features
    )[0]


    # Cosine similarity

    similarity = cosine_similarity(
        q1_vector,
        q2_vector
    )[0][0]


    return (
        prediction,
        decision_score,
        similarity
    )


# ============================================================
# PDF QUESTION EXTRACTION
# ============================================================

def extract_questions(pdf_text):

    # Normalize line endings

    pdf_text = pdf_text.replace(
        "\r",
        "\n"
    )


    # Split PDF into lines

    lines = pdf_text.split("\n")


    questions = []

    current_question = ""


    for line in lines:

        line = line.strip()


        # Ignore empty lines

        if not line:
            continue


        # ----------------------------------------------------
        # Detect numbered questions
        #
        # Examples:
        #
        # 1. What is Python?
        # 2. Explain Python.
        # Q1. What is Python?
        # Question 1: What is Python?
        # ----------------------------------------------------

        match = re.match(
            r"^(?:Q(?:uestion)?\s*)?"
            r"(\d+)"
            r"[\.\):\-]\s*(.*)$",
            line,
            flags=re.IGNORECASE
        )


        if match:

            # Save previous question

            if current_question.strip():

                questions.append(
                    current_question.strip()
                )


            # Start new question

            current_question = (
                match.group(2).strip()
            )


        else:

            # Continue current question

            if current_question:

                current_question += (
                    " " + line
                )


    # Add final question

    if current_question.strip():

        questions.append(
            current_question.strip()
        )


    return questions


# ============================================================
# PDF DUPLICATE ANALYSIS
# ============================================================

def analyze_pdf(
    questions,
    similarity_threshold
):

    # Clean questions

    cleaned_questions = [
        clean_text(question)
        for question in questions
    ]


    # Convert all questions to TF-IDF

    tfidf_matrix = vectorizer.transform(
        cleaned_questions
    )


    # Calculate cosine similarity

    similarity_matrix = cosine_similarity(
        tfidf_matrix
    )


    results = []


    # --------------------------------------------------------
    # Generate question pairs
    # --------------------------------------------------------

    for i in range(
        len(questions)
    ):

        for j in range(
            i + 1,
            len(questions)
        ):


            similarity = (
                similarity_matrix[i][j]
            )


            # Only process candidate pairs

            if similarity >= similarity_threshold:


                q1_vector = tfidf_matrix[i]

                q2_vector = tfidf_matrix[j]


                # Difference features

                q_diff = abs(
                    q1_vector - q2_vector
                )


                # Product features

                q_product = (
                    q1_vector.multiply(
                        q2_vector
                    )
                )


                # Combine features

                pair_features = hstack([
                    q_diff,
                    q_product
                ]).tocsr()


                # SVM prediction

                prediction = model.predict(
                    pair_features
                )[0]


                # Decision score

                decision_score = (
                    model.decision_function(
                        pair_features
                    )[0]
                )


                if prediction == 1:

                    prediction_label = (
                        "Duplicate"
                    )

                else:

                    prediction_label = (
                        "Not Duplicate"
                    )


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
                        prediction_label

                })


    return (
        pd.DataFrame(results),
        tfidf_matrix
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title(
    "🔍 Duplicate Question Detection"
)

st.write(
    """
This application detects duplicate questions using
Natural Language Processing and classical Machine Learning.

You can either:

1. Compare two questions manually.
2. Upload a PDF containing multiple questions and
   automatically identify duplicate question pairs.
"""
)


# ============================================================
# CREATE TABS
# ============================================================

tab1, tab2 = st.tabs([
    "📝 Compare Two Questions",
    "📄 Analyze PDF"
])


# ============================================================
# TAB 1 — TWO QUESTION COMPARISON
# ============================================================

with tab1:

    st.subheader(
        "📝 Compare Two Questions"
    )


    question1 = st.text_area(
        "Question 1",
        placeholder="Enter the first question...",
        height=120
    )


    question2 = st.text_area(
        "Question 2",
        placeholder="Enter the second question...",
        height=120
    )


    if st.button(
        "🔎 Check Duplicate",
        key="single_question_button",
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

            (
                prediction,
                decision_score,
                similarity
            ) = predict_duplicate(
                question1,
                question2
            )


            st.markdown("---")


            # Result

            if prediction == 1:

                st.success(
                    "✅ These questions are likely DUPLICATE."
                )

            else:

                st.error(
                    "❌ These questions are likely NOT DUPLICATE."
                )


            # Metrics

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


            # Preprocessed questions

            with st.expander(
                "View Preprocessed Questions"
            ):

                st.write(
                    "**Question 1:**"
                )

                st.code(
                    clean_text(question1)
                )


                st.write(
                    "**Question 2:**"
                )

                st.code(
                    clean_text(question2)
                )


# ============================================================
# TAB 2 — PDF ANALYSIS
# ============================================================

with tab2:

    st.subheader(
        "📄 Analyze Questions from PDF"
    )


    st.write(
        """
Upload a PDF containing numbered questions.


"""
    )


    uploaded_pdf = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        key="pdf_uploader"
    )


    # --------------------------------------------------------
    # PDF uploaded
    # --------------------------------------------------------

    if uploaded_pdf is not None:


        try:

            reader = PdfReader(
                uploaded_pdf
            )


            st.success(
                f"PDF uploaded successfully! "
                f"Number of pages: {len(reader.pages)}"
            )


            # Extract text

            pdf_text = ""


            for page in reader.pages:

                text = page.extract_text()


                if text:

                    pdf_text += (
                        text + "\n"
                    )


            # ------------------------------------------------
            # Check extracted text
            # ------------------------------------------------

            if not pdf_text.strip():

                st.error(
                    """
                    No text could be extracted from this PDF.

                    The PDF may be scanned/image-based.
                    OCR will be required for scanned PDFs.
                    """
                )


            else:


                # ------------------------------------------------
                # Extract questions
                # ------------------------------------------------

                questions = extract_questions(
                    pdf_text
                )


                st.subheader(
                    "📋 Detected Questions"
                )


                st.write(
                    f"Total questions detected: "
                    f"**{len(questions)}**"
                )


                # Display detected questions

                for i, question in enumerate(
                    questions
                ):

                    st.write(
                        f"**Q{i + 1}.** {question}"
                    )


                # ------------------------------------------------
                # Continue only if questions exist
                # ------------------------------------------------

                if len(questions) >= 2:


                    st.markdown("---")


                    # ------------------------------------------------
                    # Similarity threshold
                    # ------------------------------------------------

                    similarity_threshold = st.slider(

                        "Cosine Similarity Threshold",

                        min_value=0.0,

                        max_value=1.0,

                        value=0.30,

                        step=0.05,

                        help=(
                            "Only question pairs with "
                            "cosine similarity above this "
                            "threshold will be evaluated "
                            "by the SVM."
                        )
                    )


                    st.write(
                        f"Current threshold: "
                        f"**{similarity_threshold:.2f}**"
                    )


                    # ------------------------------------------------
                    # Analyze PDF button
                    # ------------------------------------------------

                    if st.button(
                        "🔎 Find Duplicate Questions",
                        key="pdf_analysis_button",
                        use_container_width=True
                    ):


                        with st.spinner(
                            "Analyzing question pairs..."
                        ):


                            (
                                results_df,
                                tfidf_matrix
                            ) = analyze_pdf(

                                questions,

                                similarity_threshold
                            )


                        st.markdown("---")


                        # ------------------------------------------------
                        # Summary
                        # ------------------------------------------------

                        if len(results_df) > 0:


                            duplicate_df = (
                                results_df[
                                    results_df[
                                        "Prediction"
                                    ] == "Duplicate"
                                ]
                            )


                            not_duplicate_df = (
                                results_df[
                                    results_df[
                                        "Prediction"
                                    ] == "Not Duplicate"
                                ]
                            )


                            col1, col2, col3 = (
                                st.columns(3)
                            )


                            with col1:

                                st.metric(
                                    "Questions Detected",
                                    len(questions)
                                )


                            with col2:

                                st.metric(
                                    "Duplicate Pairs",
                                    len(
                                        duplicate_df
                                    )
                                )


                            with col3:

                                st.metric(
                                    "Non-Duplicate Pairs",
                                    len(
                                        not_duplicate_df
                                    )
                                )


                            # ------------------------------------------------
                            # All candidate pairs
                            # ------------------------------------------------

                            st.markdown("---")

                            st.subheader(
                                "📊 All Candidate Question Pairs"
                            )


                            st.dataframe(
                                results_df,
                                use_container_width=True
                            )


                            # ------------------------------------------------
                            # Duplicate questions
                            # ------------------------------------------------

                            st.markdown("---")

                            st.subheader(
                                "🔴 Duplicate Questions"
                            )


                            if len(
                                duplicate_df
                            ) > 0:

                                st.dataframe(
                                    duplicate_df,
                                    use_container_width=True
                                )

                            else:

                                st.info(
                                    "No duplicate questions found."
                                )


                            # ------------------------------------------------
                            # Non-duplicate questions
                            # ------------------------------------------------

                            st.markdown("---")

                            st.subheader(
                                "🟢 Non-Duplicate Candidate Pairs"
                            )


                            if len(
                                not_duplicate_df
                            ) > 0:

                                st.dataframe(
                                    not_duplicate_df,
                                    use_container_width=True
                                )

                            else:

                                st.info(
                                    "No non-duplicate candidate pairs "
                                    "were found above the similarity "
                                    "threshold."
                                )


                            # ------------------------------------------------
                            # Download all results
                            # ------------------------------------------------

                            st.markdown("---")

                            csv = (
                                results_df
                                .to_csv(
                                    index=False
                                )
                                .encode("utf-8")
                            )


                            st.download_button(

                                label=(
                                    "⬇️ Download All Results"
                                ),

                                data=csv,

                                file_name=(
                                    "pdf_question_analysis.csv"
                                ),

                                mime="text/csv",

                                use_container_width=True
                            )


                            # ------------------------------------------------
                            # Download duplicate results
                            # ------------------------------------------------

                            if len(
                                duplicate_df
                            ) > 0:


                                duplicate_csv = (
                                    duplicate_df
                                    .to_csv(
                                        index=False
                                    )
                                    .encode(
                                        "utf-8"
                                    )
                                )


                                st.download_button(

                                    label=(
                                        "⬇️ Download "
                                        "Duplicate Questions"
                                    ),

                                    data=duplicate_csv,

                                    file_name=(
                                        "duplicate_questions.csv"
                                    ),

                                    mime="text/csv",

                                    use_container_width=True
                                )


                        else:

                            st.warning(
                                """
                                No candidate question pairs
                                were found.

                                Try lowering the cosine
                                similarity threshold.
                                """
                            )


                else:

                    st.warning(
                        "At least two questions are required "
                        "for duplicate detection."
                    )


        except Exception as e:

            st.error(
                f"Error while processing PDF: {e}"
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

Machine Learning Pipeline:

• Text Preprocessing
• TF-IDF Vectorization
• TF-IDF Difference Features
• TF-IDF Product Features
• Linear Support Vector Machine (SVM)
• Cosine Similarity
• GridSearchCV for hyperparameter tuning

The tuned Linear SVM achieved approximately:

• Accuracy: 80.60%
• F1 Score: 77.07%
• ROC-AUC: 89.28%

The application also supports PDF-based question
analysis.

Users can upload a PDF containing numbered questions,
extract questions automatically, compare candidate
question pairs, and download the duplicate detection
results.

No Deep Learning is used in this project.
"""
)
