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

    # Lowercase
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
# CREATE SVM PAIR FEATURES
# IMPORTANT:
# This must match the notebook exactly.
# ============================================================

def create_pair_features(
    q1_vector,
    q2_vector
):

    # TF-IDF difference
    q_diff = abs(
        q1_vector - q2_vector
    )

    # TF-IDF product
    q_product = q1_vector.multiply(
        q2_vector
    )

    # Combine in SAME ORDER as notebook
    q_pair_features = hstack([
        q_diff,
        q_product
    ]).tocsr()

    return q_pair_features


# ============================================================
# PREDICT TWO QUESTIONS
# ============================================================

def predict_duplicate(
    question1,
    question2
):

    # --------------------------------------------------------
    # Clean questions
    # --------------------------------------------------------

    q1_clean = clean_text(
        question1
    )

    q2_clean = clean_text(
        question2
    )


    # --------------------------------------------------------
    # Convert to TF-IDF
    # --------------------------------------------------------

    q1_vector = vectorizer.transform(
        [q1_clean]
    )

    q2_vector = vectorizer.transform(
        [q2_clean]
    )


    # --------------------------------------------------------
    # Create pair features
    # --------------------------------------------------------

    q_pair_features = create_pair_features(
        q1_vector,
        q2_vector
    )


    # --------------------------------------------------------
    # SVM prediction
    # --------------------------------------------------------

    prediction = model.predict(
        q_pair_features
    )[0]


    # --------------------------------------------------------
    # Decision score
    # --------------------------------------------------------

    decision_score = model.decision_function(
        q_pair_features
    )[0]


    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------

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
# EXTRACT QUESTIONS FROM PDF
# ============================================================

def extract_questions(pdf_text):

    # Normalize line breaks
    pdf_text = pdf_text.replace(
        "\r",
        "\n"
    )

    lines = pdf_text.split("\n")

    questions = []

    current_question = ""

    for line in lines:

        line = line.strip()

        if not line:
            continue


        # ----------------------------------------------------
        # Detect numbered questions
        #
        
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

            # Continuation of previous question
            if current_question:

                current_question += (
                    " " + line
                )


    # Save final question
    if current_question.strip():

        questions.append(
            current_question.strip()
        )

    return questions


# ============================================================
# ANALYZE ALL PDF QUESTION PAIRS
# ============================================================

def analyze_pdf(questions):

    # --------------------------------------------------------
    # Clean all questions
    # --------------------------------------------------------

    cleaned_questions = [
        clean_text(q)
        for q in questions
    ]


    # --------------------------------------------------------
    # TF-IDF transformation
    # --------------------------------------------------------

    tfidf_matrix = vectorizer.transform(
        cleaned_questions
    )


    results = []


    # --------------------------------------------------------
    # Compare every possible pair
    # --------------------------------------------------------

    for i in range(
        len(questions)
    ):

        for j in range(
            i + 1,
            len(questions)
        ):

            # ------------------------------------------------
            # Get TF-IDF vectors
            # ------------------------------------------------

            q1_vector = tfidf_matrix[i]
            q2_vector = tfidf_matrix[j]


            # ------------------------------------------------
            # CREATE FEATURES
            # SAME FUNCTION AS MANUAL PREDICTION
            # ------------------------------------------------

            q_pair_features = create_pair_features(
                q1_vector,
                q2_vector
            )


            # ------------------------------------------------
            # SVM prediction
            # ------------------------------------------------

            prediction = model.predict(
                q_pair_features
            )[0]


            # ------------------------------------------------
            # Decision score
            # ------------------------------------------------

            decision_score = model.decision_function(
                q_pair_features
            )[0]


            # ------------------------------------------------
            # Cosine similarity
            # ------------------------------------------------

            similarity = cosine_similarity(
                q1_vector,
                q2_vector
            )[0][0]


            # ------------------------------------------------
            # Prediction label
            # ------------------------------------------------

            if prediction == 1 and similarity >= 0.20:

                prediction_label = "Duplicate"

            else:

                prediction_label = "Not Duplicate"


            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

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


    return pd.DataFrame(
        results
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
Natural Language Processing and classical Machine
Learning techniques.
"""
)

st.markdown("---")


# ============================================================
# TABS
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
        placeholder="Enter the first question here...",
        height=120
    )


    question2 = st.text_area(
        "Question 2",
        placeholder="Enter the second question here...",
        height=120
    )


    st.markdown("")


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
        "Upload a PDF containing questions",
        type=["pdf"],
        key="pdf_uploader"
    )


    if uploaded_pdf is not None:

        try:

            # ------------------------------------------------
            # Read PDF
            # ------------------------------------------------

            reader = PdfReader(
                uploaded_pdf
            )


            st.success(
                f"PDF uploaded successfully! "
                f"Number of pages: {len(reader.pages)}"
            )


            # ------------------------------------------------
            # Extract text
            # ------------------------------------------------

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

The PDF may be scanned or image-based.
OCR would be required for such PDFs.
"""
                )

            else:

                # ------------------------------------------------
                # Extract numbered questions
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


                # ------------------------------------------------
                # Display detected questions
                # ------------------------------------------------

                for i, question in enumerate(
                    questions
                ):

                    st.write(
                        f"**Q{i + 1}.** {question}"
                    )


                # ------------------------------------------------
                # Need at least 2 questions
                # ------------------------------------------------

                if len(questions) >= 2:

                    st.markdown("---")


                    if st.button(
                        "🔎 Find Duplicate Questions",
                        key="pdf_analysis_button",
                        use_container_width=True
                    ):

                        with st.spinner(
                            "Analyzing all question pairs..."
                        ):

                            results_df = analyze_pdf(
                                questions
                            )


                        st.markdown("---")


                        # ------------------------------------------------
                        # Separate duplicate / non-duplicate
                        # ------------------------------------------------

                        duplicate_df = (
                            results_df[
                                results_df[
                                    "Prediction"
                                ] == "Duplicate"
                            ]
                        )


                        non_duplicate_df = (
                            results_df[
                                results_df[
                                    "Prediction"
                                ] == "Not Duplicate"
                            ]
                        )


                        # ------------------------------------------------
                        # Count possible pairs
                        # ------------------------------------------------

                        total_questions = len(
                            questions
                        )


                        expected_pairs = (
                            total_questions
                            * (total_questions - 1)
                            // 2
                        )


                        # ------------------------------------------------
                        # Summary
                        # ------------------------------------------------

                        st.subheader(
                            "📊 Analysis Summary"
                        )


                        col1, col2, col3, col4 = (
                            st.columns(4)
                        )


                        with col1:

                            st.metric(
                                "Questions Detected",
                                total_questions
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
                                    non_duplicate_df
                                )
                            )


                        with col4:

                            st.metric(
                                "Total Pairs",
                                len(
                                    results_df
                                )
                            )


                        st.info(
                            f"""
Expected pairs: **{expected_pairs}**

Analyzed pairs: **{len(results_df)}**

Duplicate + Non-Duplicate:
**{len(duplicate_df) + len(non_duplicate_df)}**
"""
                        )


                        # ------------------------------------------------
                        # ALL RESULTS
                        # ------------------------------------------------

                        st.markdown("---")

                        st.subheader(
                            "📊 All Question Pair Results"
                        )


                        st.dataframe(
                            results_df,
                            use_container_width=True,
                            height=500
                        )


                        # ------------------------------------------------
                        # DUPLICATES
                        # ------------------------------------------------

                        st.markdown("---")

                        st.subheader(
                            "🔴 Duplicate Question Pairs"
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
                                "No duplicate pairs found."
                            )


                        # ------------------------------------------------
                        # NON-DUPLICATES
                        # ------------------------------------------------

                        st.markdown("---")

                        st.subheader(
                            "🟢 Non-Duplicate Question Pairs"
                        )


                        if len(
                            non_duplicate_df
                        ) > 0:

                            st.dataframe(
                                non_duplicate_df,
                                use_container_width=True
                            )

                        else:

                            st.info(
                                "No non-duplicate pairs found."
                            )


                        # ------------------------------------------------
                        # DOWNLOAD ALL
                        # ------------------------------------------------

                        st.markdown("---")

                        st.subheader(
                            "⬇️ Download Results"
                        )


                        all_csv = (
                            results_df
                            .to_csv(
                                index=False
                            )
                            .encode(
                                "utf-8"
                            )
                        )


                        st.download_button(
                            "⬇️ Download All Results",
                            data=all_csv,
                            file_name="pdf_question_analysis.csv",
                            mime="text/csv",
                            use_container_width=True
                        )


                        # ------------------------------------------------
                        # DOWNLOAD DUPLICATES
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
                                "⬇️ Download Duplicate Questions",
                                data=duplicate_csv,
                                file_name="duplicate_questions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )


                        # ------------------------------------------------
                        # DOWNLOAD NON-DUPLICATES
                        # ------------------------------------------------

                        if len(
                            non_duplicate_df
                        ) > 0:

                            non_duplicate_csv = (
                                non_duplicate_df
                                .to_csv(
                                    index=False
                                )
                                .encode(
                                    "utf-8"
                                )
                            )


                            st.download_button(
                                "⬇️ Download Non-Duplicate Questions",
                                data=non_duplicate_csv,
                                file_name="non_duplicate_questions.csv",
                                mime="text/csv",
                                use_container_width=True
                            )


                else:

                    st.warning(
                        "At least two numbered questions "
                        "are required."
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

Final Tuned Linear SVM Performance:

• Accuracy: approximately 80.60%
• F1 Score: approximately 77.07%
• ROC-AUC: approximately 89.28%

PDF Analysis:

• Extracts text from PDF files
• Detects numbered questions
• Compares every possible question pair
• Uses the trained Linear SVM
• Shows duplicate pairs
• Shows non-duplicate pairs
• Provides CSV downloads

No Deep Learning is used in this project.
"""
)
