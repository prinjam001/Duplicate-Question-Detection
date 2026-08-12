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
# LOAD TRAINED MODEL AND TF-IDF VECTORIZER
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
# SINGLE QUESTION PAIR PREDICTION
# ============================================================

def predict_duplicate(question1, question2):

    # --------------------------------------------------------
    # Clean questions
    # --------------------------------------------------------

    q1_clean = clean_text(question1)

    q2_clean = clean_text(question2)


    # --------------------------------------------------------
    # TF-IDF transformation
    # --------------------------------------------------------

    q1_vector = vectorizer.transform(
        [q1_clean]
    )

    q2_vector = vectorizer.transform(
        [q2_clean]
    )


    # --------------------------------------------------------
    # TF-IDF difference features
    # --------------------------------------------------------

    q_diff = abs(
        q1_vector - q2_vector
    )


    # --------------------------------------------------------
    # TF-IDF product features
    # --------------------------------------------------------

    q_product = q1_vector.multiply(
        q2_vector
    )


    # --------------------------------------------------------
    # Combine pair features
    # --------------------------------------------------------

    q_pair_features = hstack([
        q_diff,
        q_product
    ]).tocsr()


    # --------------------------------------------------------
    # SVM prediction
    # --------------------------------------------------------

    prediction = model.predict(
        q_pair_features
    )[0]


    # --------------------------------------------------------
    # SVM decision score
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
# PDF QUESTION EXTRACTION
# ============================================================

def extract_questions(pdf_text):

    # --------------------------------------------------------
    # Normalize line endings
    # --------------------------------------------------------

    pdf_text = pdf_text.replace(
        "\r",
        "\n"
    )


    # --------------------------------------------------------
    # Split PDF text into lines
    # --------------------------------------------------------

    lines = pdf_text.split("\n")


    questions = []

    current_question = ""


    # --------------------------------------------------------
    # Read each line
    # --------------------------------------------------------

    for line in lines:

        line = line.strip()


        # Ignore empty lines

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


        # ----------------------------------------------------
        # New question detected
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Continuation of current question
        # ----------------------------------------------------

        else:

            if current_question:

                current_question += (
                    " " + line
                )


    # --------------------------------------------------------
    # Save final question
    # --------------------------------------------------------

    if current_question.strip():

        questions.append(
            current_question.strip()
        )


    return questions


# ============================================================
# PDF DUPLICATE ANALYSIS
# ============================================================

def analyze_pdf(questions):

    # --------------------------------------------------------
    # Clean all questions
    # --------------------------------------------------------

    cleaned_questions = [
        clean_text(question)
        for question in questions
    ]


    # --------------------------------------------------------
    # Convert all questions into TF-IDF
    # --------------------------------------------------------

    tfidf_matrix = vectorizer.transform(
        cleaned_questions
    )


    # --------------------------------------------------------
    # Calculate cosine similarity matrix
    # --------------------------------------------------------

    similarity_matrix = cosine_similarity(
        tfidf_matrix
    )


    results = []


    # --------------------------------------------------------
    # Compare EVERY possible question pair
    # --------------------------------------------------------

    for i in range(
        len(questions)
    ):

        for j in range(
            i + 1,
            len(questions)
        ):


            # ------------------------------------------------
            # Cosine similarity
            # ------------------------------------------------

            similarity = (
                similarity_matrix[i][j]
            )


            # ------------------------------------------------
            # Get question vectors
            # ------------------------------------------------

            q1_vector = tfidf_matrix[i]

            q2_vector = tfidf_matrix[j]


            # ------------------------------------------------
            # Difference features
            # ------------------------------------------------

            q_diff = abs(
                q1_vector - q2_vector
            )


            # ------------------------------------------------
            # Product features
            # ------------------------------------------------

            q_product = (
                q1_vector.multiply(
                    q2_vector
                )
            )


            # ------------------------------------------------
            # Combine features
            # ------------------------------------------------

            pair_features = hstack([
                q_diff,
                q_product
            ]).tocsr()


            # ------------------------------------------------
            # SVM prediction
            # ------------------------------------------------

            prediction = model.predict(
                pair_features
            )[0]


            # ------------------------------------------------
            # SVM decision score
            # ------------------------------------------------

            decision_score = (
                model.decision_function(
                    pair_features
                )[0]
            )


            # ------------------------------------------------
            # Convert prediction to readable label
            # ------------------------------------------------

            if prediction == 1:

                prediction_label = (
                    "Duplicate"
                )

            else:

                prediction_label = (
                    "Not Duplicate"
                )


            # ------------------------------------------------
            # Save result
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


    # --------------------------------------------------------
    # Convert results to DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )


    return results_df


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

You can:

• Compare two questions manually.

• Upload a PDF containing multiple questions.

• Detect duplicate and non-duplicate question pairs.

• Download the analysis results as CSV.
"""
)


# ============================================================
# CREATE APPLICATION TABS
# ============================================================

tab1, tab2 = st.tabs([
    "📝 Compare Two Questions",
    "📄 Analyze PDF"
])


# ============================================================
# TAB 1 — MANUAL QUESTION COMPARISON
# ============================================================

with tab1:

    st.subheader(
        "📝 Compare Two Questions"
    )


    # --------------------------------------------------------
    # Question 1
    # --------------------------------------------------------

    question1 = st.text_area(
        "Question 1",
        placeholder="Enter the first question...",
        height=120
    )


    # --------------------------------------------------------
    # Question 2
    # --------------------------------------------------------

    question2 = st.text_area(
        "Question 2",
        placeholder="Enter the second question...",
        height=120
    )


    st.markdown("")


    # --------------------------------------------------------
    # Prediction button
    # --------------------------------------------------------

    if st.button(
        "🔎 Check Duplicate",
        key="single_question_button",
        use_container_width=True
    ):


        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if not question1.strip():

            st.warning(
                "Please enter Question 1."
            )


        elif not question2.strip():

            st.warning(
                "Please enter Question 2."
            )


        else:

            # ------------------------------------------------
            # Make prediction
            # ------------------------------------------------

            (
                prediction,
                decision_score,
                similarity
            ) = predict_duplicate(
                question1,
                question2
            )


            st.markdown("---")


            # ------------------------------------------------
            # Display result
            # ------------------------------------------------

            if prediction == 1:

                st.success(
                    "✅ These questions are likely DUPLICATE."
                )

            else:

                st.error(
                    "❌ These questions are likely NOT DUPLICATE."
                )


            # ------------------------------------------------
            # Display metrics
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Show preprocessed questions
            # ------------------------------------------------

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

Recommended format:

1. What is machine learning?

2. What do you mean by machine learning?

3. How can I learn Python?

4. What is the best way to learn Python?
"""
    )


    # --------------------------------------------------------
    # Upload PDF
    # --------------------------------------------------------

    uploaded_pdf = st.file_uploader(
        "Upload a PDF containing questions",
        type=["pdf"],
        key="pdf_uploader"
    )


    # --------------------------------------------------------
    # Process uploaded PDF
    # --------------------------------------------------------

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
            # Extract PDF text
            # ------------------------------------------------

            pdf_text = ""


            for page in reader.pages:

                text = page.extract_text()


                if text:

                    pdf_text += (
                        text + "\n"
                    )


            # ------------------------------------------------
            # Check PDF text
            # ------------------------------------------------

            if not pdf_text.strip():

                st.error(
                    """
No text could be extracted from this PDF.

This may be a scanned/image-based PDF.
OCR would be required for such PDFs.
"""
                )


            else:


                # ------------------------------------------------
                # Extract questions
                # ------------------------------------------------

                questions = extract_questions(
                    pdf_text
                )


                # ------------------------------------------------
                # Display detected questions
                # ------------------------------------------------

                st.subheader(
                    "📋 Detected Questions"
                )


                st.write(
                    f"Total questions detected: "
                    f"**{len(questions)}**"
                )


                for i, question in enumerate(
                    questions
                ):

                    st.write(
                        f"**Q{i + 1}.** {question}"
                    )


                # ------------------------------------------------
                # Need at least two questions
                # ------------------------------------------------

                if len(questions) >= 2:


                    st.markdown("---")


                    # ------------------------------------------------
                    # Analyze button
                    # ------------------------------------------------

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
                        # Separate predictions
                        # ------------------------------------------------

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


                        # ------------------------------------------------
                        # Summary metrics
                        # ------------------------------------------------

                        col1, col2, col3, col4 = (
                            st.columns(4)
                        )


                        with col1:

                            st.metric(
                                "Questions Detected",
                                len(questions)
                            )


                        with col2:

                            st.metric(
                                "Duplicate Pairs",
                                len(duplicate_df)
                            )


                        with col3:

                            st.metric(
                                "Non-Duplicate Pairs",
                                len(not_duplicate_df)
                            )


                        with col4:

                            st.metric(
                                "Total Pairs Analyzed",
                                len(results_df)
                            )


                        # ------------------------------------------------
                        # Expected number of pairs
                        # ------------------------------------------------

                        expected_pairs = (
                            len(questions)
                            * (len(questions) - 1)
                            // 2
                        )


                        st.info(
                            f"""
Expected possible question pairs:
**{expected_pairs}**

Actually analyzed:
**{len(results_df)}**
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
                        # DUPLICATE RESULTS
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
                                "No duplicate question pairs found."
                            )


                        # ------------------------------------------------
                        # NON-DUPLICATE RESULTS
                        # ------------------------------------------------

                        st.markdown("---")


                        st.subheader(
                            "🟢 Non-Duplicate Question Pairs"
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
                                "No non-duplicate question pairs found."
                            )


                        # ------------------------------------------------
                        # DOWNLOAD ALL RESULTS
                        # ------------------------------------------------

                        st.markdown("---")


                        st.subheader(
                            "⬇️ Download Results"
                        )


                        csv = (
                            results_df
                            .to_csv(
                                index=False
                            )
                            .encode(
                                "utf-8"
                            )
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
                        # DOWNLOAD DUPLICATE RESULTS
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


                        # ------------------------------------------------
                        # DOWNLOAD NON-DUPLICATE RESULTS
                        # ------------------------------------------------

                        if len(
                            not_duplicate_df
                        ) > 0:


                            non_duplicate_csv = (
                                not_duplicate_df
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
                                    "Non-Duplicate Questions"
                                ),

                                data=non_duplicate_csv,

                                file_name=(
                                    "non_duplicate_questions.csv"
                                ),

                                mime="text/csv",

                                use_container_width=True
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

Final Tuned Linear SVM Performance:

• Accuracy: approximately 80.60%
• F1 Score: approximately 77.07%
• ROC-AUC: approximately 89.28%

PDF Analysis:

• Extracts text from PDF files
• Detects numbered questions
• Ignores PDF titles and descriptions
• Compares every possible question pair
• Uses the trained Linear SVM for classification
• Shows duplicate and non-duplicate pairs
• Provides downloadable CSV reports

No Deep Learning is used in this project.
"""
)
