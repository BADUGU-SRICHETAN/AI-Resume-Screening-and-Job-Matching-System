import streamlit as st
import pandas as pd
import plotly.express as px
import re

from rag import RAG_SYSTEM
from screening import screen_candidates, rank_candidates


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# INNOMATICS LOGO
# ============================================================

logo_col1, logo_col2, logo_col3 = st.columns([1, 2, 1])

with logo_col2:
    st.image(
        "Innomatics Logo.png",
        width=400
    )


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .dashboard-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .dashboard-subtitle {
        font-size: 16px;
        color: #777777;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.markdown(
    '<div class="dashboard-title">'
    'AI Resume Screening Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="dashboard-subtitle">'
    'AI-powered candidate ranking and resume analysis'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Resume Screening")

mode = st.sidebar.radio(
    "Select Mode",
    [
        "Resume Q&A",
        "Job Description Screening"
    ]
)


# ============================================================
# RESUME Q&A
# ============================================================

if mode == "Resume Q&A":

    st.header("Resume Q&A")

    st.write(
        "Ask questions about any candidate in the uploaded resumes."
    )

    query = st.text_input(
        "Ask your question",
        placeholder="Example: What are the skills of Srichetan Badugu?"
    )

    if st.button("Ask", type="primary"):

        if not query.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching resumes..."
            ):

                answer = RAG_SYSTEM(query)

            st.subheader("Answer")

            st.write(answer)


# ============================================================
# JOB DESCRIPTION SCREENING
# ============================================================

elif mode == "Job Description Screening":

    st.header(
        "Job Description Screening"
    )

    st.write(
        "Enter a Job Description to rank the most relevant candidates."
    )


    # --------------------------------------------------------
    # JOB DESCRIPTION INPUT
    # --------------------------------------------------------

    job_description = st.text_area(
        "Job Description",
        height=250,
        placeholder="""Example:

We are looking for a Data Scientist / Machine Learning Engineer.

Requirements:
- Python
- SQL
- NumPy
- Pandas
- Machine Learning
- Scikit-learn
- Feature Engineering
- Statistics
- Data Visualization

Preferred:
- Deep Learning
- NLP
- Generative AI
- RAG
- LangChain
- AI Agents

Education:
Bachelor's degree or equivalent.
"""
    )


    # --------------------------------------------------------
    # TOP K
    # --------------------------------------------------------

    top_k = st.number_input(
        "Number of Top Candidates",
        min_value=1,
        max_value=20,
        value=5
    )


    # --------------------------------------------------------
    # SCREEN CANDIDATES
    # --------------------------------------------------------

    if st.button(
        "Screen Candidates",
        type="primary"
    ):

        if not job_description.strip():

            st.warning(
                "Please enter a Job Description."
            )

        else:

            with st.spinner(
                "Analyzing Job Description and screening resumes..."
            ):

                requirements, candidate_results = screen_candidates(
                    job_description
                )

                ranked_candidates = rank_candidates(
                    candidate_results
                )


            # Store results in session state

            st.session_state[
                "requirements"
            ] = requirements

            st.session_state[
                "ranked_candidates"
            ] = ranked_candidates

            st.session_state[
                "job_description"
            ] = job_description


    # ========================================================
    # DISPLAY DASHBOARD
    # ========================================================

    if "ranked_candidates" in st.session_state:

        ranked_candidates = st.session_state[
            "ranked_candidates"
        ]

        requirements = st.session_state[
            "requirements"
        ]


        # ----------------------------------------------------
        # PREPARE DATA
        # ----------------------------------------------------

        dashboard_data = []

        for candidate in ranked_candidates:

            dashboard_data.append(
                {
                    "Candidate": candidate["candidate"],
                    "Match Score": candidate["score"]
                }
            )


        df = pd.DataFrame(
            dashboard_data
        )


        # ====================================================
        # SCREENING OVERVIEW
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Screening Overview'
            '</div>',
            unsafe_allow_html=True
        )


        total_candidates = len(
            ranked_candidates
        )


        if total_candidates > 0:

            average_score = df[
                "Match Score"
            ].mean()

            highest_score = df[
                "Match Score"
            ].max()

            top_candidate = ranked_candidates[
                0
            ]["candidate"]

        else:

            average_score = 0

            highest_score = 0

            top_candidate = "N/A"


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(
            4
        )


        with col1:

            st.metric(
                "Total Candidates",
                total_candidates
            )


        with col2:

            st.metric(
                "Average Match Score",
                f"{average_score:.0f}%"
            )


        with col3:

            st.metric(
                "Highest Match Score",
                f"{highest_score:.0f}%"
            )


        with col4:

            st.metric(
                "Top Candidate",
                top_candidate
            )


        st.divider()


        # ====================================================
        # TOP CANDIDATES
        # ====================================================

        top_candidates = ranked_candidates[
            :int(top_k)
        ]


        top_df = pd.DataFrame(
            [
                {
                    "Rank": index + 1,
                    "Candidate": candidate["candidate"],
                    "Match Score": candidate["score"]
                }

                for index, candidate
                in enumerate(top_candidates)
            ]
        )


        # ====================================================
        # CANDIDATE MATCH SCORE CHART
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Candidate Match Scores'
            '</div>',
            unsafe_allow_html=True
        )


        fig = px.bar(
            top_df,
            x="Candidate",
            y="Match Score",
            text="Match Score",
            title="Top Candidate Match Scores"
        )


        fig.update_traces(
            texttemplate="%{text:.0f}%",
            textposition="outside"
        )


        fig.update_layout(
            yaxis_title="Match Score (%)",
            xaxis_title="Candidate",
            yaxis=dict(
                range=[
                    0,
                    100
                ]
            )
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ====================================================
        # DISTRIBUTION + RANKING
        # ====================================================

        col1, col2 = st.columns(
            2
        )


        # ----------------------------------------------------
        # SCORE DISTRIBUTION
        # ----------------------------------------------------

        with col1:

            st.markdown(
                '<div class="section-title">'
                'Match Score Distribution'
                '</div>',
                unsafe_allow_html=True
            )


            fig_distribution = px.histogram(
                df,
                x="Match Score",
                nbins=10,
                title="Candidate Score Distribution"
            )


            fig_distribution.update_layout(
                xaxis_title="Match Score (%)",
                yaxis_title="Number of Candidates"
            )


            st.plotly_chart(
                fig_distribution,
                use_container_width=True
            )


        # ----------------------------------------------------
        # CANDIDATE RANKING
        # ----------------------------------------------------

        with col2:

            st.markdown(
                '<div class="section-title">'
                'Candidate Ranking'
                '</div>',
                unsafe_allow_html=True
            )


            ranking_df = top_df.copy()


            ranking_df["Match Score"] = (
                ranking_df["Match Score"]
                .map(
                    lambda x: f"{x:.0f}%"
                )
            )


            st.dataframe(
                ranking_df,
                hide_index=True,
                use_container_width=True
            )


        st.divider()


        # ====================================================
        # EXTRACTED JOB REQUIREMENTS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Extracted Job Requirements'
            '</div>',
            unsafe_allow_html=True
        )


        with st.expander(
            "View Job Requirements",
            expanded=False
        ):

            st.write(
                requirements
            )


        st.divider()


        # ====================================================
        # CANDIDATE ANALYSIS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Candidate Analysis'
            '</div>',
            unsafe_allow_html=True
        )


        candidate_names = [
            candidate["candidate"]
            for candidate
            in ranked_candidates
        ]


        selected_candidate = st.selectbox(
            "Select a candidate",
            candidate_names
        )


        selected_data = next(
            candidate
            for candidate
            in ranked_candidates
            if candidate["candidate"]
            == selected_candidate
        )


        # ----------------------------------------------------
        # SELECTED CANDIDATE HEADER
        # ----------------------------------------------------

        score_col1, score_col2 = st.columns(
            [3, 1]
        )


        with score_col1:

            st.subheader(
                selected_data["candidate"]
            )

            st.caption(
                selected_data["source"]
            )


        with score_col2:

            st.metric(
                "Match Score",
                f"{selected_data['score']:.0f}%"
            )


        # ====================================================
        # PARSE AI EVALUATION
        # ====================================================

        evaluation = selected_data[
            "evaluation"
        ]


        matched = ""

        missing = ""

        strengths = ""

        gaps = ""

        reason = ""


        matched_result = re.search(
            r"Matched Requirements:\s*(.*?)(?=\n\s*Missing Requirements:)",
            evaluation,
            re.DOTALL | re.IGNORECASE
        )


        missing_result = re.search(
            r"Missing Requirements:\s*(.*?)(?=\n\s*Candidate Strengths:)",
            evaluation,
            re.DOTALL | re.IGNORECASE
        )


        strengths_result = re.search(
            r"Candidate Strengths:\s*(.*?)(?=\n\s*Candidate Gaps:)",
            evaluation,
            re.DOTALL | re.IGNORECASE
        )


        gaps_result = re.search(
            r"Candidate Gaps:\s*(.*?)(?=\n\s*Reason:)",
            evaluation,
            re.DOTALL | re.IGNORECASE
        )


        reason_result = re.search(
            r"Reason:\s*(.*)",
            evaluation,
            re.DOTALL | re.IGNORECASE
        )


        if matched_result:

            matched = matched_result.group(
                1
            ).strip()


        if missing_result:

            missing = missing_result.group(
                1
            ).strip()


        if strengths_result:

            strengths = strengths_result.group(
                1
            ).strip()


        if gaps_result:

            gaps = gaps_result.group(
                1
            ).strip()


        if reason_result:

            reason = reason_result.group(
                1
            ).strip()


        # ====================================================
        # MATCHED / MISSING REQUIREMENTS
        # ====================================================

        col1, col2 = st.columns(
            2
        )


        with col1:

            st.subheader(
                "Matched Requirements"
            )

            st.info(
                matched
                if matched
                else "No matched requirements found."
            )


        with col2:

            st.subheader(
                "Missing Requirements"
            )

            st.warning(
                missing
                if missing
                else "No missing requirements found."
            )


        # ====================================================
        # STRENGTHS / GAPS
        # ====================================================

        col1, col2 = st.columns(
            2
        )


        with col1:

            st.subheader(
                "Candidate Strengths"
            )

            st.success(
                strengths
                if strengths
                else "No strengths found."
            )


        with col2:

            st.subheader(
                "Candidate Gaps"
            )

            st.warning(
                gaps
                if gaps
                else "No gaps found."
            )


        # ====================================================
        # AI EVALUATION REASON
        # ====================================================

        st.subheader(
            "AI Evaluation Reason"
        )


        st.write(
            reason
            if reason
            else "No evaluation reason found."
        )