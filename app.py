import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from query_router import process_query
from screening import (
    screen_candidates,
    rank_candidates,
    calculate_match_counts,
    get_display_name,
)


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Resumind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. THEME
# ============================================================

COLORS = {
    "background": "#0a1020",
    "panel": "#111a2d",
    "border": "#27344d",
    "text": "#edf2ff",
    "muted": "#a2aec5",
    "accent": "#8b8cff",
    "green": "#22c55e",
    "yellow": "#eab308",
    "red": "#ef4444",
}


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                ellipse at 8% 0%,
                rgba(99, 102, 241, 0.15),
                transparent 34%
            ),
            radial-gradient(
                ellipse at 95% 8%,
                rgba(45, 212, 191, 0.08),
                transparent 28%
            ),
            #0a1020;
        color: #edf2ff;
    }

    .block-container {
        max-width: 1540px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    .brand-label {
        color: #a5a6ff;
        font-size: 0.82rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 0.5rem;
    }

    .main-title {
        font-size: clamp(2.2rem, 4vw, 3.4rem);
        line-height: 1.12;
        font-weight: 850;
        color: #f8faff;
        letter-spacing: -1.5px;
        margin-bottom: 0.6rem;
    }

    .main-subtitle {
        color: #a2aec5;
        font-size: 1rem;
        line-height: 1.8;
        margin-bottom: 1.3rem;
    }

    h1, h2, h3, h4 {
        color: #f3f6ff !important;
        letter-spacing: -0.35px;
    }

    [data-testid="stCaptionContainer"] {
        color: #a2aec5 !important;
    }

    .stTextInput input,
    .stTextArea textarea {
        background: #0f1728 !important;
        color: #f8fafc !important;
        border: 1px solid #2d3b57 !important;
        border-radius: 12px !important;
    }

    .stTextInput input {
        min-height: 48px;
    }

    .stTextArea textarea {
        font-size: 0.93rem !important;
        line-height: 1.6;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #7786a2 !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(135deg, #6d70f6, #5659d9);
        color: white;
        border: 1px solid rgba(174, 176, 255, 0.28);
        border-radius: 11px;
        min-height: 44px;
        padding: 0.45rem 1.1rem;
        font-weight: 700;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #a5a6ff;
        color: white;
        filter: brightness(1.1);
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            rgba(22, 32, 54, 0.98),
            rgba(15, 23, 40, 0.98)
        );
        border: 1px solid #27344d;
        border-radius: 15px;
        padding: 1rem 1.1rem;
        min-height: 105px;
    }

    div[data-testid="stMetricLabel"] {
        color: #a2aec5 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f8faff !important;
        font-weight: 750;
    }

    div[data-testid="stExpander"] {
        background: rgba(17, 26, 45, 0.84);
        border: 1px solid #27344d;
        border-radius: 13px;
        overflow: hidden;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #27344d;
        border-radius: 12px;
        overflow: hidden;
    }

    a {
        color: #a9b4ff !important;
    }

    hr {
        border-color: #26334b !important;
        margin-top: 1.4rem;
        margin-bottom: 1.4rem;
    }

    .footer {
        text-align: center;
        color: #74819b;
        font-size: 0.8rem;
        margin-top: 2.8rem;
        line-height: 1.8;
    }

    @media (max-width: 720px) {
        .block-container {
            padding-top: 1.2rem;
        }

        .main-title {
            letter-spacing: -0.7px;
        }

        div[data-testid="stMetric"] {
            padding: 0.8rem;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            transition: none !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def safe_score(candidate):
    """Return a valid score between 0 and 100."""

    try:
        score = float(candidate.get("score", 0) or 0)
    except (ValueError, TypeError):
        return 0.0

    if pd.isna(score):
        return 0.0

    return max(0.0, min(100.0, score))


def score_category(score):
    """Classify candidates using fixed score thresholds."""

    if score >= 80:
        return "Strong score (80%+)"

    if score >= 60:
        return "Potential score (60–79%)"

    return "Low score (<60%)"


def get_candidate_name(candidate):
    name = candidate.get("candidate", "Unknown Candidate")
    return get_display_name(name)


def prepare_candidate_dataframe(candidates):
    rows = []

    for candidate in candidates:
        score = safe_score(candidate)

        rows.append({
            "Candidate": get_candidate_name(candidate),
            "Score": score,
            "Category": score_category(score),
            "Reason": candidate.get("reason", ""),
            "Email": candidate.get("email") or "Not Available",
            "Phone": candidate.get("phone") or "Not Available",
            "LinkedIn": candidate.get("linkedin") or "Not Available",
            "GitHub": candidate.get("github") or "Not Available",
        })

    return pd.DataFrame(rows)


def show_candidate_contact(candidate):
    email = candidate.get("email") or "Not Available"
    phone = candidate.get("phone") or "Not Available"
    linkedin = candidate.get("linkedin") or "Not Available"
    github = candidate.get("github") or "Not Available"

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📧 Email**")
        st.write(email)

        st.markdown("**📱 Mobile**")
        st.write(phone)

    with col2:
        st.markdown("**🔗 LinkedIn**")

        if linkedin != "Not Available":
            st.markdown(f"[Open LinkedIn profile]({linkedin})")
        else:
            st.write("Not Available")

        st.markdown("**💻 GitHub**")

        if github != "Not Available":
            st.markdown(f"[Open GitHub profile]({github})")
        else:
            st.write("Not Available")


def show_candidate_cards(candidates, heading):
    st.markdown(f"### {heading}")

    if not candidates:
        st.info("No candidates to display.")
        return

    for index, candidate in enumerate(candidates, start=1):
        score = safe_score(candidate)
        name = get_candidate_name(candidate)
        category = score_category(score)

        with st.expander(
            f"#{index} · {name} · {score:.1f}% · {category}",
            expanded=(index == 1),
        ):
            st.markdown("**Evaluation explanation**")
            st.write(
                candidate.get(
                    "reason",
                    "No evaluation explanation provided.",
                )
            )

            st.divider()
            show_candidate_contact(candidate)


def make_candidate_chart(dataframe, title):
    """Create a consistent score-based candidate chart."""

    chart_df = dataframe.sort_values(
        "Score",
        ascending=True,
    ).copy()

    color_map = {
        "Strong score (80%+)": COLORS["green"],
        "Potential score (60–79%)": COLORS["yellow"],
        "Low score (<60%)": COLORS["red"],
    }

    fig = px.bar(
        chart_df,
        x="Score",
        y="Candidate",
        color="Category",
        color_discrete_map=color_map,
        orientation="h",
        text=chart_df["Score"].map(
            lambda score: f"{score:.1f}%"
        ),
        hover_data=["Reason"],
        title=title,
    )

    fig.update_traces(
        textposition="outside",
        cliponaxis=False,
    )

    fig.update_layout(
        height=max(350, min(850, 38 * len(chart_df))),
        xaxis_title="Estimated Match Score (%)",
        xaxis_range=[0, 110],
        yaxis_title="Candidate",
        plot_bgcolor=COLORS["panel"],
        paper_bgcolor=COLORS["panel"],
        font_color=COLORS["text"],
        margin=dict(l=10, r=45, t=60, b=45),
        legend_title_text="Score Category",
    )

    return fig


# ============================================================
# 5. RESUMIND AI HEADER
# ============================================================

st.markdown(
    '<div class="brand-label">AI-Powered Recruitment Intelligence</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🧠 Resumind AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-subtitle">
        <strong>Smarter Resumes. Better Matches.</strong>
        <br>
        Intelligent Resume Screening, Semantic Search
        & Candidate Intelligence.
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# 6. MAIN INTERFACE
# ============================================================

left_col, right_col = st.columns([1, 1], gap="large")


# ============================================================
# 7. ASK ABOUT RESUMES
# ============================================================

with left_col:
    st.markdown("## 💬 Ask About Resumes")

    st.caption(
        "Ask questions about candidate skills, projects, "
        "education, experience or contact details."
    )

    with st.form("resume_query_form"):
        query = st.text_input(
            "Resume Query",
            placeholder=(
                "Example: Which candidates know Python and SQL?"
            ),
        )

        ask_button = st.form_submit_button(
            "🔎 Search Resumes",
            use_container_width=True,
        )

    if ask_button:
        if not query.strip():
            st.warning("Please enter a resume-related question.")

        else:
            with st.spinner("Analyzing resumes..."):
                try:
                    result = process_query(query.strip())

                    st.session_state["last_query"] = query.strip()
                    st.session_state["last_query_result"] = result

                except Exception as error:
                    st.error(
                        f"Could not process the query: {error}"
                    )

    result = st.session_state.get("last_query_result")

    if result:
        if result.get("type") == "resume_qa":
            st.markdown("### 💡 Answer")

            st.success(
                result.get(
                    "answer",
                    "No answer was returned.",
                )
            )

        elif result.get("type") == "candidate_search":
            query_candidates = result.get("results", [])

            st.markdown("### 🏆 Relevant Candidates")

            if not query_candidates:
                st.info("No matching candidates were returned.")

            else:
                query_candidates = sorted(
                    query_candidates,
                    key=safe_score,
                    reverse=True,
                )

                query_df = prepare_candidate_dataframe(
                    query_candidates
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Candidates Found",
                        len(query_df),
                    )

                with col2:
                    st.metric(
                        "Average Score",
                        f"{query_df['Score'].mean():.1f}%",
                    )

                with col3:
                    st.metric(
                        "Highest Score",
                        f"{query_df['Score'].max():.1f}%",
                    )

                st.caption(
                    "Scores estimate relevance to the query; "
                    "they do not verify candidate qualifications."
                )

                st.plotly_chart(
                    make_candidate_chart(
                        query_df,
                        "Candidate Query Relevance",
                    ),
                    use_container_width=True,
                )

                show_candidate_cards(
                    query_candidates,
                    "Candidate Details",
                )

        else:
            st.markdown("### 💡 Result")
            st.json(result)


# ============================================================
# 8. JOB DESCRIPTION MATCHING
# ============================================================

with right_col:
    st.markdown("## 📋 Job Description Matching")

    st.caption(
        "Compare available resumes against the requirements "
        "of your selected job."
    )

    with st.form("job_description_form"):
        job_description = st.text_area(
            "Job Description",
            placeholder=(
                "Paste a job description here...\n\n"
                "Example: Looking for a Data Scientist with "
                "Python, SQL, machine learning, Pandas, "
                "Generative AI, LLMs and RAG experience."
            ),
            height=300,
        )

        find_button = st.form_submit_button(
            "🔍 Find Candidates",
            use_container_width=True,
        )

    if find_button:
        if not job_description.strip():
            st.warning(
                "Please enter a job description before screening."
            )

        else:
            with st.spinner(
                "Evaluating resumes against the job description..."
            ):
                try:
                    requirements, candidate_results = (
                        screen_candidates(
                            job_description.strip()
                        )
                    )

                    ranked_candidates = rank_candidates(
                        candidate_results
                    )

                    st.session_state["ranked_candidates"] = (
                        ranked_candidates
                    )

                    st.session_state["requirements"] = requirements

                    st.session_state["job_description"] = (
                        job_description.strip()
                    )

                except Exception as error:
                    st.error(
                        f"Could not screen candidates: {error}"
                    )


# ============================================================
# 9. SCREENING RESULTS
# ============================================================

ranked_candidates = st.session_state.get(
    "ranked_candidates",
    [],
)

if ranked_candidates:
    st.divider()

    st.markdown("## 🏆 Candidate Screening Results")

    st.caption(
        "Candidates are ranked by estimated relevance "
        "to the job description."
    )

    candidate_df = prepare_candidate_dataframe(
        ranked_candidates
    )

    total_candidates = len(candidate_df)

    scores = candidate_df["Score"].tolist()

    average_score = (
        sum(scores) / len(scores)
        if scores
        else 0
    )

    highest_score = max(scores, default=0)

    match_counts = calculate_match_counts(
        ranked_candidates
    )

    strong_matches = match_counts["strong_matches"]
    potential_matches = match_counts["potential_matches"]
    low_matches = match_counts["low_matches"]

    # ========================================================
    # 10. KPI CARDS
    # ========================================================

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        st.metric(
            "👥 Resumes Screened",
            total_candidates,
        )

    with kpi2:
        st.metric(
            "📊 Average Score",
            f"{average_score:.1f}%",
        )

    with kpi3:
        st.metric(
            "🏆 Highest Score",
            f"{highest_score:.1f}%",
        )

    with kpi4:
        st.metric(
            "🟢 Score ≥ 80%",
            strong_matches,
        )

    with kpi5:
        st.metric(
            "🔴 Score < 60%",
            low_matches,
        )

    st.caption(
        "Strong score: 80% or above. Potential score: "
        "60–79.9%. Low score: below 60%. These are estimated "
        "relevance categories, not verified hiring decisions."
    )

    # ========================================================
    # 11. MATCH SUMMARY
    # ========================================================

    st.markdown("### 📌 Match Summary")

    summary1, summary2, summary3 = st.columns(3)

    with summary1:
        st.metric(
            "🟢 Strong Score Matches",
            strong_matches,
        )

    with summary2:
        st.metric(
            "🟡 Potential Score Matches",
            potential_matches,
        )

    with summary3:
        st.metric(
            "🔴 Low Score Matches",
            low_matches,
        )

    # ========================================================
    # 12. CANDIDATE SCORE CHART
    # ========================================================

    st.divider()

    st.markdown("### 📊 Candidate Match Visualization")

    st.caption(
        "Green = 80%+ · Yellow = 60–79.9% · Red = below 60%."
    )

    st.plotly_chart(
        make_candidate_chart(
            candidate_df,
            "Estimated Candidate Match Scores",
        ),
        use_container_width=True,
    )

    # ========================================================
    # 13. SCORE DISTRIBUTION
    # ========================================================

    st.divider()

    analytics1, analytics2 = st.columns(2, gap="large")

    with analytics1:
        st.markdown("### 📈 Score Distribution")

        fig_distribution = px.histogram(
            candidate_df,
            x="Score",
            nbins=10,
            labels={
                "Score": "Estimated Match Score (%)",
                "count": "Number of Candidates",
            },
        )

        fig_distribution.update_layout(
            height=350,
            xaxis_title="Estimated Match Score (%)",
            xaxis_range=[0, 100],
            yaxis_title="Number of Candidates",
            plot_bgcolor=COLORS["panel"],
            paper_bgcolor=COLORS["panel"],
            font_color=COLORS["text"],
            margin=dict(l=10, r=20, t=25, b=45),
        )

        st.plotly_chart(
            fig_distribution,
            use_container_width=True,
        )

    # ========================================================
    # 14. SCORE CATEGORY DISTRIBUTION
    # ========================================================

    with analytics2:
        st.markdown("### 🎯 Score Category Distribution")

        category_order = [
            "Strong score (80%+)",
            "Potential score (60–79%)",
            "Low score (<60%)",
        ]

        category_counts = (
            candidate_df["Category"]
            .value_counts()
            .reindex(category_order, fill_value=0)
        )

        category_df = pd.DataFrame({
            "Category": category_counts.index,
            "Candidates": category_counts.values,
        })

        fig_category = go.Figure(
            data=[
                go.Pie(
                    labels=category_df["Category"],
                    values=category_df["Candidates"],
                    hole=0.58,
                    marker=dict(
                        colors=[
                            COLORS["green"],
                            COLORS["yellow"],
                            COLORS["red"],
                        ]
                    ),
                    textinfo="label+value",
                    hovertemplate=(
                        "<b>%{label}</b>"
                        "<br>Candidates: %{value}"
                        "<extra></extra>"
                    ),
                )
            ]
        )

        fig_category.update_layout(
            height=350,
            paper_bgcolor=COLORS["panel"],
            font_color=COLORS["text"],
            margin=dict(l=10, r=10, t=25, b=20),
            showlegend=False,
        )

        st.plotly_chart(
            fig_category,
            use_container_width=True,
        )

    # ========================================================
    # 15. TOP CANDIDATE
    # ========================================================

    st.divider()

    st.markdown("### 🥇 Highest-Scoring Candidate")

    top_candidate = ranked_candidates[0]

    top_candidate_name = get_candidate_name(
        top_candidate
    )

    top_candidate_score = safe_score(
        top_candidate
    )

    top_category = score_category(
        top_candidate_score
    )

    top_col1, top_col2, top_col3 = st.columns([3, 1, 1])

    with top_col1:
        st.markdown(f"## {top_candidate_name}")

        st.caption(
            "This candidate has the highest estimated score "
            "among the screened resumes."
        )

    with top_col2:
        st.metric(
            "Estimated Score",
            f"{top_candidate_score:.1f}%",
        )

    with top_col3:
        st.metric(
            "Score Category",
            top_category,
        )

    st.markdown("**Evaluation explanation**")

    st.write(
        top_candidate.get(
            "reason",
            "No evaluation explanation was returned.",
        )
    )

    # ========================================================
    # 16. CANDIDATE DETAILS
    # ========================================================

    st.divider()

    show_candidate_cards(
        ranked_candidates,
        "📇 Candidate Contact Information and Evaluation",
    )

    # ========================================================
    # 17. EXTRACTED REQUIREMENTS
    # ========================================================

    requirements = st.session_state.get(
        "requirements"
    )

    if requirements:
        with st.expander(
            "📋 View Extracted Job Requirements"
        ):
            st.write(requirements)

    # ========================================================
    # 18. DOWNLOAD RESULTS
    # ========================================================

    st.divider()

    st.markdown("### 📥 Export Screening Results")

    export_df = candidate_df[
        [
            "Candidate",
            "Score",
            "Category",
            "Reason",
            "Email",
            "Phone",
            "LinkedIn",
            "GitHub",
        ]
    ].copy()

    export_df["Score"] = export_df["Score"].round(1)

    csv_data = export_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Download Candidate Results (CSV)",
        data=csv_data,
        file_name="resumind_ai_screening_results.csv",
        mime="text/csv",
    )

elif "ranked_candidates" in st.session_state:
    st.info(
        "No candidate results were returned for this job description."
    )


# ============================================================
# 19. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        <strong>🧠 Resumind AI</strong>
        <br>
        Smarter Resumes. Better Matches.
        <br>
        AI-Powered Resume Screening & Candidate Intelligence
        <br><br>
        Built with Python · Streamlit · RAG · Large Language Models
    </div>
    """,
    unsafe_allow_html=True,
)