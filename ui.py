import base64
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


def setup_page():
    """Apply custom css and setup page (without setting page config)"""
    apply_custom_css()


def display_header():
    # SkillSync AI - Professional IT branding
    st.markdown(
        """
        <div style="text-align:center; padding: 1.5rem 0; border-bottom: 1px solid #334155; margin-bottom: 2rem;">
            <h1 style="font-weight: 800; letter-spacing: -1px; color: #ffffff; margin:0;">SkillSync <span style="color: #6366f1;">AI</span></h1>
            <p style="color: #9AA0A6; font-size: 1.1rem; font-weight: 400;">Next-Gen Recruitment Intelligence System</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def apply_custom_css():
    st.markdown(
        """
        <style>
        /* Main Container */
        .main, .stApp {
            background-color: #0f172a !important;
            color: #f8fafc !important;
        }

        /* Tabs spacing */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }

        /* Default tab style */
        .stTabs [data-baseweb="tab"] {
            height: 45px;
            background-color: transparent !important;
            border-radius: 4px 4px 0 0;
            color: #94a3b8 !important;
            font-weight: 600;
        }

        /* Active tab */
        .stTabs [aria-selected="true"] {
            color: #6366f1 !important;
            border-bottom: 2px solid #6366f1 !important;
            background-color: transparent !important;
        }

        /* Buttons */
        .stButton button {
            background-color: #6366f1 !important;
            color: white !important;
            border-radius: 6px !important;
            border: none !important;
            font-weight: 600 !important;
            padding: 0.6rem 2rem !important;
        }

        .stButton button:hover {
            filter: brightness(90%);
        }

        /* Warning message */
        div.stAlert {
            background-color: #0000 !important;
            color: white !important;
        }

        /* Input fields */
        .stTextInput input,
        .stTextArea textarea,
        .stSelectbox div {
        
        }

        /* Horizontal rule */
        hr {
            border: none;
            height: 2px;
            background-color: #334155;
        }

        /* Markdown text */
        .stMarkdown, .stMarkdown p {
            color: #f8fafc !important;
        }

        /* Skill tags */
        .skills-tag {
            display: inline-block;
            background-color: #312e81;
            color: #e0e7ff;
            padding: 5px 12px;
            border-radius: 15px;
            margin: 5px;
            font-weight: bold;
            border: 1px solid #4338ca;
        }

        .skill-tag.missing {
            background-color: #334155;
            color: #94a3b8;
            border: 1px solid #475569;
        }

        /* Strength & improvements layout */
        .strength-improvements {
            display: flex;
            gap: 20px;
        }

        .strength-improvements > div {
            flex: 1;
        }

        /* Card styling */
        .card {
            background-color: #1e293b;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            border: 1px solid #334155;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        }

        /* Improvement suggestions */
        .improvement-uten {
            background-color: #0f172a;
            padding: 15px;
            margin: 10px 0;
            border-radius: 6px;
            border: 1px solid #334155;
        }

        /* Comparison layout */
        .compariosn-container {
            display: flex;
            gap: 20px;
            margin-top: 15px;
        }

        .compariosion-box {
            flex: 1;
            background-color: #0f172a;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #334155;
        }

        /* Weakness details */
        .weakness-detail {
            background-color: #312e81;
            padding: 10px 15px;
            margin: 5px 0;
            border-radius: 6px;
            border-left: 3px solid #6366f1;
        }

        /* Solution details */
        .solution-detaiol {
            background-color: #1e293b;
            padding: 10px 15px;
            margin: 5px 0;
            border-radius: 6px;
            border-left: 3px solid #6366f1;
        }

        /* Example details */
        .example-datils {
            background-color: #0f172a;
            padding: 10px 15px;
            margin: 5px 0;
            border-radius: 6px;
            border-left: 3px solid #6366f1;
        }

        /* Download button */
        .download-btn {
            display: inline-block;
            background-color: #1e293b;
            color: #f8fafc;
            padding: 10px 20px;
            border-radius: 8px;
            text-decoration: none;
            border: 1px solid #6366f1;
            text-align: center;
            width: 100%;
            font-weight: 600;
        }

        .download-btn:hover {
            background-color: #6366f1;
            color: white;
        }

        /* Pie chart container */
        .pie-chat-container {
            padding: 10px;
            background-color: #1e293b;
            border-radius: 10px;
            margin-bottom: 15px;
            border: 1px solid #334155;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
 


def setup_sidebar():
    with st.sidebar:
        st.markdown("<h2 style='margin:0;'>SkillSync <span style='color:#6366f1;'>AI</span></h2>", unsafe_allow_html=True)
        st.caption("v1.4.0 | Multi-Model Intelligence")
        st.markdown("---")
        
        st.subheader("Model Configuration")
        
        # 1. Select Provider 
        provider = st.selectbox(
            "Select AI Provider", 
            ["OpenAI", "Google Gemini", "Anthropic", "Groq (Llama/Mistral)"]
        )
        
        # 2. Dynamic Input & Model Selection 
        api_key = ""
        model_name = ""
        
        if provider == "OpenAI":
            api_key = st.text_input("OpenAI API Key", type="password", help="Get it from platform.openai.com")
            # 2nd Option: gpt-4-turbo (High reasoning) or gpt-3.5-turbo (Cheap/Fast)
            model_name = st.selectbox("Select Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
            
        elif provider == "Google Gemini":
            api_key = st.text_input("Google API Key", type="password", help="Get it from aistudio.google.com")
            # 2nd Option: gemini-1.5-flash (Extremely fast and free tier available)
            model_name = st.selectbox("Select Model", ["gemini-1.5-pro", "gemini-1.5-flash"])
            
        elif provider == "Anthropic":
            api_key = st.text_input("Anthropic API Key", type="password", help="Get it from console.anthropic.com")
            # 2nd Option: claude-3-haiku (Fastest and cheapest Claude model)
            model_name = st.selectbox("Select Model", ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"])
            
        elif provider == "Groq (Llama/Mistral)":
            api_key = st.text_input("Groq API Key", type="password", help="Get it for FREE from console.groq.com")
            # 2nd Option: llama3-8b (The fastest model in the world right now)
            model_name = st.selectbox("Select Model", ["llama-3.1-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"])

        # --- Status Messages ---
        if api_key:
            st.success(f"{provider} Connected!")
        else:
            st.warning(f"Please enter your {provider} API Key.")

        st.markdown("---")        
        st.info(f"System optimized for {model_name} & Semantic Benchmarking.")
        with st.expander("How to use?"):
            st.write(f"1. Get your {provider} API Key")
            st.write("2. Upload Resume (PDF)")
            st.write("3. Select Target Role")
            st.write("4. Click Analyze")
            
    # Return everything for the agent setup
    return {
        "provider": provider, 
        "openai_api_key": api_key, 
        "model_name": model_name
    }




def role_selection_section(role_requirements):
    st.markdown(
        """
        <div class="card">
            <h3>Select the role you're applying for:</h3>
            <p style="color: #9AA0A6; font-size: 0.9rem;">Choose from industry standard roles or provide your own JD</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. The Interaction Area (Dropdown and Checkbox)
    col1, col2 = st.columns([2, 1])

    with col1:
        role = st.selectbox(
            "Select the role you're applying for:",
            list(role_requirements.keys()),
            label_visibility="collapsed" 
        )

    with col2:
        # Checkbox logic 
        upload_jd = st.checkbox("Upload Custom Job Description")

    custom_jd = None

    # 3. Custom JD Upload Section
    if upload_jd:
        st.markdown(
            """
            <div class="card">
                <h3 style="margin-bottom: 0.5rem;">Upload Custom Job Description</h3>
                <p style="color: #9AA0A6; font-size: 0.9rem;">Supported formats: PDF, TXT</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        custom_jd_file = st.file_uploader(
            "Upload Job Description", 
            type=["pdf", "txt"],
            label_visibility="collapsed" 
        )

        if custom_jd_file:
            st.success(f"Job Description '{custom_jd_file.name}' uploaded successfully!")
            custom_jd = custom_jd_file

    # 4. Required Skills Section (Only if not uploading custom JD)
    if not upload_jd:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.info(f"**Required Skills:** {', '.join(role_requirements[role])}")
        st.markdown(
            f"<p style='margin-bottom:0;'>Selection Cutoff: <b>75/100</b></p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)        
    return role, custom_jd


def resume_upload_section():
    st.markdown(
        """
        <div class="card">
            <h3>Upload your resume</h3>
            <p>Supported format: PDF</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_resume = st.file_uploader(
        "Upload Resume",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_resume is not None:
        st.success(f"Resume '{uploaded_resume.name}' uploaded successfully!")

    return uploaded_resume


def create_score_pie_chart(score):
    """Create a professional pie chart for the score visualization"""
    """Professional Donut Chart matching the Indigo/Slate Theme"""
    
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="#1e293b")

    sizes = [score, 100 - score]
    colors = ["#6366f1", "#0f172a"] 
    explode = (0.05, 0)

    ax.pie(
        sizes,
        colors=colors,
        explode=explode,
        startangle=90,
        wedgeprops={"width": 0.4, "edgecolor": "#1e293b", "linewidth": 2},
    )

    # score text in the center
    ax.text(0, 0, f"{score}%", ha="center", va="center", fontsize=24, fontweight="bold", color="white")

    # Pass/Fail indicator colors matching Indigo Theme
    status = "PASS" if score >= 75 else "FAIL"
    status_color = "#10b981" if score >= 75 else "#ef4444" 

    ax.text(0, -0.20, status, ha="center", va="center", fontsize=14, fontweight="bold", color=status_color)
    ax.set_facecolor("#1e293b")
    ax.set_aspect("equal")
    return fig


def display_analysis_result(analysis_result):
    if not analysis_result:
        return

    oberall_score = analysis_result.get('overall_score', 0)
    selected = analysis_result.get("selected", False)
    skill_scores = analysis_result.get("skill_scores", {}) 
    detailed_weaknesses = analysis_result.get("detailed_weaknesses", [])

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(
        '<div style="text-align:right; font-size:0.8rem; color:#888; margin-bottom:10px;">'
        'Powered by SkillSync AI</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Overall Score", f"{oberall_score}/100")
        fig = create_score_pie_chart(oberall_score)
        st.pyplot(fig)

    with col2:
        if selected:
            st.markdown(
                "<h2 style='color:#4CAF50;'>Congratulations! You have been selected.</h2>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<h2 style='color:#333333;'>Unfortunately! You were not selected.</h2>",
                unsafe_allow_html=True
            )
        st.write(analysis_result.get('reasoning', ''))

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="strength-improvements">', unsafe_allow_html=True)

    # Strengths
    st.subheader("Strengths")
    strengths = analysis_result.get("strengths", [])
    if strengths:
        for skill in strengths:
            st.markdown(
                f'<div class="skill-tag">{skill} ({skill_scores.get(skill, "N/A")}/10)</div>',
                unsafe_allow_html=True
            )
    else:
        st.write("No notable strengths identified.")

    # Weaknesses
    st.subheader("Area for Improvement")
    missing_skills = analysis_result.get("missing_skills", [])
    if missing_skills:
        for skill in missing_skills:
            st.markdown(
                f'<div class="skill-tag missing">{skill} ({skill_scores.get(skill, "N/A")}/10)</div>',
                unsafe_allow_html=True
            )
    else:
        st.write("No significant areas for improvement.")

    st.markdown('</div>', unsafe_allow_html=True)

    # weaknesses section
    if detailed_weaknesses:
        st.markdown('<hr>', unsafe_allow_html=True)
        st.subheader("Detailed Weakness Analysis")

        for weakness in detailed_weaknesses:
            skill_name = weakness.get('skill', '')
            score = weakness.get('score', 0)

            with st.expander(f"{skill_name} (score: {score}/10)"):
                # detail = weakness.get('details', 'No specific details provided.')
                detail = weakness.get('weakness', 'No specific details provided.')

                if detail.startswith('```json') or '{' in detail:
                    detail = "The resume lacks examples of this skill."

                st.markdown(
                    f'<div class="weakness-detail"><strong>Issue:</strong> {detail}</div>',
                    unsafe_allow_html=True
                )

                if 'suggestions' in weakness and weakness['suggestions']:
                    st.markdown("<strong>How to improve:</strong>", unsafe_allow_html=True)
                    for i, suggestion in enumerate(weakness['suggestions']):
                        st.markdown(
                            f'<div class="solution-detail">{i+1}. {suggestion}</div>',
                            unsafe_allow_html=True
                        )

                if 'example' in weakness and weakness['example']:
                    st.markdown("<strong>Example addition:</strong>", unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="example-detail">{weakness["example"]}</div>',
                        unsafe_allow_html=True
                    )

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        report_content = f"""
        # SkillSync AI - Resume Analysis Report
        ## Overall score: {oberall_score}/100
        Status: {"Shortlisted" if selected else "Not Selected"}

        ## Analysis Reasoning
        {analysis_result.get('reasoning', 'No reasoning provided.')}

        ## Strengths
        {", ".join(strengths if strengths else ["None identified"])}

        ## Detailed Weakness Analysis
        """

        for weakness in detailed_weaknesses:
            skill_name = weakness.get('skill', '')
            score = weakness.get('score', 0)
            # detail = weakness.get('details', 'No specific details provided.')
            detail = weakness.get('weakness', 'No specific details provided.')

            if detail.startswith('```json') or '{' in detail:
                detail = "The resume lacks examples of this skill."

            report_content += f"\n### {skill_name} (score: {score}/10)\n"
            report_content += f"Issue: {detail}\n"

            if 'suggestions' in weakness and weakness['suggestions']:
                report_content += "\nImprovement suggestions:\n"
                for sugg in weakness['suggestions']:
                    report_content += f"- {sugg}\n"

            if 'example' in weakness and weakness['example']:
                report_content += f"\nExample: {weakness['example']}\n"


        report = f"SkillSync AI Analysis\nScore: {oberall_score}%\nResult: {'Selected' if selected else 'Not Selected'}"
        b64 = base64.b64encode(report.encode()).decode()
        st.markdown(f'<a href="data:text/plain;base64,{b64}" download="skillsync_report.txt" class="download-btn">Download Report</a>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        report_content += "\n---\nAnalysis provided by SkillSync AI Agent"
    st.markdown('</div>', unsafe_allow_html=True)


def resume_qa_section(has_resume, ask_question_func=None):
    if not has_resume:
        st.warning("Please upload and analyze a resume first.")
        return 
        
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("Ask Question about the resume")
    user_question = st.text_input("Enter your question about the resume:", placeholder="What is the candidate's most recent experience?")

    if user_question and ask_question_func:
        with st.spinner("Searching resume and generating response..."):
            response = ask_question_func(user_question)

            st.markdown('<div style="background-color:#111122; padding:15px; border-radius:5px; border-left: 5px solid #d32f2f;">', unsafe_allow_html=True)
            st.write(response)
            st.markdown('</div>', unsafe_allow_html=True)

    # example questions 
    with st.expander("Example Questions"):
        example_questions = [
            "What is the candidate's most recent role?", 
            "How many years of experience does the candidate have with python?", 
            "What educational qualifications does the candidate have?", 
            "Has the candidate managed teams before?", 
            "What projects has the candidate worked on?", 
            "Does the candidate have experience with cloud technologies?", 
        ]
        for question in example_questions:
            if st.button(question, key=f"q_{question}"):
                st.session_state.current_question = question 
                st.rerun() 
    st.markdown('</div>', unsafe_allow_html=True)


def interview_question_section(has_resume, generate_question_func=None):
    if not has_resume:
        st.warning("Please upload and analyze a resume first.")
        return 
    
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        question_types = st.multiselect(
            "Select question types:", 
            ["Basic", "Technical", "Experience", "Scenario", "coding", "Behavioral"],
            default=["Basic", "Technical"]
        )

    with col2:
        difficulty = st.select_slider(
            "Question difficulty:", 
            options=["Easy", "Medium", "Hard"], 
            value="Medium"
        )

    num_questions = st.slider("Number of questions:", 3, 15, 5)

    if st.button("Generate Interview Questions"):
        if generate_question_func:
            with st.spinner("Generating Personalized interview questions..."):
                questions = generate_question_func(question_types, difficulty, num_questions)

                # content for download 
                download_content = "# SkillSync AI - Interview Questions\n\n"
                download_content += f"Difficulty: {difficulty}\n"
                download_content += f"Types: {', '.join(question_types)}\n\n"

                for i, (q_type, question) in enumerate(questions):
                    with st.expander(f"{q_type} Question {i+1}"):
                        st.write(question)

                        # for coding question  
                        if q_type == "coding":
                            st.code("# Write your solution here", language="python")

                    # download content 
                    download_content += f"## {i+1}. {q_type} Question\n"
                    download_content += f"{question}\n\n"
                    if q_type == "coding":
                        download_content += "```python\n# write your solution here\n```\n\n"

                # download content 
                download_content += "\n---\nQuestion generated by SkillSync AI Agent"

                # download button 
                if questions:
                    st.markdown("---")
                    question_bytes = download_content.encode()
                    b64 = base64.b64encode(question_bytes).decode()
                    href = f'<a class="download-btn" href="data:text/markdown;base64,{b64}" download="SkillSync_interview_questions.md">Download All Questions</a>'
                    st.markdown(href, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def resume_improvement_section(has_resume, improve_resume_func=None):
    if not has_resume:
        st.warning("Please upload and analyze a resume first.")
        return 
    
    st.markdown('<div class="card">', unsafe_allow_html=True)

    improvement_areas = st.multiselect(
        "Select area to improve:",
        ["Content", "Format", "Skills highlighting", "Experience Description", "Education", "Projects", "Achievements", "Overall Structure"],
        default=["Content", "Skills highlighting"]
    )

    target_role = st.text_input("Target role (optional):", placeholder="e.g., Senior Data Scientist at Google")

    if st.button("Generate Resume Improvements"):
        if improve_resume_func:
            with st.spinner("Analyzing and generating improvements..."):
                improvements = improve_resume_func(improvement_areas, target_role)

                # content for download 
                download_content = f"# SkillSync AI - Resume Improvement Suggestions\n\nTarget Role: {target_role if target_role else 'Not specified'}\n\n"

                for area, suggestions in improvements.items():
                    with st.expander(f"Improvements for {area}", expanded=True):
                        st.markdown(f"<p>{suggestions['description']}</p>", unsafe_allow_html=True)

                        st.subheader("Specific Suggestions")
                        for i, specific in enumerate(suggestions["specific"]):
                            st.markdown(f'<div class="solution-detail"><strong>{i+1}.</strong> {specific}</div>', unsafe_allow_html=True)

                        if "before_after" in suggestions:
                            st.markdown('<div class="comparison-container">', unsafe_allow_html=True)

                            st.markdown('<div class="comparison-box">', unsafe_allow_html=True)
                            st.markdown("<strong>Before:</strong>", unsafe_allow_html=True)
                            st.markdown(f"<pre>{suggestions['before_after']['before']}</pre>", unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                            st.markdown('<div class="comparison-box">', unsafe_allow_html=True)
                            st.markdown("<strong>After:</strong>", unsafe_allow_html=True)
                            st.markdown(f"<pre>{suggestions['before_after']['after']}</pre>", unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                    # download content
                    download_content += f"## Improvement for {area}\n\n"
                    download_content += f"{suggestions['description']}\n\n"
                    download_content += "### Specific Suggestions\n\n"
                    for i, specific in enumerate(suggestions["specific"]):
                        download_content += f"{i+1}. {specific}\n"
                    if "before_after" in suggestions:
                        download_content += "### Before\n\n"
                        download_content += f"```\n{suggestions['before_after']['before']}\n```\n\n"
                        download_content += "### After\n\n"
                        download_content += f"```\n{suggestions['before_after']['after']}\n```\n\n"

                # download content
                download_content += "\n---\nProvided by SkillSync AI Agent"

                # Add download button
                st.markdown("---")
                report_bytes = download_content.encode()
                b64 = base64.b64encode(report_bytes).decode()
                href = f'<a class="download-btn" href="data:text/markdown;base64,{b64}" download="SkillSync_resume_improvement.md">Download All Suggestions</a>'
                st.markdown(href, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def improved_resume_section(has_resume, get_improved_resume_func=None):
    if not has_resume:
        st.warning("Please upload and analyze a resume first")
        return 
    
    st.markdown('<div class="card">', unsafe_allow_html=True)

    target_role = st.text_input(
        "Target role:",
        placeholder="e.g., Senior Software Engineer"
    )

    highlight_skills = st.text_area(
        "Paste your JD to get updated resume",
        placeholder="e.g., Python, React, Cloud Architecture"
    )

    if st.button("Generate Improved Resume"):
        if get_improved_resume_func:
            with st.spinner("Creating improved resume..."):
                improved_resume = get_improved_resume_func(
                    target_role,
                    highlight_skills
                )

                st.subheader("Improved Resume")
                st.text_area("Improved Resume", improved_resume, height=400)

                # Download buttons
                col1, col2 = st.columns(2)

                with col1:
                    # TXT file download
                    resume_bytes = improved_resume.encode()
                    b64 = base64.b64encode(resume_bytes).decode()
                    href = (
                        f'<a class="download-btn" '
                        f'href="data:text/plain;base64,{b64}" '
                        f'download="SkillSync_improved_resume.txt">'
                        f'Download as TXT</a>'
                    )
                    st.markdown(href, unsafe_allow_html=True)

                with col2:
                    # Markdown file download
                    md_content = f"""# {target_role if target_role else "Professional"} Resume
{improved_resume}

---
Resume enhanced by SkillSync AI Agent
"""
                    md_bytes = md_content.encode()
                    md_b64 = base64.b64encode(md_bytes).decode()
                    md_href = (
                        f'<a class="download-btn" '
                        f'href="data:text/markdown;base64,{md_b64}" '
                        f'download="SkillSync_improved_resume.md">'
                        f'Download as Markdown</a>'
                    )
                    st.markdown(md_href, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def create_tabs():
    return st.tabs([
        "Resume Analysis",
        "Resume Q&A",
        "Interview Questions",
        "Resume Improvement",
        "Improved Resume"
    ])


def batch_ranking_section(has_agent, rank_func=None, role_requirements=None):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Batch Resume Ranking (Top 5)")
    st.info("Upload up to 5 resumes to compare them against the selected role requirements.")

    uploaded_files = st.file_uploader(
        "Upload up to 5 Resumes", 
        type=["pdf"], 
        accept_multiple_files=True,
        key="batch_uploader"
    )

    if uploaded_files:
        if len(uploaded_files) > 5:
            st.error("Please upload a maximum of 5 resumes.")
            return

        if st.button("Rank Resumes", type="primary"):
            if not rank_func:
                st.error("Analysis function not found.")
                return

            with st.spinner(f"Analyzing {len(uploaded_files)} resumes..."):
                results = rank_func(uploaded_files)
                
                if results:
                    st.markdown("### 🏆 Candidate Leaderboard")
                    df = pd.DataFrame(results)
                    df.index = df.index + 1 
                    st.table(df)

                    for i, res in enumerate(results):
                        color = "#10b981" if res['score'] >= 75 else "#94a3b8"
                        st.markdown(f"""
                        <div style="padding:10px; border-radius:8px; background-color:#1e293b; border-left: 5px solid {color}; margin-bottom:10px;">
                            <span style="font-weight:bold; color:{color}">#{i+1}</span> | 
                            <strong>{res['candidate_name']}</strong> - Score: {res['score']}% 
                            <br><small>Strengths: {res['top_strengths']}</small>
                        </div>
                        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)