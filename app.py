import streamlit as st
import atexit
import ui as ui
import os
from agent import ResumeAnalysisAgent


# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SkillSync AI | Recruitment Intelligence",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------------------------------------------------------
# Role Requirements Dictionary
# -----------------------------------------------------------------------------
ROLE_REQUIREMENTS = {
    "AI/ML Engineer": [
        "Python", "Pytorch", "Tensorflow", "Machine Learning", "Deep Learning",
        "MLOps", "Scikit-Learn", "NLP", "Computer Vision", "Reinforcement Learning",
        "Hugging Face", "Data Engineering", "Feature Engineering", "AutoML"
    ],
    "Frontend Engineer": [
        "React", "Vue", "Angular", "HTML5", "CSS3", "JavaScript", "TypeScript",
        "Next.js", "Svelte", "Bootstrap", "Tailwind CSS", "GraphQL", "Redux",
        "WebAssembly", "Three.js", "Performance Optimization"
    ],
    "Backend Engineer": [
        "Python", "Java", "Node.js", "REST APIs", "Cloud services", "Kubernetes",
        "Docker", "GraphQL", "Microservices", "gRPC", "Spring Boot", "Flask",
        "FastAPI", "SQL & NoSQL Database", "Redis", "RabbitMQ", "CI/CD"
    ],
    "Data Engineer": [
        "Python", "SQL", "Apache Spark", "Hadoop", "Kafka", "ETL pipelines",
        "Airflow", "BigQuery", "Redshift", "Data Warehousing", "Snowflake",
        "Azure Data Factory", "GCP", "AWS Glue", "DBT"
    ],
    "DevOps Engineer": [
        "Kubernetes", "Docker", "Terraform", "CI/CD", "AWS", "Azure", "GCP",
        "Jenkins", "Ansible", "Prometheus", "Grafana", "Helm",
        "Linux Administration", "Networking", "Site Reliability Engineering (SRE)"
    ],
    "Product Manager": [
        "Product Strategy", "User Research", "Agile Methodologies", "Roadmapping",
        "Market Analysis", "Stakeholder Management", "Data Analysis",
        "User Stories", "Product Lifecycle", "A/B Testing", "KPI Definition",
        "Prioritization", "Competitive Analysis", "Customer Journey Mapping"
    ],
    "Data Scientist": [
        "Python", "R", "SQL", "Machine Learning", "Statistics",
        "Data Visualization", "Pandas", "Numpy", "Scikit-learn", "Jupyter",
        "Hypothesis Testing", "Experimental Design", "Feature Engineering",
        "Model Evaluation"
    ]
}


# -----------------------------------------------------------------------------
# Initialize Streamlit Session State
# -----------------------------------------------------------------------------
if 'resume_agent' not in st.session_state: st.session_state.resume_agent = None
if 'resume_analyzed' not in st.session_state: st.session_state.resume_analyzed = None
if 'analysis_result' not in st.session_state: st.session_state.analysis_result = None


# ------------------------------------------------------------------------------
# Agent Setup
# ------------------------------------------------------------------------------
def setup_agent(config):
    """
    Initialize or update the ResumeAnalysisAgent based on sidebar config.
    """
    provider = config.get('provider')
    # This must match the key name in ui.py (which is 'openai_api_key')
    api_key = config.get('openai_api_key', "").strip() 
    model_name = config.get('model_name')

    if not api_key:
        return None

    # Re-initialize if agent is None OR if settings changed
    if st.session_state.resume_agent is None:
        st.session_state.resume_agent = ResumeAnalysisAgent(
            provider=provider, 
            api_key=api_key, 
            model_name=model_name
        )
    else:
        # Check if config has changed to avoid unnecessary re-initialization
        agent = st.session_state.resume_agent
        if (agent.provider != provider or 
            agent.model_name != model_name or 
            agent.api_key != api_key):
            
            st.session_state.resume_agent = ResumeAnalysisAgent(
                provider=provider, 
                api_key=api_key, 
                model_name=model_name
            )
            
    return st.session_state.resume_agent



# --------------------------
# Resume analysis functions
# --------------------------
def analyze_resume(agent, resume_file, role, custom_jd):
    """
    Analyze a resume using the agent with either role requirements or custom JD.
    Args:
        agent (ResumeAnalysisAgent): Initialized agent.
        resume_file: Uploaded resume file.
        role (str): Role to match against ROLE_REQUIREMENTS.
        custom_jd (str or None): Optional custom job description.
    Returns:
        dict or None: Analysis result
    """
    if not resume_file:
        st.error("Please upload a resume.")
        return None

    try:
        with st.spinner("Analyzing resume.... This may take a minute."):
            if custom_jd:
                result = agent.analyze_resume(resume_file, custom_jd=custom_jd)
            else:
                result = agent.analyze_resume(resume_file, role_requirements=ROLE_REQUIREMENTS[role])

            st.session_state.resume_analyzed = True
            st.session_state.analysis_result = result
            return result
    except Exception as e:
        st.error(f"Error analyzing resume: {e}")
        return None


def ask_question(agent, question):
    """
    Ask question about the analyzed resume.
    Args:
        agent (ResumeAnalysisAgent)
        question (str)
    Returns:
        str: Response
    """
    try:
        with st.spinner("Generating response..."):
            response = agent.ask_question(question)
            return response
    except Exception as e:
        return f"Error: {e}"


def generate_interview_questions(agent, question_types, difficulty, num_questions):
    """
    Generate interview questions based on the resume.
    Args:
        agent (ResumeAnalysisAgent)
        question_types (list)
        difficulty (str)
        num_questions (int)
    Returns:
        list: Generated questions
    """
    try:
        with st.spinner("Generating personalized interview questions...."):
            return agent.generate_interview_questions(question_types, difficulty, num_questions)
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return []


def improve_resume(agent, improvement_areas, target_role):
    """ 
    Suggest improvements for the resume.
    Args:
        agent (ResemeAnalysisAgent)
        improvements_areas (lits)
        target_role (str) 
    Returns:
        dict: Suggestion for improvement
    """
    try:
        with st.spinner("Analyzing and generating improvements...."):
            return agent.improve_resume(improvement_areas, target_role)
    except Exception as e:
        st.error(f"Error generating improvements: {e}")
        return {}


def get_improved_resume(agent, target_role, highlight_skills):
    """
    Generate an improved version of the resume with highlighted skills.
    Args:
        agent (ResumeAnalysisAgent)
        target_role (str)
        highlight_skills (list)
    Returns:
        str: Updated resume content or error message
    """
    try:
        with st.spinner("Creating improved resume..."):
            return agent.get_improved_resume(target_role, highlight_skills)
    except Exception as e:
        st.error(f"Error creating improved resume: {e}")
        return "Error generating improved resume."


# --------------------------
# Cleanup Function
# --------------------------
def cleanup():
    """ Clean up resources on app exit safely """
    try:
        if "resume_agent" in st.session_state and st.session_state.resume_agent is not None:
            st.session_state.resume_agent.cleanup()
    except (AttributeError, RuntimeError):
        pass

atexit.register(cleanup)


# --------------------------
# Main Application
# --------------------------
def main():
    """
    Main Streamlit application function. Handles UI rendering, tabs, and 
    user interactions for resume analysis, Q&A, interview question generation, 
    and resume improvements.
    """
    # UI Initialization
    ui.setup_page()
    ui.display_header()
    config = ui.setup_sidebar()

    #  Agent Initialization
    agent = setup_agent(config)

    # Tab Setup
    tabs = st.tabs([
        "Resume Analysis",
        "Resume Q&A",
        "Interview Questions",
        "Resume Improvements",
        "Improved Resume", 
        "Batch Ranking" # New Tab
    ])

    # --------------------------
    # Tab 1: Resume Analysis
    # --------------------------
    with tabs[0]:
        # FIX: Capture role and save it to session state immediately
        role, custom_jd = ui.role_selection_section(ROLE_REQUIREMENTS)
        st.session_state.selected_role = role # <--- ADD THIS LINE
        
        uploaded_resume = ui.resume_upload_section()
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Analyze Resume", type='primary'):
                if agent and uploaded_resume:
                    analyze_resume(agent, uploaded_resume, role, custom_jd)

        if st.session_state.analysis_result:
            ui.display_analysis_result(st.session_state.analysis_result)
    # --------------------------
    # Tab 2: Resume Q&A
    # --------------------------
    with tabs[1]:
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.resume_qa_section(
                has_resume=True,
                ask_question_func=lambda q: ask_question(st.session_state.resume_agent, q)
            )
        else:
            st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")

    # --------------------------
    # Tab 3: Interview Questions
    # --------------------------
    with tabs[2]:
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.interview_question_section(
                has_resume=True,
                generate_question_func=lambda types, diff, num:
                generate_interview_questions(st.session_state.resume_agent, types, diff, num)
            )
        else:
            st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")

    # --------------------------
    # Tab 4: Resume Improvements
    # --------------------------
    with tabs[3]:
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.resume_improvement_section(
                has_resume=True,
                improve_resume_func=lambda areas, role: improve_resume(st.session_state.resume_agent, areas, role)
            )
        else:
            st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")

    # --------------------------
    # Tab 5: Improved Resume
    # --------------------------
    with tabs[4]:
        if st.session_state.resume_analyzed and st.session_state.resume_agent:
            ui.improved_resume_section(
                has_resume=True,
                get_improved_resume_func=lambda role, skills:
                get_improved_resume(st.session_state.resume_agent, role, skills)
            )
        else:
            st.warning("Please upload and analyze a resume first in the 'Resume Analysis' tab.")

        
    # --------------------------
    # Tab 6: Batch Ranking
    # --------------------------
    with tabs[5]:
        if st.session_state.resume_agent:
            # Use the role selected in Tab 1, or default to the first one
            current_role = st.session_state.get('selected_role', list(ROLE_REQUIREMENTS.keys())[0])
            
            # Pass the rank function to the UI
            ui.batch_ranking_section(
                has_agent=True,
                rank_func=lambda files: st.session_state.resume_agent.rank_multiple_resumes(
                    files, 
                    role_requirements=ROLE_REQUIREMENTS[current_role]
                )
            )
        else:
            st.warning("Please provide an API Key in the sidebar first.")

if __name__ == "__main__":
    main()
