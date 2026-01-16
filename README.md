# 🤖 SkillSync AI: Next-Gen Recruitment Intelligence System

SkillSync AI is a professional-grade recruitment assistant that leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to automate resume screening, technical interview preparation, and ATS optimization.   
Designed with a modern "Indigo & Slate" minimal interface, this tool provides data-driven insights for high-end IT recruitment environments.

## 🚀 Key Features
- **Semantic Tech-Audit:** Scores resumes (0-100) against industry-standard IT roles or custom Job Descriptions (JDs) using semantic vector similarity.
- **Contextual RAG Engine:** Ask specific questions about a candidate's history, such as "Does the applicant have production experience with AWS?" or "Extract their leadership achievements."
- **AI Interview Architect:** Generates personalized Technical, Scenario-based, and Behavioral questions based specifically on the candidate’s unique background.
- **ATS Optimization Engine:** Identifies technical gaps and provides "Before & After" examples to enhance resume formatting and content for Applicant Tracking Systems.
- **Dynamic Improvement Engine:** Provides actionable suggestions to fix resume weaknesses identified during the audit.
- **Enterprise UI:** A minimal, professional dashboard designed for high-end IT recruitment environments.

## 🛠️ Tech Stack
- **Frontend:** Streamlit (Custom Indigo/Slate Theme)
- **Orchestration:** LangChain 0.3 (RetrievalQA, Vector Stores, Text Splitters)
- **LLM:** OOpenAI GPT-4o (Reasoning & Generation)
- **Embeddings:** HuggingFace all-MiniLM-L6-v2 (Free & Local Processing)
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **PDF Processing:** PyPDF2 (PDF) and Python IO (TXT)

## 🏗️ Architecture Workflow
1. **Ingestion:** Extracts raw text from PDF/TXT resumes using specialized parsers.
2. **Vectorization:** Splits content into semantic chunks. These are converted into high-dimensional vectors via Local HuggingFace Embeddings (saving API costs).
3. **Storage:** Chunks are indexed in a local FAISS vector store for sub-second retrieval.
4. **Intelligence Layer:**
    - **Audit Mode:** Performs semantic scoring across specific technical competencies.
    - **Q&A Mode:** Uses a RetrievalQA chain to provide grounded answers based only on the uploaded document.

## 🚀 Deployment & CI/CD Pipeline
This project is containerized with Docker and deployed on AWS using a fully automated CI/CD pipeline via GitHub Actions.
- **Containerization:** Docker (Multi-stage builds)
- **CI/CD:** GitHub Actions (Automated testing and deployment)
- **Cloud:** Optimized for deployment on AWS (EC2/ECS) or Streamlit Cloud.


## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.11+
- An OpenAI API Key

### 2. Clone the Repository
    git clone https://github.com/yourusername/skillsync-ai.git
    cd skillsync-ai

### 3. Setup Virtual Environment
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate

### 4. Install Dependencies
    pip install streamlit langchain langchain-openai faiss-cpu PyPDF2 matplotlib pandas langchain-huggingface sentence-transformers

### 5. Run the Application
    streamlit run app.py


## 📸 Usage Guide
- **Configure:** Enter your OpenAI API key in the sidebar.
- **Upload:** Drop a PDF resume into the "Resume Analysis" tab.
- **Target:** Select a predefined technical role (e.g., AI Engineer, DevOps) or upload a custom JD.
- **Audit:** Click "Analyze Resume" to generate the competency report.
- **Optimize:** Use the "Interview Questions" and "Resume Improvements" tabs to prepare for the hiring process or refine the candidate's profile.

## 📜 License
- This project is licensed under the MIT License.

## [IMPORTANT]
- Disclaimer : SkillSync AI is an assistant tool. Automated results should always be validated by human subject matter experts.
