# 🤖 SkillSync AI: Next-Gen Recruitment Intelligence System
SkillSync AI is a professional-grade recruitment assistant that leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to automate resume screening, technical interview preparation, and ATS optimization.     
Designed with a modern "Indigo & Slate" minimal interface, this tool provides data-driven insights for high-end IT recruitment environments.


## 🚀 Key Features
- **Multi-Model Intelligence:** Model-agnostic architecture supporting OpenAI (GPT-4o), Google Gemini (1.5 Pro), Anthropic (Claude 3.5), and Groq (Llama 3.1).
- **Semantic Tech-Audit:** Scores resumes (0-100) against industry-standard IT roles or custom Job Descriptions (JDs) using semantic vector similarity.
- **Multi-Candidate Leaderboard:** Batch processing capability to upload up to 5 resumes simultaneously, ranking them from best-to-worst based on specific role alignment.
- **Contextual RAG Engine:** Ask specific questions about a candidate's history, such as "Does the applicant have production experience with AWS?"
- **AI Interview Architect:** Generates personalized Technical, Scenario-based, and Behavioral questions based specifically on the candidate’s unique background.
- **ATS Optimization Engine:** Identifies technical gaps and provides "Before & After" examples to enhance resume formatting for Applicant Tracking Systems.
- **Enterprise UI:** A minimal, professional dashboard designed for high-end IT recruitment environments.

## 🛠️ Tech Stack
- **Frontend:** Streamlit (Custom Indigo/Slate Theme)
- **Orchestration:** LangChain 0.3 (RetrievalQA, Multi-Model Router, Text Splitters)
- **LLM:** OpenAI GPT-4o, Google Gemini 1.5 Pro, Anthropic Claude 3.5 Sonnet, Groq (Llama 3.1 / Mixtral).
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local Processing)
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **PDF Processing:** PyPDF2
- **Deployment:** Docker & Hugging Face Spaces

## 🏗️ Architecture Workflow
1. **Ingestion:** Extracts raw text from PDF/TXT resumes using specialized parsers.
2. **Vectorization:** Splits content into semantic chunks converted into high-dimensional vectors via local embeddings.
3. **Storage:** Chunks are indexed in a local FAISS vector store for sub-second retrieval.
4. **Intelligence Layer:** 
    - **Multi-Model Router:** Dynamically switches between AI providers based on user preference.
    - **Audit Mode:** Performs semantic scoring across specific technical competencies.
    - **Ranking Mode:** Iterative semantic evaluation of multiple documents to produce a comparative leaderboard.
    - **Q&A Mode:** Uses a RetrievalQA chain to provide grounded answers based only on the document.


## 🚀 Deployment (Hugging Face Spaces)
This project is containerized with **Docker** and deployed on **Hugging Face Spaces** for high-performance AI hosting.

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.11+
- An API Key from OpenAI, Google AI Studio, Anthropic, or Groq.

### 2. Clone the Repository
    git clone https://github.com/Shiwam-m/SkillSync-AI
    cd skillsync-ai

### 3. Setup Virtual Environment
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Linux/Mac:
    source venv/bin/activate

### 4. Install Dependencies
    pip install -r requirements.txt

### 5. Run the Application
    streamlit run app.py

## 📸 Usage Guide
- **Configure:** Select your preferred AI Provider and enter your API key in the sidebar.
- **Upload:** Drop a PDF resume into the "Resume Analysis" tab.
- **Target:** Select a predefined technical role (e.g., AI Engineer, DevOps) or upload a custom JD.
- **Audit:** Click "Analyze Resume" to generate the competency report.
- **Batch Rank:** Use the "Batch Ranking" tab to compare up to 5 candidates at once for a specific role to find the best fit quickly.
- **Optimize:** Use the "Interview Questions" and "Resume Improvements" tabs to prepare for the hiring process.

## 📜 License
- This project is licensed under the MIT License.

## [IMPORTANT]
- Disclaimer : SkillSync AI is an assistant tool. Automated results should always be validated by human subject matter experts.
