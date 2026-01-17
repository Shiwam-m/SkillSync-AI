import re
import io
import os
import PyPDF2
import tempfile
import json
import time
from concurrent.futures import ThreadPoolExecutor

# Stable Imports for LangChain 0.3
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings


# --------------------------------------------------------------
# Resume Analysis Agent
# --------------------------------------------------------------
class ResumeAnalysisAgent:
    def __init__(self, api_key=None, model_name="gpt-4o", cutoff_score=75):
        """
        Initialize the ResumeAnalysisAgent.
        Args:
            api_key (str): API key for OpenAI or other LLM services.
            cutoff_score (int, optional): Minimum score to consider a candidate selected. Defaults to 75.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.cutoff_score = cutoff_score

        # LOCAL EMBEDDINGS
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name="all-MiniLM-L6-v2",
        #     model_kwargs={'device': 'cpu'}
        # )

        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            self.embeddings = None


        # OPENAI CONFIG
        if self.api_key:
            self.llm = ChatOpenAI(
                openai_api_key=self.api_key,
                model=self.model_name,
                temperature=0.1,
                timeout=60
            )

        # Internal state 
        self.resume_text = None
        self.rag_vectorstore= None 
        self.analysis_result = None 
        self.jd_text = None 
        self.extracted_skills = None 
        self.resume_weaknesses = []
        self.resume_strengths = []
        self.improvement_suggestions = {}


    # -------------------------------------------------------------------------
    # File Extraction Methods
    # -------------------------------------------------------------------------
    def extract_text_from_pdf(self, pdf_file):
        """ Extract text from a PDF file """
        try:
            if hasattr(pdf_file, 'getvalue'):
                reader = PyPDF2.PdfReader(pdf_file)
            else:
                reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text 
        except Exception as e:
            print(f"Error extracting text from PDF : {e}")
            return ""
        

    def extract_text_from_txt(self, txt_file):
        """Extract text from a TXT file."""
        try:
            if hasattr(txt_file, 'getvalue'):
                return txt_file.getvalue().decode('utf-8')
            else:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            print(f"Error extracting text from TXT file: {e}")
            return ""
        

    def extract_text_from_file(self, file):
        """
        Extract text from a file (PDF or TXT).

        Args:
            file: Uploaded file object.

        Returns:
            str: Extracted text or empty string.
        """
        if hasattr(file, 'name'):
            file_extension = file.name.split('.')[-1].lower()
        else:
            file_extension = file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            return self.extract_text_from_pdf(file)
        elif file_extension == 'txt':
            return self.extract_text_from_txt(file)
        else:
            print(f"Unsupported file extension : {file_extension}")
            return ""


    # -------------------------------------------------------------------------
    # Vector Store Methods
    # -------------------------------------------------------------------------
    def create_rag_vector_store(self, text):
        """
        Create a vector store for Retrieval-Augmented Generation (RAG).

        Args:
            text (str): Resume text.

        Returns:
            FAISS: Vector store object.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        """embedding model understand the relation b\w the chunks
            This is temp vector stre not in file 
        """
        vectorstore = FAISS.from_texts(chunks, self.embeddings)
        return vectorstore  

    
    def create_vector_store(self, text):
        """
            Create a simpler vector store for semantic skill analysis.
            Args:
                text (str): Text to embed.
            Returns:
            FAISS: Vector store object.
        """
        vectorstore = FAISS.from_texts([text], self.embeddings)
        return vectorstore


   # -------------------------------------------------------------------------
    # Skill Analysis Methods
    # -------------------------------------------------------------------------
    def analyze_skill(self, qa_chain, skill):
        """
        Analyze a specific skill in the resume.

        Args:
            qa_chain: QA chain object for semantic querying.
            skill (str): Skill to analyze.

        Returns:
            tuple: (skill, score [0-10], reasoning)
        """
        query = f"On a scale of 0-10, how clearly does the candidate mention proficiency in {skill}? Provide a numeric rating first, followed by reasoning."
        try:
            response = qa_chain.invoke(query)["result"]
            
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            match = re.search(r"(\d{1,2})", response_text)
            score = int(match.group(1)) if match else 0 
            reasoning = response_text.split('.', 1)[1].strip() if '.' in response_text and len(response_text.split('.')) > 1 else response_text
            
            return skill, min(score, 10), reasoning
        except Exception as e:
            print(f"Skill Analysis Error: {e}")
            return skill, 0, "Error during analysis"


    def analyze_resume_weeknesses(self):
        if not self.resume_text or not self.extracted_skills:
            return []
    
        missing = self.analysis_result.get("missing_skills", [])
        if not missing:
            return []

        # Combined all missing skills into ONE prompt to avoid RateLimitError
        skills_to_analyze = missing[:5] 
        skills_list_str = ", ".join(skills_to_analyze)

        prompt = f"""
        Analyze the following missing skills in the candidate's resume: {skills_list_str}
        
        Resume Content: {self.resume_text[:2000]}
        
        Provide a detailed analysis for each skill in the following JSON format:
        {{
            "weaknesses": [
                {{
                    "skill": "Skill Name",
                    "weakness": "Concise description of what is missing",
                    "improvement_suggestions": ["suggestion 1", "suggestion 2"],
                    "example_addition": "A bullet point to add"
                }}
            ]
        }}
        Return ONLY valid JSON.
        """
        
        try:
            raw_response = self.llm.invoke(prompt)
            # Use regex to find the JSON part in case the LLM adds markdown
            json_match = re.search(r'\{.*\}', raw_response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                # Map the data and ensure we include the score from the original analysis
                self.resume_weaknesses = []
                for w in data.get("weaknesses", []):
                    w["score"] = self.analysis_result.get("skill_scores", {}).get(w['skill'], 0)
                    self.resume_weaknesses.append(w)
            return self.resume_weaknesses
        except Exception as e:
            print(f"Error in batch weakness analysis: {e}")
            return []


    # -------------------------------------------------------------------------
    # Job Description Methods
    # -------------------------------------------------------------------------
    def extract_skill_from_jd(self, jd_text):
        """
        Extract technical skills from a job description.

        Args:
            jd_text (str): Job description text.

        Returns:
            list: List of extracted skills.
        """
        try:
            prompt = f"""
            Extract a comprehensive list of technicall skills, technologies, and competencies requiires from this job description. 
            Formate the output as a Python list of strings. Only include the list noothing else.

            Job Description:
            {jd_text}
            """
            response = self.llm.invoke(prompt)        
            skills = re.findall(r"'(.*?)'|\"(.*?)\"", response)
            return [s[0] or s[1] for s in skills if (s[0] or s[1])]
        
        except Exception as e:
            print(f"Error extracting skills from job description: {e}")
            return []
           

         
    # sementic skills analysis
    def sementic_skill_analysis(self, resume_text, skills):
        """
        Analyze resume skills semantically and score them.

        Args:
            resume_text (str): Resume text.
            skills (list): Skills to analyze.

        Returns:
            dict: Analysis results including scores and strengths/weaknesses.
        """
        vectorostore = self.create_vector_store(resume_text)
        retriever = vectorostore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm = self.llm,
            retriever=retriever,
        )    

        skill_score = {}
        skill_reasoning = {}
        missing_skills = []
        total_score = 0

        # FIX: Removed ThreadPoolExecutor to prevent RateLimitError
        for skill in skills:
            skill_name, score, reasoning = self.analyze_skill(qa_chain, skill)
            skill_score[skill_name] = score
            skill_reasoning[skill_name] = reasoning
            total_score += score 
            if score <= 5:
                missing_skills.append(skill_name)
            time.sleep(0.5) # Small 0.5s pause to respect Free Tier API limits
            
        overall_score = int((total_score / (10 * len(skills))) * 100) if skills else 0
        selected = overall_score >= self.cutoff_score
        reasoning = "Candidate evaluated based on explicit resume content using semantic similarity."
        strengths = [skill for skill, score in skill_score.items() if score >= 7]
        improvements_areas = missing_skills if not selected else []

        self.resume_strengths = strengths

        return {
            "overall_score": overall_score,
            "skill_scores": skill_score,
            "skill_reasonong": skill_reasoning,
            "selected": selected,
            "reasoning": reasoning,
            "missing_skills": missing_skills,
            "strengths": strengths, 
            "improvement_areas": improvements_areas
        }

        # This code analyzes a resume against multiple skills in parallel to produce an explainable, fair, and fast hiring decision.
        # with ThreadPoolExecutor(max_workers=5) as executor:
        #     results = list(executor.map(lambda skill: self.analyze_skill(qa_chain, skill), skills))
        # for skill, score, reasoning in results:
        #     skill_score[skill] = score
        #     skill_reasoning[skill] = reasoning
        #     total_score += score 
        #     if score <= 5:
        #         missing_skills.append(skill)
        # overall_score = int((total_score / (10 * len(skills))) * 100)
        # selected = overall_score >= self.cutoff_score
        # reasoning = "Candidate evaluated based on explict resume content using semantic similarity and clear numeric scoring."
        # strengths = [skill for skill, score in skill_score.items() if score >= 7]
        # improvements_areas = missing_skills if not selected else []
        # self.resume_strengths = strengths
        # return {
        #     "overall_score": overall_score,
        #     "skill_scores": skill_score,
        #     "skill_reasonong": skill_reasoning,
        #     "selected": selected,
        #     "reasoning": reasoning,
        #     "missing_skills": missing_skills,
        #     "strengths": strengths, 
        #     "improvement_areas": improvements_areas
        # }
    

    # -------------------------------------------------------------------------
    # Main Resume Analysis
    # -------------------------------------------------------------------------
    def analyze_resume(self, resume_file, role_requirements=None, custom_jd=None):
        """
        Analyze a resume against role requirements or custom JD.

        Args:
            resume_file: Uploaded resume file.
            role_requirements (list, optional): Skills required for a role.
            custom_jd: Custom job description file.

        Returns:
            dict: Analysis results including skills, strengths, weaknesses.
        """
        self.resume_text = self.extract_text_from_file(resume_file)

        # store in temporary file ...
        with tempfile.NamedTemporaryFile(delete=False, suffix='txt', mode='w', encoding='utf-8') as tmp:
            tmp.write(self.resume_text)
            self.resume_file_path = tmp.name

        self.rag_vectorstore = self.create_rag_vector_store(self.resume_text)

        if custom_jd:
            self.jd_text = self.extract_text_from_file(custom_jd)
            self.extracted_skills = self.extract_skill_from_jd(self.jd_text)
            self.analysis_result = self.sementic_skill_analysis(self.resume_text, self.extracted_skills)
        elif role_requirements:
            self.extracted_skills = role_requirements

            self.analysis_result = self.sementic_skill_analysis(self.resume_text, role_requirements)

        if self.analysis_result and "missing_skills" in self.analysis_result and self.analysis_result['missing_skills']:
            self.analyze_resume_weeknesses()
            self.analysis_result["detailed_weaknesses"] = self.resume_weaknesses 

        return self.analysis_result


    # -------------------------------------------------------------------------
    # RAG Q&A Methods
    # -------------------------------------------------------------------------
    def ask_question(self, question):
        """ Ask a question the resume """
        if not self.rag_vectorstore or not self.resume_text:
            return "Please analyze a resume first."

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.rag_vectorstore.as_retriever()
        )
        result = qa_chain.invoke(question)["result"]

        # Extract content from Chat message if necessary
        return result.content if hasattr(result, 'content') else str(result)
         
        
    # -------------------------------------------------------------------------
    # Interview Question Generation
    # -------------------------------------------------------------------------
    def generate_interview_questions(self, question_type, difficulty, num_questions):
        """
        Generate personalized interview questions based on the resume.

        Args:
            question_types (list): List of question types (e.g., Technical, HR).
            difficulty (str): Difficulty level (Easy, Medium, Hard).
            num_questions (int): Number of questions to generate.

        Returns:
            list: Generated questions as tuples (type, question text).
        """
        if not self.resume_text or not self.extracted_skills:
            return []

        try:
            context = f"""
            Resume Content: {self.resume_text[:1500]}
            Skills: {', '.join(self.extracted_skills)}
            """
            
            prompt = f""" 
            Generate {num_questions} personalized {difficulty.lower()} level interview questions for this candidate.
            Target Types: {', '.join(question_type)}

            Return ONLY a JSON object in this exact format:
            {{
                "questions": [
                    {{"type": "Technical", "question": "Your question text here?"}},
                    {{"type": "Behavioral", "question": "Your question text here?"}}
                ]
            }}
            {context}
            """
            response = self.llm.invoke(prompt)
            
            # Clean JSON parsing
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                # Convert to the list of tuples the UI expects
                raw_questions = [(q['type'], q['question']) for q in data.get("questions", [])]
                return raw_questions[:num_questions]
            
            return []
        except Exception as e:
            print(f"Error: {e}")
            return []
            

    def improve_resume(self, improvement_area, target_role=""):

        """
        Generate actionable suggestions to improve the resume.
        Args:
            improvement_area (list): List of areas to improve, e.g., ["Skills Highlighting", "Experience"]
            target_role (str): Optional target role/job title for context

        Returns:
            dict: Improvement suggestions structured by area
        """
        if not self.resume_text:
            return {}

        try:
            imporovements = {}
            
            # Handle Skills Highlighting separately if resume weaknesses exist
            for area in improvement_area:
                if area == "Skills Highlighting" and self.resume_weaknesses:
                    skill_improvements = {
                        "description" : "Your resume nedds to better highlight key skills that are important for the role.",
                        "specific": []
                    }

                    before_after_examples = {}
                    for weakness in self.resume_weaknesses:
                        skill_name = weakness.get("skill", "")
                        if "suggestions" in weakness and weakness["suggestions"]:
                            for suggestion in weakness["suggestions"]:
                                skill_improvements["specific"].append(f"** {skill_name}** : {suggestion}")

                        if "example" in weakness and weakness["example"]:
                            resume_chunks = self.resume_text.split('\n\n')
                            relevant_chunk = ""

                            for chunk in resume_chunks:
                                if skill_name.lower() in chunk.lower() or "experience" in chunk.lower():
                                    relevant_chunk = chunk
                                    break
                            
                            if relevant_chunk:
                                before_after_examples = {
                                    "before": relevant_chunk.strip(),
                                    "after": relevant_chunk.strip() + "\n " + weakness["example"]
                                }

                    if before_after_examples:
                        skill_improvements["before_after"] = before_after_examples
                    imporovements["Skills Highlighting"] = skill_improvements

            remaning_areas = [area for area in improvement_area  if area not in imporovements]
            if remaning_areas:
                llm = self.llm 
                
                # context with resume analysis and weakness 
                weaknesses_text = ""
                if self.resume_weaknesses:
                    weaknesses_text = "Resume Weaknesses:\n"
                    for i, weakness in enumerate(self.resume_weaknesses):
                        weaknesses_text += f"{i+1}. {weakness['skill']}: {weakness['derail']}\n"
                        if "suggestions" in weakness:
                            for j, sugg in enumerate(weakness["suggestions"]):
                                weaknesses_text += f"   - {sugg}\n"

                context = f"""
                Resume Content:
                {self.resume_text}

                skills to focous on: {', '.join(self.extracted_skills)}
                Strengths: {', '.join(self.analysis_result.get('strengths', []))}
                Area for improvement: {', '.join(self.analysis_result.get('missing_skills', []))}
                {weaknesses_text}

                Target role: {target_role if target_role else "Not Specified"}
                """
                
                prompt = f"""
                Provide detail suggestions to improve this resume in this following areas: {', '.join(remaning_areas)}
                {context}

                For each imporovement area, Provide:
                1. A general description of what needs improvement 
                2. 3-5 specific actioable suggestions 
                3. Where relevant, provide a before/after example 

                Formate the response as a JSON object with improvement area as keys, each containing:
                - "description": general description 
                - "specific": list of specific suggestions
                - "before_after": (where applicable) a dict with "before" and "after" examples

                Only include the requested imporovement areas that aren't already covered.
                Focous particularly on addressing the resume weaknesss identified.
                """

                response = llm.invoke(prompt)
                ai_improvements = {}
                json_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', response.content)
                if json_match:
                    try:
                        ai_improvements = json.loads(json_match.group(1))
                        imporovements.update(ai_improvements)
                    except json.JSONDecodeError:
                        pass

                # IF JSON parsing failed, create structured output manually 
                if not ai_improvements:
                    sections = response.content.split("##")

                    for section in sections:
                        if not sections.strip():
                            continue

                        lines = section.strip().split("\n")
                        area = None 

                        for line in lines:
                            if not area and line.strip():
                                area = line.strip()
                                imporovements[area] = {
                                    "description": "",
                                    "specific": []
                                }

                            elif area and "specific" in imporovements[area]:
                                if line.strip().startswith("- "):
                                    imporovements[area]["specific"].append(line.strip()[2:])
                                elif not imporovements[area]['description']:
                                    imporovements[area]["description"] += line.strip()

            """ 99% if in big clg : then rand any girl khas kr : villege girl all - in 
            - no need of merraige
            - Not facilities yaha pe aane ke bad : In sabo ka tentoram start 
            - Flutting style is best : sachine : mere satha and all things but not merrage 
            - body make .. good 
            - Not abuse ..
            - dar, bajte kahi bhi nhi : all are human... kise bhi bat ko laker..
            - Teeth...
            - chain, laptop banwana : flex, more small... (ashu)
            """
            # Ensure all requested areas are included
            for area in improvement_area :
                if area not in imporovements:
                    imporovements[area] = {
                        "description": f"Improvements needed in {area}",
                        "specific": ["Review and enhance this section"]
                    }

            return imporovements

        except Exception as e:
            print(f"Error generating resume improvements: {e}")
            return {area: {"description": "Error generating suggestions", "specific": []} for area in improvement_area }
        

    def get_improved_resume(self, target_role="", highlight_skills=""):
        """ Generate an improved version of the resume optimized for the job description """
        if not self.resume_text:
            return "Please upload and analyze a resume first."
        
        try:
            # Parse highlight skills if provided 
            skills_to_highlight = []
            if highlight_skills:

                if len(highlight_skills) > 100:
                    self.jd_text = highlight_skills
                    try:
                        parsed_skills = self.extract_skill_from_jd(highlight_skills)
                        if parsed_skills:
                            skills_to_highlight = parsed_skills
                        else:
                            skills_to_highlight = [s.strip() for s in highlight_skills.split(",") if s.strip()]
                    except:
                        skills_to_highlight = [s.strip() for s in highlight_skills.split(",") if s.strip()]
                else:
                    skills_to_highlight = [s.strip() for s in highlight_skills.split(",") if s.strip()]
            
            if not skills_to_highlight and self.analysis_result:
                skills_to_highlight = self.analysis_result.get('missing_skills', [])
                skills_to_highlight.extend([
                    skill for skill in self.analysis_result.get('strengths', [])
                    if skill not in skills_to_highlight
                ])

                if self.extracted_skills:
                    skills_to_highlight.extend([
                        skill for skill in self.extracted_skills
                        if skill not in skills_to_highlight
                    ])

            weakness_context = ""
            improvement_examples = ""

            if self.resume_weaknesses:
                weakness_context = "Address these specific weakness:\n"

                for weakness in self.resume_weaknesses:
                    skill_name = weakness.get('skill', '')
                    weakness_context += f"- {skill_name}: {weakness.get('detail', '')}\n"

                    if 'suggestions' in weakness and weakness['suggestions']:
                        weakness_context += " Suggested improvements:\n"
                        for suggestion in weakness['suggestions']:
                            weakness_context += f" *{suggestion}\n"
                        
                    if 'example' in weakness and weakness['example']:
                        improvement_examples += f"For {skill_name}: {weakness['example']}\n\n"

            llm = self.llm

            jd_context = ""
            if self.jd_text:
                jd_context = f"job Description:\n{self.jd_text}\n\n"
            elif target_role:
                jd_context = f"Target Role: {target_role}\n\n"

            prompt = f"""
            Rewrite and improve this resume to make it highly optimized for the target job. 
            {jd_context}
            Original Resume:
            {self.resume_text}

            Skills to highlight (in order of priority): {', '.join(skills_to_highlight)}
            {weakness_context}

            Here are specific example of content to add:
            {improvement_examples}

            Please improve the resume by:
            1. Adding strong, quantifiable achievements 
            2. Highlighting the specified skills strategically for ATS scanning
            3. Adddressing all the weakness area identified with the specific suggestion proovided
            4. Incorporating the example improvements provoided above 
            5. Structuring information in a clear, professsional formate 
            6. Using industry-standard terminology 
            7. Ensuring all relevant experience is properly emphasized 
            8. Adding measurable outcomes and achievements 

            Return only the improved resume text without any additional explanations.
            Formate the resume in a modern, clear style with clear section headings.
            """
            response = llm.invoke(prompt)
            improved_resume = response.content.strip()

            with tempfile.NamedTemporaryFile(delete=False, suffix='txt', mode='w', encoding='utf-8') as tmp:
                tmp.write(improved_resume)
                self.improved_resume_path = tmp.name

            return improved_resume
        
        except Exception as e:
            print(f"Error genetaing improved resume: {e}")
            return "Error generating improved resume. Please try again."

        
    def cleanup(self):
        """ Clean up temporary files """
        try:
            if hasattr(self, 'resume_file_path') and os.path.exists(self.resume_file_path):
                os.unlink(self.resume_file_path)

            if hasattr(self, 'improved_resume_path') and os.path.exists(self.improved_resume_path):
                os.unlink(self.improved_resume_path)
        except Exception as e:
            print(f"Error cleaning up temporary files: {e}")
