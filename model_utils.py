"""
model_utils.py — AI Engine for Smart Resume Analyzer
=====================================================
Handles PDF extraction, skill matching, semantic similarity,
and LLM-powered recruiter feedback generation.
"""

import os
import re
import json
import time
import logging

import pdfplumber
import spacy
from openai import OpenAI
from dotenv import load_dotenv

# ──────────────────────────────────────
# Configuration
# ──────────────────────────────────────
load_dotenv()

logger = logging.getLogger(__name__)

# Lazy-loaded spaCy and SentenceTransformer to reduce memory at process start
nlp = None
_sentence_model = None

def _get_nlp():
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = None
    return nlp

def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _sentence_model

# LLM Client setup
env_key = os.getenv("OPENAI_API_KEY")
api_key = env_key if (env_key and env_key != "your-api-key-here") else None

base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None

# ──────────────────────────────────────
# Skill Database (80+ industry skills)
# ──────────────────────────────────────
SKILL_DB = [
    # Programming Languages
    "python", "java", "javascript", "typescript", "c\\+\\+", "c#",
    "ruby", "golang", "rust", "kotlin", "swift", "scala", "php",
    "r programming",
    # Web & Frontend
    "react", "angular", "vue\\.js", "next\\.js", "node\\.js", "express",
    "html", "css", "tailwind", "bootstrap", "sass",
    # Data & ML
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "data analysis", "data engineering",
    "data visualization", "power bi", "tableau",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "dynamodb", "cassandra",
    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "terraform", "jenkins", "ci/cd", "github actions", "gitlab",
    "linux", "nginx",
    # Tools & Frameworks
    "flask", "django", "spring boot", "fastapi",
    "rest api", "graphql", "microservices",
    "git", "agile", "scrum", "jira",
    # AI & LLM
    "langchain", "llm", "prompt engineering", "rag",
    "openai", "hugging face", "transformers",
    # Soft Skills & Domains
    "project management", "team leadership", "communication",
    "problem solving", "system design",
    "excel", "matlab", "spark", "hadoop", "kafka", "airflow",
]

# Compile regex patterns for word-boundary matching
_skill_patterns = {}
for skill in SKILL_DB:
    # Escape special regex characters in skills like c++ and vue.js
    _skill_patterns[skill] = re.compile(r"\b" + skill + r"\b", re.IGNORECASE)

# ──────────────────────────────────────
# Core Functions
# ──────────────────────────────────────
def extract_text(file_path):
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()  # Return original case, lowercasing will happen when parsing skills/embedding


def compute_similarity(text1, text2):
    """Compute cosine similarity between two texts using Sentence Transformers."""
    model = _get_sentence_model()
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    from sentence_transformers import util
    return util.cos_sim(emb1, emb2).item()


def extract_skills(text):
    """Extract skills using pre-compiled regex patterns."""
    text_lower = text.lower()
    skills_found = []
    for skill, pattern in _skill_patterns.items():
        if pattern.search(text_lower):
            display_name = skill.replace("\\", "")
            skills_found.append(display_name)
    return list(set(skills_found))


def skill_gap(job_skills, resume_skills):
    """Compare job skills with candidate skills and identify overlap/gap."""
    matched = list(set(job_skills) & set(resume_skills))
    missing = list(set(job_skills) - set(resume_skills))
    return matched, missing


def final_score(similarity, skill_score):
    """Compute the weighted final match score."""
    return round((similarity * 0.7 + skill_score * 0.3) * 100, 2)


def generate_feedback(matched, missing):
    """Fallback simple feedback string."""
    feedback = ""
    if matched:
        feedback += "Strong in: " + ", ".join(matched) + ". "
    if missing:
        feedback += "Needs improvement in: " + ", ".join(missing)
    else:
        feedback += "Great match for the role."
    return feedback


# ──────────────────────────────────────
# LLM Feedback Generator
# ──────────────────────────────────────

SYSTEM_PROMPT = """You are an elite Senior Technical Recruiter, HR Lead, and Executive Career Coach.
Analyze the candidate's resume against the provided job description requirements.

Your feedback must satisfy these rules:
- Think step-by-step internally like a human expert.
- Avoid generic templates or robotic writing.
- Highlight specific strengths, achievements, and project items in the resume.
- Prioritize resume improvement suggestions clearly as Critical, Important, or Optional.
- Provide a personalized career path recommendation (roles, skills, certifications, and realistic project ideas).
- Match percentage must be justified.

Return ONLY a valid JSON object matching the schema below. Do not wrap the JSON in markdown formatting (like ```json). Ensure keys exist and have specific values."""

USER_PROMPT_TEMPLATE = """Analyze this candidate's details against the Job Description:

JOB DESCRIPTION:
{job_desc}

CANDIDATE RESUME:
{resume_text}

PRE-COMPUTED SKILL ANALYSIS:
- Skills found in resume: {resume_skills}
- Matching job skills: {matched_skills}
- Missing job skills: {missing_skills}
- Semantic similarity: {similarity_pct}%

Please output a JSON response structured exactly like this:
{{
  "recruiter_verdict": "Detailed executive summary (3-4 sentences) evaluating strengths and gaps from a recruiter perspective.",
  "job_fit": {{
    "match_percentage": <integer 0-100>,
    "explanation": "2-3 sentences justifying the match score based on experience and overlap.",
    "strengths": ["Strength 1 (specific to resume content)", "Strength 2"],
    "weaknesses": ["Weakness 1 (specific to resume content)", "Weakness 2"]
  }},
  "improvement_suggestions": [
    {{
      "priority": "critical",
      "area": "Area name",
      "suggestion": "Detailed, specific suggestion for improving the resume (e.g. ATS optimization, formatting, project description)."
    }},
    {{
      "priority": "important",
      "area": "Area name",
      "suggestion": "Actionable suggestion."
    }},
    {{
      "priority": "optional",
      "area": "Area name",
      "suggestion": "Nice-to-have feedback."
    }}
  ],
  "career_recommendations": {{
    "suitable_roles": ["Role 1", "Role 2"],
    "skills_to_learn": ["Skill 1 (with justification)", "Skill 2"],
    "certifications": ["Certification 1", "Certification 2"],
    "projects_to_build": ["Detailed project idea 1 based on their skills", "Project idea 2"]
  }},
  "reasoning_chain": "Your step-by-step recruiter rationale connecting the candidate's experience and education to the target role (3-4 sentences)."
}}
"""

def generate_llm_feedback(resume_text, job_desc, matched, missing, resume_skills=None, similarity=0.0):
    """Call the LLM to get deep structured analysis, falling back to rule-based analysis on failure."""
    prompt = USER_PROMPT_TEMPLATE.format(
        job_desc=job_desc[:3000],
        resume_text=resume_text[:4000],
        resume_skills=", ".join(resume_skills or []),
        matched_skills=", ".join(matched),
        missing_skills=", ".join(missing),
        similarity_pct=round(similarity * 100, 1)
    )

    for attempt in range(3):
        try:
            # If no LLM client is configured, skip API call and use fallback
            if client is None:
                raise RuntimeError("No LLM client configured; using fallback response")

            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            raw_content = response.choices[0].message.content.strip()

            # Clean markdown code block wraps if present
            if raw_content.startswith("```"):
                raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content)
                raw_content = re.sub(r"\s*```$", "", raw_content)

            parsed = json.loads(raw_content)
            # Ensure correct keys are populated
            required = ["recruiter_verdict", "job_fit", "improvement_suggestions", "career_recommendations", "reasoning_chain"]
            if all(k in parsed for k in required):
                return parsed
            else:
                raise ValueError("JSON missing required fields")

        except Exception as e:
            logger.warning("LLM attempt %d failed: %s", attempt + 1, e)
            time.sleep(1)

    # Fallback structure
    return {
        "recruiter_verdict": f"The candidate matches {len(matched)} of the required skills. " + generate_feedback(matched, missing),
        "job_fit": {
            "match_percentage": int(similarity * 100),
            "explanation": "Evaluated based on semantic similarity and skill overlap.",
            "strengths": [f"Knowledge of {s}" for s in matched[:3]] if matched else ["Matches baseline keywords"],
            "weaknesses": [f"Missing {s}" for s in missing[:3]] if missing else ["No direct keyword gaps"]
        },
        "improvement_suggestions": [
            {
                "priority": "critical",
                "area": "Skill Gap",
                "suggestion": f"Focus on acquiring or highlighting: {', '.join(missing[:3])}" if missing else "Optimize formatting for ATS."
            }
        ],
        "career_recommendations": {
            "suitable_roles": ["Software Developer", "IT Professional"],
            "skills_to_learn": missing[:3] if missing else ["Advanced System Architecture"],
            "certifications": ["Relevant cloud developer or database certifications"],
            "projects_to_build": ["Create a portfolio showing practical integration of these skills."]
        },
        "reasoning_chain": "Fallback logic triggered due to API timeout or error. Generated using keyword heuristic matching."
    }