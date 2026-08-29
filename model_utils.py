"""Lightweight NLP and OpenAI helpers for resume screening."""

import json
import logging
import math
import os
import re
import time
from collections import Counter

import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)

_openai_client = None


def _get_openai_client():
    """Create the API client lazily inside the serving worker."""
    global _openai_client
    key = os.getenv("OPENAI_API_KEY")
    if not key or key == "your-api-key-here":
        return None
    if _openai_client is None:
        _openai_client = OpenAI(
            api_key=key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            timeout=45.0,
            max_retries=1,
        )
    return _openai_client


SKILL_DB = [
    "python", "java", "javascript", "typescript", "c\\+\\+", "c#", "ruby",
    "golang", "rust", "kotlin", "swift", "scala", "php", "r programming",
    "react", "angular", "vue\\.js", "next\\.js", "node\\.js", "express",
    "html", "css", "tailwind", "bootstrap", "sass", "machine learning",
    "deep learning", "natural language processing", "computer vision",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "data analysis", "data engineering", "data visualization", "power bi",
    "tableau", "sql", "mysql", "postgresql", "mongodb", "redis",
    "elasticsearch", "dynamodb", "cassandra", "aws", "azure", "gcp",
    "google cloud", "docker", "kubernetes", "terraform", "jenkins", "ci/cd",
    "github actions", "gitlab", "linux", "nginx", "flask", "django",
    "spring boot", "fastapi", "rest api", "graphql", "microservices", "git",
    "agile", "scrum", "jira", "langchain", "llm", "prompt engineering", "rag",
    "openai", "hugging face", "transformers", "project management",
    "team leadership", "communication", "problem solving", "system design",
    "excel", "matlab", "spark", "hadoop", "kafka", "airflow",
]

_skill_patterns = {
    skill: re.compile(r"\b" + skill + r"\b", re.IGNORECASE) for skill in SKILL_DB
}
_token_pattern = re.compile(r"[a-z0-9+#.]{2,}", re.IGNORECASE)


def extract_text(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()


def _cosine(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _lexical_similarity(text1, text2):
    left = Counter(_token_pattern.findall(text1.lower()))
    right = Counter(_token_pattern.findall(text2.lower()))
    vocabulary = left.keys() | right.keys()
    return _cosine([left[token] for token in vocabulary], [right[token] for token in vocabulary])


def compute_similarity(text1, text2):
    """Use hosted semantic embeddings without loading PyTorch into Render RAM."""
    client = _get_openai_client()
    if client is not None:
        try:
            response = client.embeddings.create(
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                input=[text1[:12000], text2[:12000]],
                encoding_format="float",
            )
            return _cosine(response.data[0].embedding, response.data[1].embedding)
        except Exception as exc:
            logger.warning("Embedding request failed; using lexical fallback: %s", exc)
    return _lexical_similarity(text1, text2)


def extract_skills(text):
    found = []
    for skill, pattern in _skill_patterns.items():
        if pattern.search(text):
            found.append(skill.replace("\\", ""))
    return sorted(set(found))


def skill_gap(job_skills, resume_skills):
    matched = sorted(set(job_skills) & set(resume_skills))
    missing = sorted(set(job_skills) - set(resume_skills))
    return matched, missing


def final_score(similarity, skill_score):
    return round((similarity * 0.7 + skill_score * 0.3) * 100, 2)


def generate_feedback(matched, missing):
    parts = []
    if matched:
        parts.append("Strong in: " + ", ".join(matched) + ".")
    if missing:
        parts.append("Needs improvement in: " + ", ".join(missing) + ".")
    else:
        parts.append("Great match for the role.")
    return " ".join(parts)


SYSTEM_PROMPT = """You are a senior technical recruiter and career coach.
Compare the resume with the job description. Give specific, evidence-based feedback,
not generic advice. Return only a valid JSON object in the requested schema. Provide a
concise decision rationale; do not reveal private chain-of-thought."""

USER_PROMPT_TEMPLATE = """JOB DESCRIPTION:
{job_desc}

CANDIDATE RESUME:
{resume_text}

PRE-COMPUTED ANALYSIS:
- Resume skills: {resume_skills}
- Matched skills: {matched_skills}
- Missing skills: {missing_skills}
- Semantic similarity: {similarity_pct}%

Return this exact JSON structure:
{{
  "recruiter_verdict": "3-4 sentence recruiter summary",
  "job_fit": {{
    "match_percentage": 0,
    "explanation": "evidence-based explanation",
    "strengths": ["specific strength"],
    "weaknesses": ["specific weakness"]
  }},
  "improvement_suggestions": [
    {{"priority": "critical", "area": "area", "suggestion": "action"}}
  ],
  "career_recommendations": {{
    "suitable_roles": ["role"],
    "skills_to_learn": ["skill and reason"],
    "certifications": ["certification"],
    "projects_to_build": ["project idea"]
  }},
  "reasoning_chain": "concise evidence-based decision rationale"
}}
"""


def _fallback_feedback(matched, missing, similarity):
    return {
        "recruiter_verdict": f"The candidate matches {len(matched)} required skills. "
        + generate_feedback(matched, missing),
        "job_fit": {
            "match_percentage": int(similarity * 100),
            "explanation": "Evaluated using document similarity and skill overlap.",
            "strengths": [f"Knowledge of {skill}" for skill in matched[:3]]
            or ["Matches baseline terminology"],
            "weaknesses": [f"Missing {skill}" for skill in missing[:3]]
            or ["No direct skill gaps detected"],
        },
        "improvement_suggestions": [{
            "priority": "critical",
            "area": "Skill gap",
            "suggestion": f"Demonstrate: {', '.join(missing[:3])}"
            if missing else "Add measurable outcomes to major achievements.",
        }],
        "career_recommendations": {
            "suitable_roles": ["Software Developer", "IT Professional"],
            "skills_to_learn": missing[:3] or ["System design"],
            "certifications": ["A relevant cloud or role-based certification"],
            "projects_to_build": ["Build a deployed project demonstrating the target skills"],
        },
        "reasoning_chain": "The recommendation is based on document similarity and verified skill overlap.",
    }


def generate_llm_feedback(resume_text, job_desc, matched, missing, resume_skills=None, similarity=0.0):
    client = _get_openai_client()
    if client is None:
        return _fallback_feedback(matched, missing, similarity)

    prompt = USER_PROMPT_TEMPLATE.format(
        job_desc=job_desc[:3000],
        resume_text=resume_text[:4000],
        resume_skills=", ".join(resume_skills or []),
        matched_skills=", ".join(matched),
        missing_skills=", ".join(missing),
        similarity_pct=round(similarity * 100, 1),
    )
    required = {
        "recruiter_verdict", "job_fit", "improvement_suggestions",
        "career_recommendations", "reasoning_chain",
    }

    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
                max_tokens=1400,
            )
            parsed = json.loads(response.choices[0].message.content)
            if required.issubset(parsed):
                return parsed
            raise ValueError("LLM JSON is missing required fields")
        except Exception as exc:
            logger.warning("LLM feedback attempt %d failed: %s", attempt + 1, exc)
            if attempt == 0:
                time.sleep(0.5)

    return _fallback_feedback(matched, missing, similarity)
