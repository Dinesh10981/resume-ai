# 🤖 Smart AI Resume Analyzer

An intelligent resume screening and ranking system powered by NLP and LLMs. Upload candidate resumes, compare them against job descriptions, and receive detailed AI-powered recruiter feedback — all in one place.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1+-green?logo=flask&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Local-brightgreen?logo=mongodb&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-purple?logo=openai&logoColor=white)

---

## ✨ Features

### 🔍 Core Analysis
- **PDF Resume Parsing** — Extracts text from uploaded PDF resumes using `pdfplumber`
- **Semantic Similarity** — Computes cosine similarity between resumes and job descriptions using OpenAI `text-embedding-3-small`, with a lightweight lexical fallback
- **Skill Extraction** — Matches 80+ industry skills using word-boundary regex patterns
- **Skill Gap Analysis** — Identifies matched and missing skills against job requirements
- **Candidate Ranking** — Weighted scoring (70% semantic + 30% skill match) with ranked results

### 🧠 AI-Powered Insights (5 Features)
1. **Recruiter Verdict** — Human-like executive summary from a senior recruiter's perspective
2. **Job Fit Breakdown** — Match percentage with specific strengths and weaknesses
3. **Resume Improvement Suggestions** — Prioritized as Critical / Important / Optional
4. **Career Roadmap** — Personalized role recommendations, skills to learn, certifications, and project ideas
5. **Smart Reasoning Chain** — Step-by-step explanation of how the AI arrived at its analysis

### 📊 Dashboard
- **Candidate Database** — All analyzed candidates stored in MongoDB
- **Expandable AI Cards** — Click "Expand" to view full AI analysis for each candidate
- **Delete Records** — Remove candidates from the database with one click
- **Ranked Results** — Candidates sorted by match score

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.11, Flask |
| **AI/ML** | OpenAI GPT-4o-mini, text-embedding-3-small, skill extraction |
| **Database** | MongoDB (local via MongoDB Compass) |
| **Frontend** | HTML5, CSS3 (Glassmorphism design), Vanilla JS |
| **PDF Parsing** | pdfplumber |

---

## 📁 Project Structure

```
resume-ai/
├── app.py                  # Flask server — routes, validation, DB integration
├── model_utils.py          # AI engine — PDF extraction, NLP, LLM prompts
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys, config)
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── static/
│   └── style.css           # Dark-mode glassmorphism UI styles
├── templates/
│   ├── index.html          # Upload page with drag-and-drop
│   └── result.html         # Results dashboard with AI insight panels
└── uploads/                # Temporary PDF storage (auto-cleaned)
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.11+**
- **MongoDB** running locally (install via [MongoDB Community](https://www.mongodb.com/try/download/community) or [MongoDB Compass](https://www.mongodb.com/products/compass))
- **OpenAI API Key** (optional — app works without it using fallback mode)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/resume-ai.git
cd resume-ai
```

### 2. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Edit the `.env` file in the project root:

```env
# LLM Configuration (Optional — works without it using fallback mode)
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# MongoDB
MONGODB_URI=mongodb://localhost:27017/

# Flask
FLASK_SECRET_KEY=your-random-secret-key
FLASK_DEBUG=true
```

> **Note:** Without a valid API key, the app still works but uses basic keyword-matching feedback instead of full AI analysis.

### 5. Start MongoDB
Make sure MongoDB is running locally. You can verify by opening MongoDB Compass and connecting to `mongodb://localhost:27017/`.

### 6. Run the Application
```bash
python app.py
```

Open your browser at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📖 How to Use

### Analyzing Resumes
1. Open the app at `http://127.0.0.1:5000`
2. Paste the **job description** in the text area
3. Upload one or more **PDF resumes** (drag & drop or click to browse)
4. Click **"Analyze & Rank Candidates"**
5. Wait for the AI to process (a loading spinner will appear)
6. View ranked results with scores, skill tags, and AI insights

### Viewing AI Insights
- Click the **"Expand 🔍"** button on any candidate row
- This reveals the full AI analysis panel with:
  - 📊 Job Fit Breakdown (match %, strengths, weaknesses)
  - 💡 Resume Improvement Suggestions (prioritized)
  - 🚀 Career Roadmap (roles, skills, certifications, projects)
  - 🧠 AI Reasoning Chain

### Managing Candidates
- Click **"📊 View Stored Candidates Database"** to see all historical analyses
- Click the **🗑️** button to delete a candidate record
- All data is stored in MongoDB (`resume_db.candidates`)

---

## ⚙️ API Key Configuration

The app supports any **OpenAI-compatible API**:

| Provider | `OPENAI_BASE_URL` | `LLM_MODEL` |
|----------|-------------------|-------------|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini`, `gpt-4o` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.1-70b-versatile` |
| Together AI | `https://api.together.xyz/v1` | `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` |

### Without an API Key
The app runs in **fallback mode**:
- ✅ PDF parsing works
- ✅ Semantic similarity works
- ✅ Skill matching works
- ✅ Scoring and ranking works
- ⚠️ AI feedback uses basic keyword templates instead of LLM-generated insights

---

## 🔒 Security

- API keys are stored in `.env` (never committed to Git)
- File uploads restricted to **PDF only** with a **16MB size limit**
- Filenames sanitized with `werkzeug.secure_filename` to prevent path traversal
- Uploaded files are **automatically deleted** after processing
- MongoDB ObjectIds properly serialized for safe rendering

---

## 🗄️ Database

Data is stored in MongoDB:
- **Database:** `resume_db`
- **Collection:** `candidates`

Each document contains:
```json
{
  "name": "candidate_resume.pdf",
  "score": 72.5,
  "similarity": 68.3,
  "skills": ["python", "flask", "sql"],
  "matched": ["python", "sql"],
  "missing": ["docker", "aws"],
  "feedback": "Strong in: python, sql. Needs improvement in: docker, aws.",
  "llm_feedback": {
    "recruiter_verdict": "...",
    "job_fit": { "match_percentage": 72, "explanation": "...", "strengths": [...], "weaknesses": [...] },
    "improvement_suggestions": [{ "priority": "critical", "area": "...", "suggestion": "..." }],
    "career_recommendations": { "suitable_roles": [...], "skills_to_learn": [...], "certifications": [...], "projects_to_build": [...] },
    "reasoning_chain": "..."
  },
  "created_at": "2026-06-28T12:00:00Z"
}
```

You can view and manage the database using **MongoDB Compass** connected to `mongodb://localhost:27017/`.

---

## 📝 License

This project is for educational and personal use.

---

## 🙏 Acknowledgments

- [OpenAI embeddings](https://platform.openai.com/docs/guides/embeddings) for semantic similarity
- [pdfplumber](https://github.com/jsvine/pdfplumber) for PDF text extraction
- [OpenAI](https://openai.com/) for LLM-powered analysis
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [MongoDB](https://www.mongodb.com/) for data persistence

