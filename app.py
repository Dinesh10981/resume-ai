"""
app.py — Flask Server for Smart AI Resume Screening
===================================================
Handles routes, secure file uploads, validation, database integration,
and coordinates the AI screening pipeline.
"""

import os
from datetime import datetime
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

# Import core AI and helper functions
from model_utils import (
    extract_text,
    compute_similarity,
    extract_skills,
    skill_gap,
    final_score,
    generate_feedback,
    generate_llm_feedback
)

app = Flask(__name__)


def get_server_config():
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() in ("true", "1", "yes")
    return {
        "host": os.getenv("FLASK_HOST", "0.0.0.0"),
        "port": int(os.getenv("PORT", "5000")),
        "debug": debug_mode,
    }


# ──────────────────────────────────────
# Security & Configuration Settings
# ──────────────────────────────────────
app.secret_key = os.getenv("FLASK_SECRET_KEY", "prod-session-key-fallback")
UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Set 16MB file upload limit
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"pdf"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# MongoDB connection from env variables
mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client["resume_db"]
collection = db["candidates"]

# ──────────────────────────────────────
# HTTP Routes
# ──────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    # 1. Validate Form Input
    job_desc = request.form.get("job_desc", "").strip()
    if not job_desc:
        flash("Job description is required.", "error")
        return redirect(url_for("index"))

    files = request.files.getlist("resumes")
    if not files or files[0].filename == "":
        flash("At least one resume file must be selected.", "error")
        return redirect(url_for("index"))

    job_skills = extract_skills(job_desc)
    results = []

    for file in files:
        if not file or not allowed_file(file.filename):
            flash(f"Invalid file type for {file.filename}. Only PDF is allowed.", "error")
            continue

        # 2. Save file securely
        filename = secure_filename(file.filename)
        # Ensure name uniqueness inside temp uploads folder if concurrent requests run
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        
        try:
            file.save(filepath)

            # 3. Extract text
            resume_text = extract_text(filepath)
            if not resume_text:
                flash(f"Could not read text content from {filename}.", "error")
                continue

            # 4. Perform Analysis
            similarity = compute_similarity(job_desc, resume_text)
            resume_skills = extract_skills(resume_text)
            matched, missing = skill_gap(job_skills, resume_skills)

            skill_score = len(matched) / (len(job_skills) + 1)
            final = final_score(similarity, skill_score)

            # 5. Call LLM feedback generator
            llm_feedback = generate_llm_feedback(
                resume_text,
                job_desc,
                matched,
                missing,
                resume_skills=resume_skills,
                similarity=similarity
            )

            # 6. Save Data to MongoDB
            data = {
                "name": filename,
                "score": final,
                "similarity": round(similarity * 100, 2),
                "skills": resume_skills,
                "matched": matched,
                "missing": missing,
                "feedback": generate_feedback(matched, missing),
                "llm_feedback": llm_feedback,
                "created_at": datetime.utcnow()
            }
            
            # Save into DB
            collection.insert_one(data)
            
            # Convert ObjectId to string for HTML rendering
            data["_id"] = str(data["_id"])
            results.append(data)

        except Exception as e:
            app.logger.error("Error processing file %s: %s", filename, e)
            flash(f"Failed to analyze {filename}: {str(e)}", "error")

        finally:
            # 7. Clean up temporary uploaded file
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as cleanup_err:
                    app.logger.warning("Failed to remove temp file %s: %s", filepath, cleanup_err)

    # Sort results by score (descending)
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return render_template("result.html", results=results)


@app.route("/dashboard")
def dashboard():
    # Fetch historical candidate screens, format Mongo ID to string
    try:
        cursor = collection.find().sort("score", -1)
        data = []
        for item in cursor:
            item["_id"] = str(item["_id"])
            data.append(item)
    except Exception as e:
        app.logger.error("Database connection failure: %s", e)
        flash("Could not connect to database to fetch candidates dashboard.", "error")
        data = []

    return render_template("result.html", results=data)


@app.route("/delete/<candidate_id>", methods=["POST"])
def delete_candidate(candidate_id):
    try:
        collection.delete_one({"_id": ObjectId(candidate_id)})
        flash("Candidate record deleted successfully.", "success")
    except Exception as e:
        app.logger.error("Failed to delete candidate %s: %s", candidate_id, e)
        flash("Failed to delete candidate record.", "error")
    return redirect(url_for("dashboard"))


# ──────────────────────────────────────
# Application Execution
# ──────────────────────────────────────

if __name__ == "__main__":
    config = get_server_config()
    app.run(host=config["host"], port=config["port"], debug=config["debug"])
