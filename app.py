"""Flask server for the AI resume screening application."""

import os
from datetime import datetime, timezone

from bson import ObjectId
from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, url_for
from pymongo import MongoClient
from werkzeug.utils import secure_filename

load_dotenv()

from model_utils import (  # noqa: E402
    compute_similarity,
    extract_skills,
    extract_text,
    final_score,
    generate_feedback,
    generate_llm_feedback,
    skill_gap,
)

app = Flask(__name__)


def get_server_config():
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() in ("true", "1", "yes")
    return {
        "host": os.getenv("FLASK_HOST", "0.0.0.0"),
        "port": int(os.getenv("PORT", "5000")),
        "debug": debug_mode,
    }


app.secret_key = os.getenv("FLASK_SECRET_KEY", "prod-session-key-fallback")
UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf"}
MAX_RESUMES_PER_REQUEST = int(os.getenv("MAX_RESUMES_PER_REQUEST", "10"))
DASHBOARD_LIMIT = int(os.getenv("DASHBOARD_LIMIT", "50"))

_mongo_client = None
_candidate_collection = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_candidate_collection():
    """Create MongoDB objects lazily inside the Gunicorn worker."""
    global _mongo_client, _candidate_collection
    if _candidate_collection is None:
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        _mongo_client = MongoClient(
            mongo_uri,
            connect=False,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
        )
        _candidate_collection = _mongo_client["resume_db"]["candidates"]
    return _candidate_collection


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return {"status": "ok"}, 200


@app.route("/upload", methods=["POST"])
def upload():
    job_desc = request.form.get("job_desc", "").strip()
    if not job_desc:
        flash("Job description is required.", "error")
        return redirect(url_for("index"))

    files = [item for item in request.files.getlist("resumes") if item.filename]
    if not files:
        flash("At least one resume file must be selected.", "error")
        return redirect(url_for("index"))
    if len(files) > MAX_RESUMES_PER_REQUEST:
        flash(f"Upload at most {MAX_RESUMES_PER_REQUEST} resumes at a time.", "error")
        return redirect(url_for("index"))

    job_skills = extract_skills(job_desc)
    results = []

    for file in files:
        if not allowed_file(file.filename):
            flash(f"Invalid file type for {file.filename}. Only PDF is allowed.", "error")
            continue

        filename = secure_filename(file.filename)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], f"{timestamp}_{filename}")

        try:
            file.save(filepath)
            resume_text = extract_text(filepath)
            if not resume_text:
                flash(f"Could not read text content from {filename}.", "error")
                continue

            similarity = compute_similarity(job_desc, resume_text)
            resume_skills = extract_skills(resume_text)
            matched, missing = skill_gap(job_skills, resume_skills)
            skill_score = len(matched) / max(len(job_skills), 1)
            score = final_score(similarity, skill_score)
            llm_feedback = generate_llm_feedback(
                resume_text,
                job_desc,
                matched,
                missing,
                resume_skills=resume_skills,
                similarity=similarity,
            )

            data = {
                "name": filename,
                "score": score,
                "similarity": round(similarity * 100, 2),
                "skills": resume_skills,
                "matched": matched,
                "missing": missing,
                "feedback": generate_feedback(matched, missing),
                "llm_feedback": llm_feedback,
                "created_at": datetime.now(timezone.utc),
            }
            get_candidate_collection().insert_one(data)
            data["_id"] = str(data["_id"])
            results.append(data)
        except Exception:
            app.logger.exception("Error processing file %s", filename)
            flash(f"Failed to analyze {filename}. Please try again.", "error")
        finally:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    app.logger.warning("Failed to remove temporary file %s", filepath)

    results.sort(key=lambda item: item["score"], reverse=True)
    return render_template("result.html", results=results)


@app.route("/dashboard")
def dashboard():
    try:
        projection = {
            "name": 1,
            "score": 1,
            "similarity": 1,
            "skills": 1,
            "matched": 1,
            "missing": 1,
            "feedback": 1,
            "llm_feedback": 1,
            "created_at": 1,
        }
        cursor = (
            get_candidate_collection()
            .find({}, projection)
            .sort([("score", -1), ("created_at", -1)])
            .limit(DASHBOARD_LIMIT)
        )
        data = []
        for item in cursor:
            item["_id"] = str(item["_id"])
            data.append(item)
    except Exception:
        app.logger.exception("Database failure while loading dashboard")
        flash("Could not load the candidates dashboard.", "error")
        data = []
    return render_template("result.html", results=data)


@app.route("/delete/<candidate_id>", methods=["POST"])
def delete_candidate(candidate_id):
    try:
        get_candidate_collection().delete_one({"_id": ObjectId(candidate_id)})
        flash("Candidate record deleted successfully.", "success")
    except Exception:
        app.logger.exception("Failed to delete candidate %s", candidate_id)
        flash("Failed to delete candidate record.", "error")
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    config = get_server_config()
    app.run(host=config["host"], port=config["port"], debug=config["debug"])
