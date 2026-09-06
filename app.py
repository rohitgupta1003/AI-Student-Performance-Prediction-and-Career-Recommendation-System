"""
AI-Powered Student Performance Prediction & Career Recommendation System
--------------------------------------------------------------------------
Single-file Flask application.

What it does:
1. Generates a synthetic student dataset (replace with a real Kaggle CSV
   in load_or_generate_data() if you want real data instead).
2. Trains a RandomForestRegressor to predict a student's expected score.
3. Uses simple rule-based logic to recommend career paths and skills to
   improve, based on the student's interest area and current skill set.
4. Serves a form (GET /) and a results page (POST /predict) using Flask,
   with all HTML/CSS embedded in this same file (no templates folder
   needed).

Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# 1. DATA + MODEL TRAINING (runs once, when the app starts)
# ---------------------------------------------------------------------------

def load_or_generate_data(n_students=400, seed=42):
    """
    Generates a synthetic dataset of students.
    Swap this out for: pd.read_csv("your_real_dataset.csv") if you find
    a suitable dataset on Kaggle (search "student performance dataset").
    """
    rng = np.random.default_rng(seed)

    previous_marks = rng.integers(35, 100, n_students)
    attendance = rng.integers(50, 100, n_students)
    study_hours = rng.uniform(0, 8, n_students).round(1)
    programming_skill = rng.integers(1, 11, n_students)

    # Final score is a weighted combination of the above + random noise.
    # This mimics a realistic (but fake) relationship for training purposes.
    noise = rng.normal(0, 5, n_students)
    final_score = (
        0.45 * previous_marks
        + 0.25 * attendance
        + 3.0 * study_hours
        + 1.5 * programming_skill
        + noise
    )
    final_score = np.clip(final_score, 0, 100)

    df = pd.DataFrame({
        "previous_marks": previous_marks,
        "attendance": attendance,
        "study_hours": study_hours,
        "programming_skill": programming_skill,
        "final_score": final_score,
    })
    return df


def train_model(df):
    feature_cols = ["previous_marks", "attendance", "study_hours", "programming_skill"]
    X = df[feature_cols]
    y = df["final_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    print(f"Model trained. Train R^2: {train_r2:.3f} | Test R^2: {test_r2:.3f}")

    return model


DATA = load_or_generate_data()
MODEL = train_model(DATA)


def predict_performance(previous_marks, attendance, study_hours, programming_skill):
    X_new = pd.DataFrame([{
        "previous_marks": previous_marks,
        "attendance": attendance,
        "study_hours": study_hours,
        "programming_skill": programming_skill,
    }])
    predicted_score = float(MODEL.predict(X_new)[0])
    predicted_score = round(max(0, min(100, predicted_score)), 1)

    if predicted_score >= 75:
        category = "Good"
    elif predicted_score >= 50:
        category = "Average"
    else:
        category = "Needs Improvement"

    return predicted_score, category


# ---------------------------------------------------------------------------
# 2. RULE-BASED CAREER RECOMMENDATION LOGIC
# ---------------------------------------------------------------------------

CAREER_SKILL_REQUIREMENTS = {
    "Data Analyst": ["Python", "SQL", "Excel", "Statistics", "Data Visualization"],
    "Machine Learning Engineer": ["Python", "Machine Learning", "Statistics", "Data Structures", "SQL"],
    "Data Scientist": ["Python", "Statistics", "Machine Learning", "SQL", "Data Visualization"],
    "Software Developer": ["Data Structures", "OOP (Java/C++)", "Git", "Problem Solving", "System Design"],
    "Web Developer": ["HTML/CSS", "JavaScript", "Python or Node.js", "Databases"],
    "UI/UX Designer": ["Design Tools (Figma)", "HTML/CSS", "User Research", "Prototyping"],
    "Business Analyst": ["Excel", "SQL", "Communication", "Data Visualization", "Domain Knowledge"],
    "Network/Systems Engineer": ["Networking Basics", "Linux", "Security Fundamentals", "Cloud Basics"],
}


def recommend_careers(interest, programming_skill):
    interest = interest.lower()

    if interest in ("coding", "software development"):
        candidates = ["Software Developer", "Web Developer"]
        if programming_skill >= 7:
            candidates.insert(0, "Machine Learning Engineer")
    elif interest in ("data science", "ai/ml"):
        candidates = ["Data Analyst", "Data Scientist"]
        if programming_skill >= 6:
            candidates.insert(0, "Machine Learning Engineer")
    elif interest == "design":
        candidates = ["UI/UX Designer", "Web Developer"]
    elif interest == "business":
        candidates = ["Business Analyst", "Data Analyst"]
    elif interest in ("hardware", "networking"):
        candidates = ["Network/Systems Engineer", "Software Developer"]
    else:
        candidates = ["Software Developer", "Data Analyst"]

    # de-duplicate while preserving order, keep top 3
    seen = set()
    ordered = []
    for c in candidates:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered[:3]


def recommend_skills_to_improve(top_career, known_skills):
    required = CAREER_SKILL_REQUIREMENTS.get(top_career, [])
    known_lower = {s.lower() for s in known_skills}
    missing = [s for s in required if s.lower() not in known_lower]
    return missing if missing else ["You already cover the core skills — focus on building real projects."]


# ---------------------------------------------------------------------------
# 3. FLASK APP + EMBEDDED HTML/CSS TEMPLATES
# ---------------------------------------------------------------------------

app = Flask(__name__)

BASE_STYLE = """
<style>
  body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    margin: 0;
    padding: 40px 20px;
    display: flex;
    justify-content: center;
  }
  .card {
    background: #1e293b;
    border-radius: 14px;
    padding: 32px 36px;
    max-width: 560px;
    width: 100%;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
  }
  h1 { font-size: 22px; margin-bottom: 4px; color: #f8fafc; }
  p.subtitle { color: #94a3b8; margin-top: 0; margin-bottom: 24px; font-size: 14px; }
  label { display: block; margin-top: 16px; margin-bottom: 6px; font-size: 14px; color: #cbd5e1; }
  input[type=number], select {
    width: 100%; padding: 10px; border-radius: 8px; border: 1px solid #334155;
    background: #0f172a; color: #f1f5f9; font-size: 14px; box-sizing: border-box;
  }
  input[type=range] { width: 100%; }
  .checkboxes { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 6px; }
  .checkboxes label {
    display: flex; align-items: center; gap: 6px; background: #0f172a;
    padding: 6px 10px; border-radius: 8px; font-size: 13px; margin: 0; border: 1px solid #334155;
  }
  button {
    margin-top: 26px; width: 100%; padding: 12px; border: none; border-radius: 8px;
    background: #6366f1; color: white; font-size: 15px; font-weight: 600; cursor: pointer;
  }
  button:hover { background: #4f46e5; }
  .result-box {
    background: #0f172a; border-radius: 10px; padding: 18px 20px; margin-top: 18px;
    border: 1px solid #334155;
  }
  .score { font-size: 34px; font-weight: 700; color: #a5b4fc; }
  .tag {
    display: inline-block; background: #16a34a22; color: #4ade80; padding: 4px 10px;
    border-radius: 20px; font-size: 13px; margin: 4px 6px 0 0; border: 1px solid #16a34a55;
  }
  .skill-tag {
    display: inline-block; background: #f59e0b22; color: #fbbf24; padding: 4px 10px;
    border-radius: 20px; font-size: 13px; margin: 4px 6px 0 0; border: 1px solid #f59e0b55;
  }
  a.back { color: #818cf8; font-size: 13px; text-decoration: none; }
</style>
"""

FORM_PAGE = """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Student Performance & Career Predictor</title>""" + BASE_STYLE + """</head>
<body>
  <div class="card">
    <h1>🎓 Student Performance & Career Recommender</h1>
    <p class="subtitle">Fill in your details to get a predicted score, career suggestions, and skill gaps.</p>
    <form action="/predict" method="post">

      <label>Previous semester marks (%)</label>
      <input type="number" name="previous_marks" min="0" max="100" required value="75">

      <label>Attendance (%)</label>
      <input type="number" name="attendance" min="0" max="100" required value="80">

      <label>Average study hours per day</label>
      <input type="number" step="0.1" name="study_hours" min="0" max="12" required value="2.5">

      <label>Programming skill (1 = beginner, 10 = advanced)</label>
      <input type="number" name="programming_skill" min="1" max="10" required value="5">

      <label>Primary area of interest</label>
      <select name="interest">
        <option value="coding">Coding / Software Development</option>
        <option value="data science">Data Science / AI-ML</option>
        <option value="design">Design (UI/UX)</option>
        <option value="business">Business / Analytics</option>
        <option value="hardware">Hardware / Networking</option>
      </select>

      <label>Skills you already know</label>
      <div class="checkboxes">
        <label><input type="checkbox" name="skills" value="Python"> Python</label>
        <label><input type="checkbox" name="skills" value="SQL"> SQL</label>
        <label><input type="checkbox" name="skills" value="HTML/CSS"> HTML/CSS</label>
        <label><input type="checkbox" name="skills" value="JavaScript"> JavaScript</label>
        <label><input type="checkbox" name="skills" value="Data Structures"> Data Structures</label>
        <label><input type="checkbox" name="skills" value="Machine Learning"> Machine Learning</label>
        <label><input type="checkbox" name="skills" value="Statistics"> Statistics</label>
        <label><input type="checkbox" name="skills" value="Git"> Git</label>
      </div>

      <button type="submit">Predict My Performance & Career Path</button>
    </form>
  </div>
</body>
</html>
"""

RESULT_PAGE = """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Your Results</title>""" + BASE_STYLE + """</head>
<body>
  <div class="card">
    <h1>📊 Your Results</h1>
    <p class="subtitle">Based on the information you provided.</p>

    <div class="result-box">
      <div>Predicted Score</div>
      <div class="score">{{ score }}%</div>
      <div>Category: <strong>{{ category }}</strong></div>
    </div>

    <div class="result-box">
      <div>🎯 Recommended Career Paths</div>
      <div>
        {% for c in careers %}
          <span class="tag">{{ c }}</span>
        {% endfor %}
      </div>
    </div>

    <div class="result-box">
      <div>📚 Skills You Should Improve (for {{ careers[0] }})</div>
      <div>
        {% for s in skill_gaps %}
          <span class="skill-tag">{{ s }}</span>
        {% endfor %}
      </div>
    </div>

    <p style="margin-top:20px;"><a class="back" href="/">&larr; Try again with different values</a></p>
  </div>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(FORM_PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    previous_marks = float(request.form["previous_marks"])
    attendance = float(request.form["attendance"])
    study_hours = float(request.form["study_hours"])
    programming_skill = int(request.form["programming_skill"])
    interest = request.form["interest"]
    known_skills = request.form.getlist("skills")

    score, category = predict_performance(
        previous_marks, attendance, study_hours, programming_skill
    )
    careers = recommend_careers(interest, programming_skill)
    skill_gaps = recommend_skills_to_improve(careers[0], known_skills)

    return render_template_string(
        RESULT_PAGE,
        score=score,
        category=category,
        careers=careers,
        skill_gaps=skill_gaps,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)