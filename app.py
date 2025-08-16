import streamlit as st
from sentence_transformers import SentenceTransformer, util
import docx2txt
import PyPDF2

job_roles = {
    "Data Scientist": "Python, ML, data analysis, statistics, pandas, numpy",
    "Business Analyst": "Excel, Tableau, business analysis, reporting, stakeholder management",
    "Software Engineer": "Java, Python, software development, coding, algorithms",
    "ML Engineer": "ML models, Python, scikit-learn, model deployment, data analysis",
    "Frontend Developer": "React, HTML, CSS, JavaScript, UI development",
    "Backend Developer": "Node.js, Python, database management, APIs",
    "Full Stack Developer": "React, Node.js, MongoDB, Python, web development",
    "Project Manager": "Team leadership, planning, agile, project coordination",
    "HR Manager": "Recruitment, employee relations, HR policies, communication",
    "Marketing Specialist": "SEO, content marketing, social media, campaigns",
    "Data Analyst": "SQL, Excel, Tableau, data visualization, reporting",
    "UI/UX Designer": "Figma, Adobe XD, prototyping, user interface design",
    "DevOps Engineer": "AWS, CI/CD, Docker, Kubernetes, automation",
    "QA Engineer": "Testing, Selenium, automation, manual testing",
    "Cybersecurity Analyst": "Network security, vulnerability assessment, ethical hacking"
}

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_file(file):
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + " "
        return text
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    else:
        st.error("Only PDF or DOCX files are supported")
        return ""

def match_resume(resume_text, job_roles):
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    job_names = list(job_roles.keys())
    job_descs = list(job_roles.values())
    job_embs = model.encode(job_descs, convert_to_tensor=True)
    cos_scores = util.cos_sim(resume_emb, job_embs)[0]
    best_idx = cos_scores.argmax()
    best_role = job_names[best_idx]
    confidence = float(cos_scores[best_idx]) * 100
    sorted_idx = cos_scores.argsort(descending=True)
    other_roles = [job_names[i] for i in sorted_idx if i != best_idx][:2]
    return best_role, confidence, other_roles

st.title("Resume Job Role Predictor")
uploaded_file = st.file_uploader("Upload your resume (PDF/DOCX)", type=["pdf","docx"])

if uploaded_file is not None:
    resume_text = extract_file(uploaded_file)
    if resume_text:
        best_role, confidence, other_roles = match_resume(resume_text, job_roles)
        st.subheader("✅ Prediction Results")
        st.write(f"**Best Role:** {best_role}")
        st.write(f"**Confidence:** {confidence:.1f}%")
        st.write(f"**Other Possible Roles:** {other_roles}")
