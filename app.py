import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
import PyPDF2
import docx
import os
import json
import re
from groq import Groq
import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt", quiet=True)

STOPWORDS = {
    "the","is","are","was","were","a","an","of","to","in","and","for",
    "on","with","as","by","at","from","that","this","it"
}

def fallback_quiz_from_text(text, num_questions=5):
    sentences = sent_tokenize(text)
    questions = []

    # Select definition-like sentences
    candidates = [
        s for s in sentences
        if len(s.split()) > 8 and (" is " in s.lower() or " are " in s.lower())
    ]

    random.shuffle(candidates)

    for sent in candidates:
        words = word_tokenize(sent)
        keywords = [
            w for w in words
            if w.isalpha() and w.lower() not in STOPWORDS and len(w) > 3
        ]

        if len(keywords) < 4:
            continue

        answer = random.choice(keywords)
        question_text = sent.replace(answer, "_____")

        distractors = random.sample(
            [w for w in keywords if w != answer],
            min(3, len(keywords)-1)
        )

        options = distractors + [answer]
        random.shuffle(options)

        questions.append({
            "q": question_text,
            "options": options,
            "correct": options.index(answer),
            "explanation": f"'{answer}' appears in the uploaded document."
        })

        if len(questions) >= num_questions:
            break

    return questions



# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="SMART QUIZ AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================
# DARK THEME
# ======================
st.markdown("""
<style>
.stApp { background-color: #0e1117; color: white; }
h1, h2, h3 { color: #00ff99; }
.card {
    background-color: #161b22;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #30363d;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ======================
# FILE PROCESSING
# ======================
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return " ".join(page.extract_text() or "" for page in reader.pages)

def extract_text_from_txt(file):
    return file.read().decode("utf-8", errors="ignore")

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def process_uploaded_file(uploaded_file):
    uploaded_file.seek(0)
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(BytesIO(uploaded_file.read()))
    elif uploaded_file.type == "text/plain":
        return extract_text_from_txt(BytesIO(uploaded_file.read()))
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(BytesIO(uploaded_file.read()))
    return ""

# ======================
# GROQ AI GENERATORS
# ======================
def parse_ai_json(text):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return json.loads(match.group()) if match else []

def generate_ai_questions(topic, difficulty, num_questions):
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Missing Groq API key")

        client = Groq(api_key=api_key)

        prompt = f"""
Generate {num_questions} MCQs on "{topic}"
Difficulty: {difficulty}

Return ONLY a JSON array like:
[
  {{
    "q": "Question",
    "options": ["A", "B", "C", "D"],
    "correct": 0,
    "explanation": "Why this is correct"
  }}
]
"""

        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            timeout=20
        )

        if not res or not res.choices:
            raise ValueError("Empty response from Groq")

        text = res.choices[0].message.content
        if not text:
            raise ValueError("Groq returned empty text")

        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            raise ValueError("JSON not found in Groq response")

        return json.loads(match.group())

    except Exception as e:
        st.warning("Groq unavailable. Switching to offline mode.")
        return []

    
def generate_file_questions(file_text, difficulty, num_questions):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"""
    Generate {num_questions} multiple choice question
    ONLY from the content below.

CONTENT:
\"\"\"{file_text[:20000]}\"\"\"


Rules:
- 4 options per question
- Correct answer must be accurate
- Provide explanation

Return ONLY JSON:

[
  {{
    "q": "Question?",
    "options": ["A","B","C","D"],
    "correct": 1,
    "explanation": "Why this is correct"
  }}
]
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    text = res.choices[0].message.content
    match = re.search(r"\[.*\]", text, re.DOTALL)
    return json.loads(match.group())

# ======================
# SESSION STATE
# ======================
for k, v in {
    "quiz_active": False,
    "questions": [],
    "current": 0,
    "score": 0,
    "answers": [],
    "completed": False,
    "file_text": None
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================
# HEADER
# ======================
st.markdown("""
<h1 style="text-align:center;">🧠 SMART QUIZ AI</h1>
<p style="text-align:center;color:#9da5b4;">
AI & File Based Quiz Generator
</p>
""", unsafe_allow_html=True)

# ======================
# TABS
# ======================
tab1, tab2, tab3, tab4 = st.tabs(
    ["⚡ GENERATE", "📁 FILE UPLOAD", "🎮 QUIZ", "🏆 RESULTS"]
)

# ======================
# TAB 1: GENERATE
# ======================
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    quiz_mode = st.radio(
        "Quiz Source",
        ["AI Topic Based", "File Based"]
    )

    topic = st.selectbox(
        "Topic",
        ["AI", "Machine Learning", "Data Science", "Python", "Cyber Security"]
    )

    difficulty = st.select_slider(
        "Difficulty",
        ["Beginner", "Intermediate", "Advanced"]
    )

    num_questions = st.slider("Questions", 5, 15, 5)

if st.button("🚀 Generate AI Quiz"):
    with st.spinner("Generating quiz..."):
        questions = generate_ai_questions(topic, difficulty, num_questions)

    # 🔁 FALLBACK TO FILE
    if not questions:
        if "file_text" in st.session_state and st.session_state.file_text:
            questions = fallback_quiz_from_text(
                st.session_state.file_text[:5000],
                num_questions
            )
            st.info("Generated quiz from uploaded file.")
        else:
            st.error("AI unavailable. Please upload a file.")

    if questions:
        st.session_state.questions = questions
        st.session_state.quiz_active = True
        st.session_state.completed = False
        st.session_state.current = 0
        st.session_state.score = 0
        st.session_state.answers = []
        st.session_state.selected_answer = None
        st.success("Quiz ready! Go to QUIZ tab 🎯")


    st.markdown('</div>', unsafe_allow_html=True)

# ======================
# TAB 2: FILE UPLOAD
# ======================
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF / TXT / DOCX",
        type=["pdf", "txt", "docx"]
    )

    if uploaded_file:
        text = process_uploaded_file(uploaded_file)
        st.session_state.file_text = text
        st.success(f"Extracted {len(text)} characters")
        st.text_area("Preview", text[:2000], height=200, disabled=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ======================
# TAB 3: QUIZ
# ======================
with tab3:
    if st.session_state.quiz_active:
        q = st.session_state.questions[st.session_state.current]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### Question {st.session_state.current + 1}")
        st.write(q["q"])

        selected = st.radio(
            "Choose answer",
            q["options"],
            key=f"q_{st.session_state.current}"
        )

        if st.button("Next"):
            correct = selected == q["options"][q["correct"]]
            if correct:
                st.session_state.score += 1

            st.session_state.answers.append({
                "question": q["q"],
                "selected": selected,
                "correct": correct,
                "explanation": q["explanation"]
            })

            st.session_state.current += 1
            if st.session_state.current >= len(st.session_state.questions):
                st.session_state.quiz_active = False
                st.session_state.completed = True
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No active quiz")

# ======================
# TAB 4: RESULTS
# ======================
with tab4:
    if st.session_state.completed:
        total = len(st.session_state.answers)
        score = st.session_state.score
        percent = score / total * 100

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Score", f"{score}/{total}")
        st.metric("Accuracy", f"{percent:.1f}%")

        # 📊 Pie chart
        df = pd.DataFrame({
            "Result": ["Correct", "Incorrect"],
            "Count": [score, total - score]
        })

        fig = px.pie(
            df,
            values="Count",
            names="Result",
            color="Result",
            color_discrete_map={
                "Correct": "#00ff99",
                "Incorrect": "#ff4b4b"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📘 Answer Review")

        # 🔍 SHOW EXPLANATIONS
        for i, a in enumerate(st.session_state.answers, start=1):
            st.markdown(f"### Question {i}")
            st.write(a["question"])

            st.write(f"🧑 Your Answer: **{a['selected']}**")

            if a["correct"]:
                st.success("✅ Correct")
            else:
                st.error("❌ Incorrect")

            st.info(f"📘 Explanation: {a['explanation']}")

            st.markdown("---")

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Finish quiz to see results")
