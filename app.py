import streamlit as st
from groq import Groq
import PyPDF2
import io
import json

# Configuration
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def extract_text_from_pdf(uploaded_file):
    """Extract text from the uploaded PDF file"""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def analyze_matching(jd_text, cv_text):
    # Step 1: Clean text, removing invalid surrogate characters
    jd_clean = jd_text.encode('utf-8', 'ignore').decode('utf-8')
    cv_clean = cv_text.encode('utf-8', 'ignore').decode('utf-8')

    # Step 2: Clean prompt (no emojis inside variables sent to AI)
    prompt = f"""
    Analyze the following CV and JD for matching. 
    Note: The 'cv_summary' must be a summary of the candidate's capabilities based on the CV.

    Return a single JSON object:
    {{
        "score": (integer from 0-100),
        "cv_summary": "Short summary 3-4 sentences about hometown (MANDATORY), education, experience, ALL previous workplaces, and strengths.",
        "pros": ["strengths matching the JD"],
        "cons": ["missing skills or gaps compared to the JD"]
    }}
    
    JD Content: {jd_clean[:1500]}
    ---
    CV Content: {cv_clean[:2000]}
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a professional Headhunter. Respond in ENGLISH and ONLY output JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        # If an error occurs, print to console to check exact location
        print(f"Error details: {str(e)}") 
        return {"error": str(e)}

# Streamlit Interface
st.set_page_config(page_title="AI CV Matcher", layout="wide")
st.title("AI CV Matcher - Groq Powered")

col1, col2 = st.columns(2)

with col1:
    st.subheader("JD Information")
    jd_input = st.text_area("Paste the Job Description (JD) here:", height=300)

with col2:
    st.subheader("Upload CV")
    uploaded_file = st.file_uploader("Select CV PDF file", type=["pdf"])

if st.button("Start Analysis"):
    if jd_input and uploaded_file:
        with st.spinner("Analyzing..."):
            cv_text = extract_text_from_pdf(uploaded_file)
            result = analyze_matching(jd_input, cv_text)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.divider()
                
                # Display Score
                score = result.get("score", 0)
                st.metric(label="Matching Score", value=f"{score}%")
                
                # Display Resume Summary
                st.info(f"👤 **Candidate Summary:** {result.get('cv_summary', 'No summary data available')}")
                
                # Split columns for pros/cons
                c1, c2 = st.columns(2)
                with c1:
                    st.success("✅ **Strengths (Match)**")
                    for p in result.get("pros", []):
                        st.write(f"- {p}")
                
                with c2:
                    st.warning("⚠️ **Missing Points / Gaps**")
                    for c in result.get("cons", []):
                        st.write(f"- {c}")
    else:
        st.warning("Please provide all required information before clicking the button.")
