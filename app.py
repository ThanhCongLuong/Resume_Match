import streamlit as st
from groq import Groq
import PyPDF2
import io
import json
import streamlit as st

# Cấu hình 
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def extract_text_from_pdf(uploaded_file):
    """Trích xuất văn bản từ file PDF tải lên"""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def analyze_matching(jd_text, cv_text):
    # Bước 1: Làm sạch văn bản, loại bỏ các ký tự surrogate không hợp lệ
    jd_clean = jd_text.encode('utf-8', 'ignore').decode('utf-8')
    cv_clean = cv_text.encode('utf-8', 'ignore').decode('utf-8')

    # Bước 2: Prompt sạch (không dùng emoji bên trong biến gửi lên AI)
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
        # Nếu vẫn lỗi, in ra console để kiểm tra chính xác vị trí
        print(f"Error details: {str(e)}") 
        return {"error": str(e)}
# Giao diện Streamlit
st.set_page_config(page_title="AI CV Matcher", layout="wide")
st.title("🚀 AI CV Matcher - Groq Powered")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Thông tin JD")
    jd_input = st.text_area("Dán mô tả công việc (JD) vào đây:", height=300)

with col2:
    st.subheader("Tải lên CV")
    uploaded_file = st.file_uploader("Chọn file PDF của CV", type=["pdf"])

if st.button("Bắt đầu phân tích"):
    if jd_input and uploaded_file:
        with st.spinner("Đang phân tích..."):
            cv_text = extract_text_from_pdf(uploaded_file)
            result = analyze_matching(jd_input, cv_text)
            
            if "error" in result:
                st.error(f"Lỗi: {result['error']}")
            else:
                st.divider()
                # Hiển thị Score
                score = result.get("score", 0)
                st.metric(label="Mức độ tương thích", value=f"{score}%")
                
                # Hiển thị Tóm tắt CV (Resume Summary)
                st.info(f"👤 **Tóm tắt ứng viên:** {result.get('cv_summary', 'Không có dữ liệu tóm tắt')}")
                
                # Chia cột hiển thị ưu/nhược điểm
                c1, c2 = st.columns(2)
                with c1:
                    st.success("✅ **Điểm mạnh (Match)**")
                    for p in result.get("pros", []):
                        st.write(f"- {p}")
                
                with c2:
                    st.warning("⚠️ **Điểm còn thiếu**")
                    for c in result.get("cons", []):
                        st.write(f"- {c}")
    else:
        st.warning("Vui lòng điền đủ thông tin trước khi nhấn nút.")
