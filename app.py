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
    prompt = f"""
    Hãy phân tích CV và JD sau để so khớp. 
    Lưu ý: Phần 'summary' phải là tóm tắt về năng lực của ứng viên dựa trên CV, không phải tóm tắt JD.

    Yêu cầu trả về JSON duy nhất:
    {{
        "score": (số nguyên từ 0-100),
        "cv_summary": "Tóm tắt ngắn gọn 3 -4 câu về quê quán(BẮT BUỘC), học vấn, kinh nghiệm, TOÀN BỘ nơi đã từng làm việc và thế mạnh của ứng viên trong CV",
        "pros": ["điểm mạnh phù hợp với JD"],
        "cons": ["điểm còn thiếu hoặc chưa đạt so với JD"]
    }}
    
    Nội dung JD: {jd_text[:1500]}
    ---
    Nội dung CV: {cv_text[:2000]}
    """

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia săn đầu người (Headhunter). Hãy phản hồi bằng TIẾNG ANH và CHỈ xuất file JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
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