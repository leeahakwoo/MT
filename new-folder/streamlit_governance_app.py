
import streamlit as st
from fpdf import FPDF
from io import BytesIO
from datetime import datetime

st.set_page_config(page_title="AI 거버넌스 PDF 생성기", layout="wide")
st.title("📄 AI 거버넌스 보고서 생성기 (PDF 포맷)")

# --- 입력값 ---
st.subheader("1. 조직의 맥락 및 역할")
context = st.text_area("✅ 외부/내부 환경 이슈", placeholder="예: 인력 부족")
role = st.selectbox("✅ 조직의 AI 역할", ["개발자", "제공자", "운영자", "사용자", "복합적 역할"])

st.subheader("2. 이해관계자")
stakeholders = st.multiselect("✅ 주요 이해관계자", ["고객", "직원", "규제기관", "사회", "협력사"])
needs = st.text_area("✅ 이해관계자 요구사항")

st.subheader("3. 데이터 정보")
data_source = st.text_area("✅ 데이터 출처")
data_type = st.radio("✅ 데이터 유형", ["정형", "비정형", "민감정보 포함", "공공데이터", "혼합형"])

st.subheader("4. 정책 및 시스템")
policy_input = st.text_area("✅ 내부 정책")
infrastructure = st.text_area("✅ 인프라")

st.subheader("5. 책임자 및 역할")
cto_name = st.text_input("✅ CTO 이름")
tech_team_role = st.text_area("✅ 기술팀 역할")
quality_team_role = st.text_area("✅ 품질팀 역할")

# --- 보고서 PDF 생성 ---
def generate_pdf():
    content = f"""
[조직의 맥락 및 역할]
- 환경 이슈: {context}
- 조직 역할: {role}

[이해관계자]
- 대상: {', '.join(stakeholders)}
- 요구사항: {needs}

[데이터 정보]
- 출처: {data_source}
- 유형: {data_type}

[정책 및 시스템]
- 내부 정책: {policy_input}
- 인프라: {infrastructure}

[책임자 및 역할]
- CTO: {cto_name}
- 기술팀: {tech_team_role}
- 품질팀: {quality_team_role}
"""

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in content.strip().split("\n"):
        pdf.multi_cell(0, 10, line)
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

st.markdown("---")
if st.button("📥 PDF 문서 생성 및 다운로드"):
    pdf_buffer = generate_pdf()
    filename = f"AI_Governance_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    st.download_button("📄 다운로드 (PDF)", data=pdf_buffer, file_name=filename, mime="application/pdf")
