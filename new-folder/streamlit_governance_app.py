
import streamlit as st
import pandas as pd
from datetime import datetime
from docx import Document
from io import BytesIO

st.set_page_config(page_title="AI 거버넌스 자동 생성기", layout="wide")
st.title("🧪 [디버깅용] AI 거버넌스 문서 생성기")

# --- Sidebar 고급 기능 ---
st.sidebar.header("🛠️ 고급 기능")
risk_suggestion = st.sidebar.checkbox("🔍 리스크 자동 제안")
sensitivity_check = st.sidebar.checkbox("🔒 민감도 분류 지원")
dashboard_preview = st.sidebar.checkbox("📊 대시보드 미리보기")

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

# --- 디버깅 출력 ---
st.subheader("📋 디버그 출력 (입력값)")
st.json({
    "context": context,
    "role": role,
    "stakeholders": stakeholders,
    "needs": needs,
    "data_source": data_source,
    "data_type": data_type,
    "policy_input": policy_input,
    "infrastructure": infrastructure,
    "cto_name": cto_name,
    "tech_team_role": tech_team_role,
    "quality_team_role": quality_team_role
})

# --- 문서 문장 생성 ---
def generate_text():
    try:
        parts = []
        parts.append(f"[조직 환경] {context}")
        parts.append(f"[조직 역할] {role}")
        parts.append(f"[이해관계자] {', '.join(stakeholders)} / 요구: {needs}")
        parts.append(f"[데이터] 출처: {data_source} / 유형: {data_type}")
        parts.append(f"[정책] {policy_input}")
        parts.append(f"[인프라] {infrastructure}")
        parts.append(f"[책임자] CTO: {cto_name}, 기술팀: {tech_team_role}, 품질팀: {quality_team_role}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"[문장 생성 오류] {str(e)}"

# --- 문서 생성 함수 ---
def generate_docx_buffer():
    doc = Document()
    doc.add_paragraph("📌 문서 생성 테스트 시작")  # 초기 문장
    content = generate_text()
    doc.add_paragraph(content)

    buffer = BytesIO()
    try:
        doc.save(buffer)
        buffer.seek(0)
        return buffer, None
    except Exception as e:
        return None, str(e)

# --- 생성 버튼 ---
st.markdown("---")
if st.button("📄 문서 생성하기"):
    buffer, error = generate_docx_buffer()
    if error:
        st.error(f"문서 저장 중 오류 발생: {error}")
    else:
        filename = f"AI_Governance_Debug_{datetime.now().strftime('%Y%m%d_%H%M')}.docx"
        st.download_button(
            label="📥 다운로드 (Word)",
            data=buffer,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
