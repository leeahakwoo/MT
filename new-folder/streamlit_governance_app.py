
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="AI 거버넌스 문서 미리보기", layout="wide")
st.title("📋 AI 거버넌스 문서 (실시간 화면 출력 전용)")

# --- 입력 수집 ---
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

# --- 문서 스타일 출력 구성 ---
def render_document():
    st.markdown("---")
    st.subheader("📄 AI 거버넌스 문서 미리보기")

    st.markdown("### 1. 조직의 맥락 및 역할")
    st.markdown(f"- 환경 이슈: **{context}**")
    st.markdown(f"- 조직 역할: **{role}**")

    st.markdown("### 2. 이해관계자")
    st.markdown(f"- 대상: **{', '.join(stakeholders)}**")
    st.markdown(f"- 요구사항: **{needs}**")

    st.markdown("### 3. 데이터 정보")
    st.markdown(f"- 출처: **{data_source}**")
    st.markdown(f"- 유형: **{data_type}**")

    st.markdown("### 4. 정책 및 시스템")
    st.markdown(f"- 내부 정책: **{policy_input}**")
    st.markdown(f"- 인프라: **{infrastructure}**")

    st.markdown("### 5. 책임자 및 역할")
    st.markdown(f"- CTO: **{cto_name}**")
    st.markdown(f"- 기술팀: **{tech_team_role}**")
    st.markdown(f"- 품질팀: **{quality_team_role}**")

# --- 출력 버튼 ---
st.markdown("---")
if st.button("📋 화면에 문서 내용 보기"):
    render_document()
