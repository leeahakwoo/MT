
import streamlit as st

st.set_page_config(page_title="AI 거버넌스 실시간 미리보기", layout="wide")
st.title("🤖 AI 거버넌스 입력 정보 미리보기")

st.subheader("1. 조직의 맥락 및 역할")
org_context = st.text_area("✅ 조직의 외부/내부 환경 이슈를 기술해주세요:") or ""
org_role = st.selectbox("✅ 조직의 AI 역할을 선택해주세요:", 
                        ["개발자", "제공자", "운영자", "사용자", "복합적 역할"]) or ""

st.subheader("2. 이해관계자")
stakeholders = st.multiselect("✅ 주요 이해관계자를 선택하세요:", 
                              ["고객", "직원", "규제기관", "사회", "협력사"])
stakeholder_needs = st.text_area("✅ 이해관계자의 요구사항을 간략히 서술해주세요:") or ""

st.subheader("3. 데이터 정보")
data_sources = st.text_area("✅ 데이터 출처 및 특성을 기술해주세요:") or ""
data_type = st.radio("✅ 데이터 유형을 선택하세요:", 
                     ["정형", "비정형", "민감정보 포함", "공공데이터", "혼합형"], index=0) or ""

st.subheader("4. 내부 정책 및 시스템")
policies_infra = st.text_area("✅ 내부 AI 정책 및 인프라 정보를 입력해주세요:") or ""

st.subheader("5. 역할과 책임")
cto_name = st.text_input("CTO (총괄 책임자) 이름:") or ""
tech_team = st.text_area("기술팀 역할 및 책임:") or ""
quality_team = st.text_area("품질팀 역할 및 책임:") or ""

# 고급 기능
st.sidebar.header("🛠️ 고급 기능")
show_risks = st.sidebar.checkbox("🔍 리스크 자동 제안")
show_sensitivity = st.sidebar.checkbox("🔒 민감도 분류 지원")
show_summary = st.sidebar.checkbox("📊 요약 대시보드")

# 실행 버튼
if st.button("📋 화면에 문서 내용 보기"):
    st.markdown("## 📄 AI 거버넌스 문서 (미리보기)")

    st.markdown("### 1. 조직의 맥락 및 역할")
    st.markdown(f"- 환경 이슈: {org_context}")
    st.markdown(f"- 조직 역할: {org_role}")

    st.markdown("### 2. 이해관계자")
    st.markdown(f"- 이해관계자: {', '.join(stakeholders) if stakeholders else '없음'}")
    st.markdown(f"- 요구사항: {stakeholder_needs}")

    st.markdown("### 3. 데이터 정보")
    st.markdown(f"- 출처: {data_sources}")
    st.markdown(f"- 유형: {data_type}")

    st.markdown("### 4. 정책 및 인프라")
    st.markdown(f"- 정책 및 인프라: {policies_infra}")

    st.markdown("### 5. 역할과 책임")
    st.markdown(f"- CTO: {cto_name}")
    st.markdown(f"- 기술팀: {tech_team}")
    st.markdown(f"- 품질팀: {quality_team}")

    if show_risks:
        st.markdown("### 🔍 제안된 리스크")
        st.markdown("- 데이터 편향 및 대표성 부족
- 자동화 오류 또는 설명 불가능한 결과
- 사용자 오용 가능성
- 법/규제 위반 가능성")

    if show_sensitivity:
        st.markdown("### 🔒 예상 민감도 결과")
        if "민감정보" in data_type:
            st.error("❗ 예상 민감도: 높음")
        else:
            st.success("✅ 예상 민감도: 낮음 또는 보통")

    if show_summary:
        st.markdown("### 📊 요약")
        st.markdown(f"- 조직 역할: {org_role}")
        st.markdown(f"- 이해관계자 수: {len(stakeholders)}")
        st.markdown(f"- 데이터 유형: {data_type}")
        st.markdown(f"- 리스크 제안: {'활성화됨' if show_risks else '비활성화'}")
