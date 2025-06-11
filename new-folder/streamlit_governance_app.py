import streamlit as st

st.set_page_config(page_title="AI 거버넌스 미리보기", layout="wide")
st.title("🤖 AI 거버넌스 입력 실시간 미리보기")

# --- 입력값 수집 ---
st.subheader("1. 조직의 맥락 및 역할")
org_context = st.text_area("✅ 조직의 외부/내부 환경 이슈:") or ""
org_role = st.selectbox("✅ 조직의 AI 역할:",
                        ["개발자", "제공자", "운영자", "사용자", "복합적 역할"])

st.subheader("2. 이해관계자")
stakeholders = st.multiselect("✅ 주요 이해관계자:",
                              ["고객", "직원", "규제기관", "사회", "협력사"])
stakeholder_needs = st.text_area("✅ 이해관계자의 요구사항:") or ""

st.subheader("3. 데이터 정보")
data_sources = st.text_area("✅ 데이터 출처 및 특성:") or ""
data_type = st.radio("✅ 데이터 유형:",
                     ["정형", "비정형", "민감정보 포함", "공공데이터", "혼합형"],
                     index=0)

st.subheader("4. 내부 정책 및 시스템")
policies_infra = st.text_area("✅ 내부 정책 및 인프라:") or ""

st.subheader("5. 역할과 책임")
cto_name = st.text_input("CTO 이름:") or ""
tech_team = st.text_area("기술팀 역할:") or ""
quality_team = st.text_area("품질팀 역할:") or ""

# --- 고급 기능 ---
st.sidebar.header("🛠️ 고급 기능")
show_risks = st.sidebar.checkbox("🔍 리스크 자동 제안")
show_sensitivity = st.sidebar.checkbox("🔒 민감도 분류 지원")
show_summary = st.sidebar.checkbox("📊 요약 대시보드")

# --- 출력 화면 ---
if st.button("📋 미리보기 보기"):
    st.markdown("## 📄 AI 거버넌스 정보 요약")

    st.markdown("### 1. 조직의 맥락 및 역할")
    st.markdown(f"- 환경 이슈: {org_context if org_context else '없음'}")
    st.markdown(f"- 역할: {org_role}")

    st.markdown("### 2. 이해관계자")
    st.markdown(f"- 이해관계자: {', '.join(stakeholders) if stakeholders else '없음'}")
    st.markdown(f"- 요구사항: {stakeholder_needs if stakeholder_needs else '없음'}")

    st.markdown("### 3. 데이터 정보")
    st.markdown(f"- 출처: {data_sources if data_sources else '없음'}")
    st.markdown(f"- 유형: {data_type}")

    st.markdown("### 4. 정책 및 인프라")
    st.markdown(f"- 내부 정보: {policies_infra if policies_infra else '없음'}")

    st.markdown("### 5. 책임자 및 역할")
    st.markdown(f"- CTO: {cto_name if cto_name else '미입력'}")
    st.markdown(f"- 기술팀: {tech_team if tech_team else '미입력'}")
    st.markdown(f"- 품질팀: {quality_team if quality_team else '미입력'}")

    if show_risks:
        st.markdown("### 🔍 제안된 리스크")
        st.markdown("- 데이터 편향 및 대표성 부족\n- 설명 불가 결과\n- 사용자 오용\n- 법적 위반 가능성")

    if show_sensitivity:
        st.markdown("### 🔒 예상 민감도 결과")
        if data_type and isinstance(data_type, str) and "민감정보" in data_type:
            st.error("❗ 예상 민감도: 높음")
        else:
            st.success("✅ 예상 민감도: 낮음 또는 보통")

    if show_summary:
        st.markdown("### 📊 요약 대시보드")
        st.markdown(f"- 역할: {org_role}\n- 이해관계자 수: {len(stakeholders)}\n- 데이터 유형: {data_type}\n- 리스크 제안: {'활성화' if show_risks else '비활성화'}")
