import streamlit as st

# Configure the page
st.set_page_config(page_title="AI 거버넌스 요약", page_icon="🤖", layout="wide")

# Title for the app
st.title("AI 거버넌스 요약 생성기")

# Optional description/instructions for clarity
st.markdown("""AI 거버넌스 관련 입력 정보를 작성한 후 **요약 보기** 버튼을 누르면, 
입력된 내용을 토대로 섹션별로 정리된 요약이 화면에 표시됩니다.
문서 파일을 생성하거나 다운로드하지 않고 즉시 결과를 확인할 수 있습니다.""")

# Input fields for AI governance context
st.subheader("기본 정보 입력")
org_context = st.text_area("조직 맥락 및 AI 역할", placeholder="조직의 배경과 AI의 역할을 입력하세요.")
stakeholders = st.text_area("이해관계자 및 요구사항", placeholder="이해관계자와 그들의 필요 사항을 입력하세요.")
data_sources = st.text_area("데이터 소스 및 유형", placeholder="사용되는 데이터 출처와 유형을 입력하세요.")
policies_infra = st.text_area("내부 AI 정책 및 인프라", placeholder="내부 AI 관련 정책과 인프라 상황을 입력하세요.")
roles_resp = st.text_area("역할 및 책임 (예: CTO, 기술 및 품질 팀)", placeholder="각 역할 (예: CTO, 기술팀 등)과 책임 범위를 입력하세요.")

# Advanced optional features
st.subheader("고급 기능 (선택 사항)")
risk_checked = st.checkbox("위험 요소 제안")
sensitivity_checked = st.checkbox("민감도 분류")
dashboard_checked = st.checkbox("대시보드 요약 미리보기")

# Button to generate summary
submitted = st.button("요약 보기")

if submitted:
    # Compile the summary content
    summary_lines = []
    # Only add section if content is not empty or at least one field is filled?
    # We'll add all sections regardless, even if empty, but we can handle empty gracefully.
    summary_lines.append("## 조직 맥락 및 AI 역할\n" + (org_context if org_context else "입력된 내용이 없습니다."))
    summary_lines.append("## 이해관계자 및 요구사항\n" + (stakeholders if stakeholders else "입력된 내용이 없습니다."))
    summary_lines.append("## 데이터 소스 및 유형\n" + (data_sources if data_sources else "입력된 내용이 없습니다."))
    summary_lines.append("## 내부 AI 정책 및 인프라\n" + (policies_infra if policies_infra else "입력된 내용이 없습니다."))
    summary_lines.append("## 역할 및 책임\n" + (roles_resp if roles_resp else "입력된 내용이 없습니다."))

    # Advanced sections if their checkboxes are checked
    if risk_checked:
        # Generate a basic risk suggestion content
        risk_content = "AI 활용과 관련하여 고려해야 할 잠재적 위험 요소:\n"
        risk_points = []
        # Provide general risk points (could be refined based on input in the future)
        risk_points.append("- 데이터 프라이버시 및 보안: 개인 정보 및 민감한 데이터의 유출 위험을 관리해야 합니다.")
        risk_points.append("- 편향 및 공정성: AI 모델의 의사결정에서 편향이 발생하지 않도록 주의해야 합니다.")
        risk_points.append("- 투명성 및 설명 가능성: AI의 의사결정 과정을 설명하고 이해관계자에게 투명하게 공개해야 합니다.")
        risk_points.append("- 규정 준수: 업계 표준 및 관련 법규(예: AI 윤리 가이드라인)에 맞게 AI 시스템을 운영해야 합니다.")
        # Join risk points into the section content
        risk_content += "\n".join(risk_points)
        summary_lines.append("## 위험 요소 제안\n" + risk_content)

    if sensitivity_checked:
        # Basic sensitivity classification content
        sens_content = "데이터 소스의 민감도 분류:\n"
        sens_points = []
        data_text = data_sources.lower() if data_sources else ""
        # Check for keywords to determine sensitivity
        high_keywords = ["개인", "고객", "민감", "금융", "의료", "health", "financial", "personal", "customer", "employee"]
        medium_keywords = ["내부", "사내", "기밀", "internal", "confidential"]
        low_keywords = ["공개", "오픈", "open", "public"]
        # Determine if any keywords present
        high_flag = any(k in data_text for k in high_keywords)
        medium_flag = any(k in data_text for k in medium_keywords)
        low_flag = any(k in data_text for k in low_keywords)
        if high_flag:
            sens_points.append("- **높은 민감도:** 개인 식별 정보 또는 민감한 데이터 (예: 고객 개인정보)로 분류됩니다.")
        if medium_flag:
            sens_points.append("- **중간 민감도:** 내부 사용 데이터 또는 기밀 데이터로 분류됩니다.")
        if low_flag:
            sens_points.append("- **낮은 민감도:** 공개 데이터 또는 일반 공개 가능한 정보로 분류됩니다.")
        if not sens_points:
            sens_points.append("- 제공된 정보를 기반으로 민감도를 분류할 추가 단서가 없습니다.")
        sens_content += "\n".join(sens_points)
        summary_lines.append("## 민감도 분류\n" + sens_content)

    if dashboard_checked:
        # Provide a brief dashboard-friendly summary (e.g., key points from each section)
        preview_content = "다음은 입력된 정보를 간략히 요약한 대시보드용 하이라이트입니다:\n"
        preview_points = []
        if org_context:
            preview_points.append(f"- **조직/AI 맥락:** {org_context[:50]}{'...' if len(org_context) > 50 else ''}")
        if stakeholders:
            preview_points.append(f"- **이해관계자:** {stakeholders[:50]}{'...' if len(stakeholders) > 50 else ''}")
        if data_sources:
            preview_points.append(f"- **데이터:** {data_sources[:50]}{'...' if len(data_sources) > 50 else ''}")
        if policies_infra:
            preview_points.append(f"- **정책/인프라:** {policies_infra[:50]}{'...' if len(policies_infra) > 50 else ''}")
        if roles_resp:
            preview_points.append(f"- **주요 역할:** {roles_resp[:50]}{'...' if len(roles_resp) > 50 else ''}")
        if not preview_points:
            preview_points.append("- (요약할 내용이 없습니다)")
        preview_content += "\n".join(preview_points)
        summary_lines.append("## 대시보드 요약 미리보기\n" + preview_content)

    # Display the compiled summary in markdown
    st.markdown("\n\n".join(summary_lines))
