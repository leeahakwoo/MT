
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from fpdf import FPDF
import os
import warnings

warnings.filterwarnings("ignore")

st.title("🧠 ML 모델 성능 평가 + 설명가능성 + 보고서 생성")

uploaded_model = st.file_uploader("모델 업로드 (.pkl)", type=["pkl"])
uploaded_data = st.file_uploader("테스트 데이터 (.csv)", type=["csv"])

if uploaded_model and uploaded_data:
    model = joblib.load(uploaded_model)
    data = pd.read_csv(uploaded_data)

    X = data.iloc[:, :-1]
    y_true = data.iloc[:, -1]
    y_pred = model.predict(X)

    # 성능 평가
    st.subheader("📊 성능 지표")
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format(precision=3))

    # Confusion Matrix
    st.subheader("📌 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    st.pyplot(fig_cm)

    # ROC Curve
    try:
        st.subheader("📈 ROC Curve")
        y_score = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
    except:
        st.warning("ROC Curve는 이진 분류 모델에서만 지원됩니다.")

    # SHAP 설명가능성
    st.subheader("🔍 SHAP 기반 설명가능성 분석")
    with st.spinner("SHAP 계산 중..."):
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

    st.markdown("**📌 피처 중요도 (Summary Plot)**")
    fig_shap = shap.plots.bar(shap_values, show=False)
    st.pyplot(bbox_inches='tight')

    st.markdown("**🔬 개별 예측 설명 (Force Plot)**")
    shap.initjs()
    force_plot_html = shap.plots.force(explainer.expected_value, shap_values[0], X.iloc[0], matplotlib=False)
    st.components.v1.html(shap.getjs() + force_plot_html.html(), height=300)

    # PDF 보고서 생성
    if st.button("📄 PDF 보고서 생성"):
        cm_img_path = "conf_matrix.png"
        fig_cm.savefig(cm_img_path)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt="ML 모델 성능 보고서", ln=True, align="C")
        pdf.ln(10)

        for label, metrics in report.items():
            if isinstance(metrics, dict):
                pdf.cell(200, 10, txt=f"[{label}]", ln=True)
                for metric, value in metrics.items():
                    pdf.cell(200, 10, txt=f"  - {metric}: {value:.4f}", ln=True)
                pdf.ln(2)

        pdf.image(cm_img_path, x=10, w=180)

        pdf_path = "model_report.pdf"
        pdf.output(pdf_path)

        with open(pdf_path, "rb") as f:
            st.download_button("📥 보고서 다운로드", f, file_name="model_report.pdf")

        os.remove(cm_img_path)
        os.remove(pdf_path)
