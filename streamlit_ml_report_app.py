
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from fpdf import FPDF
import tempfile
import os

st.title("🧠 저장된 모델 파일 + 테스트 데이터 평가 및 보고서 생성기")

uploaded_model = st.file_uploader("모델 파일 업로드 (.pkl or .joblib)", type=["pkl", "joblib"])
uploaded_test_data = st.file_uploader("테스트 데이터 업로드 (.csv)", type=["csv"])

def generate_pdf_report(metrics, explanations, charts):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="모델 평가 자동 보고서", ln=True, align='C')

    for key, value in metrics.items():
        pdf.cell(200, 10, txt=f"{key}: {value:.3f}", ln=True)

    for explanation in explanations:
        pdf.multi_cell(0, 10, explanation)

    for chart_path in charts:
        pdf.add_page()
        pdf.image(chart_path, w=180)

    tmp_path = tempfile.mktemp(suffix=".pdf")
    pdf.output(tmp_path)
    return tmp_path

if uploaded_model and uploaded_test_data:
    try:
        model = joblib.load(uploaded_model)
        df = pd.read_csv(uploaded_test_data)

        if 'target' not in df.columns:
            st.error("❗ 'target' 컬럼이 포함되어 있어야 평가가 가능합니다.")
        else:
            X_test = df.drop(columns=['target'])
            y_test = df['target']
            y_pred = model.predict(X_test)

            # 성능지표
            report = classification_report(y_test, y_pred, output_dict=True)
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1 = report['weighted avg']['f1-score']
            st.dataframe(pd.DataFrame(report).transpose().round(3))

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig_cm)

            cm_path = tempfile.mktemp(suffix=".png")
            fig_cm.savefig(cm_path)
            plt.close(fig_cm)

            # ROC Curve
            roc_auc = None
            try:
                y_score = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = auc(fpr, tpr)

                fig_roc, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                ax.legend()
                st.pyplot(fig_roc)

                roc_path = tempfile.mktemp(suffix=".png")
                fig_roc.savefig(roc_path)
                plt.close(fig_roc)

            except:
                roc_path = None
                st.warning("ROC Curve는 이진 분류에서만 지원됩니다.")

            # 해석 생성
            explanations = [
                f"정밀도는 {precision:.2f}로, 예측 Positive 중 실제 정답 비율을 나타냅니다.",
                f"재현율은 {recall:.2f}로, 실제 Positive 중 예측 성공 비율을 나타냅니다.",
                f"F1 점수는 정밀도와 재현율의 조화 평균으로, 현재 {f1:.2f}입니다.",
            ]
            if roc_auc:
                explanations.append(f"AUC는 {roc_auc:.2f}로, 분류 성능이 매우 {'우수' if roc_auc >= 0.9 else '양호' if roc_auc >= 0.8 else '보통'}한 수준입니다.")

            # PDF 생성
            if st.button("📄 PDF 보고서 생성"):
                chart_list = [cm_path]
                if roc_path:
                    chart_list.append(roc_path)

                pdf_path = generate_pdf_report(
                    metrics={"정밀도": precision, "재현율": recall, "F1 점수": f1, "AUC": roc_auc or 0.0},
                    explanations=explanations,
                    charts=chart_list
                )
                with open(pdf_path, "rb") as f:
                    st.download_button("📥 PDF 다운로드", data=f, file_name="model_evaluation_report.pdf")

    except Exception as e:
        st.error(f"모델 평가 중 오류 발생: {e}")
