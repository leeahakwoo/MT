
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from fpdf import FPDF
import tempfile
import os

class PDF(FPDF):
    def header(self):
        self.set_font("Nanum", "", 14)
        self.cell(0, 10, "모델 성능 평가 보고서", ln=True, align="C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Nanum", "", 10)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

st.set_page_config(page_title="모델 성능 평가", layout="wide")
st.title("🧠 모델 성능 평가 (한글 PDF 포함)")

uploaded_model = st.file_uploader("1️⃣ 훈련된 모델 업로드 (.pkl, .joblib)", type=["pkl", "joblib"])
uploaded_test_data = st.file_uploader("2️⃣ 테스트 데이터 업로드 (.csv)", type=["csv"])

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    return fig, cm

def plot_roc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    return fig, roc_auc

def draw_formula_image():
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis("off")
    formula = (
        r"$\mathrm{Precision} = \frac{TP}{TP + FP} \quad "
        r"\mathrm{Recall} = \frac{TP}{TP + FN} \quad "
        r"F1 = 2 \cdot \frac{\mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}$"
    )
    ax.text(0.5, 0.5, formula, fontsize=16, ha="center", va="center")
    img_path = "formula_fpdf2.png"
    plt.savefig(img_path, bbox_inches="tight", dpi=200)
    plt.close()
    return img_path

def generate_pdf(precision, recall, f1, TP, FP, FN, formula_img_path, explanations, confusion_text):
    pdf = PDF()
    pdf.add_page()
    pdf.add_font("Nanum", "", "NanumGothic.ttf", uni=True)
    pdf.set_font("Nanum", "", 12)

    pdf.cell(0, 10, f"정밀도(Precision): {precision:.2f}", ln=True)
    pdf.cell(0, 10, f"재현율(Recall): {recall:.2f}", ln=True)
    pdf.cell(0, 10, f"F1 점수: {f1:.2f}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 8, f"[수치 근거] TP: {TP}, FP: {FP}, FN: {FN}")
    for ex in explanations:
        pdf.multi_cell(0, 8, ex)

    pdf.ln(5)
    pdf.set_font("Nanum", "B", 12)
    pdf.cell(0, 10, "[Confusion Matrix 해석]", ln=True)
    pdf.set_font("Nanum", "", 11)
    pdf.multi_cell(0, 8, confusion_text)

    if os.path.exists(formula_img_path):
        pdf.image(formula_img_path, w=180)

    path = tempfile.mktemp(suffix=".pdf")
    pdf.output(path)
    return path

if uploaded_model and uploaded_test_data:
    try:
        model = joblib.load(uploaded_model)
        df = pd.read_csv(uploaded_test_data)

        if "target" not in df.columns:
            st.error("⚠️ 'target' 컬럼이 존재해야 합니다.")
        else:
            X_test = df.drop(columns=["target"])
            y_test = df["target"]
            y_pred = model.predict(X_test)

            TP = ((y_pred == 1) & (y_test == 1)).sum()
            FP = ((y_pred == 1) & (y_test == 0)).sum()
            FN = ((y_pred == 0) & (y_test == 1)).sum()
            TN = ((y_pred == 0) & (y_test == 0)).sum()

            precision = TP / (TP + FP) if TP + FP > 0 else 0.0
            recall = TP / (TP + FN) if TP + FN > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            st.subheader("📊 평가지표")
            col1, col2, col3 = st.columns(3)
            col1.metric("정밀도 (Precision)", f"{precision:.2f}")
            col2.metric("재현율 (Recall)", f"{recall:.2f}")
            col3.metric("F1 점수", f"{f1:.2f}")

            explanations = [
                f"- 정밀도는 TP / (TP + FP) = {TP} / ({TP} + {FP}) = {precision:.2f}",
                f"- 재현율은 TP / (TP + FN) = {TP} / ({TP} + {FN}) = {recall:.2f}",
                f"- F1 점수는 2 * P * R / (P + R) = {f1:.2f}",
            ]

            st.subheader("📘 수식 해설")
            for ex in explanations:
                st.markdown(f"✅ {ex}")

            st.subheader("🧮 Confusion Matrix")
            fig, cm = plot_confusion_matrix(y_test, y_pred)
            st.pyplot(fig)

            confusion_text = f"TP: {TP} (정답 1, 예측 1), FP: {FP} (정답 0, 예측 1), FN: {FN} (정답 1, 예측 0), TN: {TN} (정답 0, 예측 0)"
            st.info(confusion_text)

            try:
                y_prob = model.predict_proba(X_test)[:, 1]
                st.subheader("📈 ROC Curve")
                fig_roc, _ = plot_roc(y_test, y_prob)
                st.pyplot(fig_roc)
            except:
                st.warning("ROC Curve 생성을 위해 predict_proba가 필요합니다.")

            # Latex 공식 Streamlit에서도 보여주기
            st.subheader("📐 수식 보기 (Latex)")
            st.latex(r"\mathrm{Precision} = rac{TP}{TP + FP}")
            st.latex(r"\mathrm{Recall} = rac{TP}{TP + FN}")
            st.latex(r"F1 = 2 \cdot rac{\mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}")

            formula_img = draw_formula_image()

            if st.button("📄 한글 PDF 보고서 생성"):
                pdf_path = generate_pdf(precision, recall, f1, TP, FP, FN, formula_img, explanations, confusion_text)
                with open(pdf_path, "rb") as f:
                    st.download_button("📥 보고서 다운로드", f, file_name="model_eval_report_korean_v2.pdf")

    except Exception as e:
        st.error(f"⚠️ 오류 발생: {e}")
