
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seab as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from fpdf import FPDF
import tempfile
import os

def generate_formula_image_fixed():
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis("off")
    formula_text = (
        r"$\mathrm{Precision} = \frac{TP}{TP + FP} \quad "
        r"\mathrm{Recall} = \frac{TP}{TP + FN} \quad "
        r"F1 = 2 \cdot \frac{\mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}$"
    )
    ax.text(0.5, 0.5, formula_text, fontsize=16, ha="center", va="center")
    formula_path = "formula_fixed.png"
    plt.savefig(formula_path, bbox_inches="tight", dpi=200)
    plt.close()
    return formula_path

st.title("🧠 Model Evaluation with All Upgrades (1-4)")

uploaded_model = st.file_uploader("Upload trained model (.pkl or .joblib)", type=["pkl", "joblib"])
uploaded_test_data = st.file_uploader("Upload test data (.csv)", type=["csv"])

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
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    return fig, roc_auc

def detailed_metric_table(TP, FP, FN):
    try:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        precision = recall = f1 = 0
    table_data = {
        "항목": ["Precision", "Recall", "F1 Score"],
        "값": [round(precision, 3), round(recall, 3), round(f1, 3)],
        "계산식": [
            f"{TP} / ({TP} + {FP})",
            f"{TP} / ({TP} + {FN})",
            f"2 * {round(precision, 3)} * {round(recall, 3)} / ({round(precision, 3)} + {round(recall, 3)})"
        ]
    }
    return pd.DataFrame(table_data), precision, recall, f1

def generate_pdf_table(df, pdf):
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Metric Summary Table:", ln=True)
    pdf.set_font("Arial", "", 11)
    col_width = pdf.w / 3.5
    row_height = 8
    pdf.cell(col_width, row_height, "항목", border=1)
    pdf.cell(col_width, row_height, "값", border=1)
    pdf.cell(col_width*2, row_height, "계산식", border=1)
    pdf.ln(row_height)
    for index, row in df.iterrows():
        pdf.cell(col_width, row_height, str(row["항목"]), border=1)
        pdf.cell(col_width, row_height, str(row["값"]), border=1)
        pdf.cell(col_width*2, row_height, str(row["계산식"]), border=1)
        pdf.ln(row_height)

def generate_pdf(metrics_df, process_lines, explanations, chart_paths, formula_image, viz_explanations):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Model Evaluation Report (Detailed)", ln=True)

    if os.path.exists(formula_image):
        pdf.image(formula_image, w=180)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Test Process Summary:", ln=True)
    pdf.set_font("Arial", "", 11)
    for line in process_lines:
        pdf.multi_cell(0, 8, line)

    generate_pdf_table(metrics_df, pdf)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Interpretation:", ln=True)
    pdf.set_font("Arial", "", 11)
    for explanation in explanations:
        pdf.multi_cell(0, 8, explanation)

    for path, desc in chart_paths:
        pdf.add_page()
        pdf.image(path, w=180)
        pdf.ln(2)
        pdf.set_font("Arial", "I", 10)
        pdf.multi_cell(0, 8, desc)

    temp_path = tempfile.mktemp(suffix=".pdf")
    pdf.output(temp_path)
    return temp_path

if uploaded_model and uploaded_test_data:
    try:
        model = joblib.load(uploaded_model)
        df = pd.read_csv(uploaded_test_data)

        if 'target' not in df.columns:
            st.error("'target' column is required in the test dataset.")
        else:
            X_test = df.drop(columns=['target'])
            y_test = df['target']
            y_pred = model.predict(X_test)

            fig_cm, cm = plot_confusion_matrix(y_test, y_pred)
            st.subheader("📌 Confusion Matrix")
            st.pyplot(fig_cm)
            st.markdown("*🔍 대각선 값이 높을수록 올바른 예측 비율이 높습니다. FP와 FN은 오예측입니다.*")

            TP = cm[1][1]
            FP = cm[0][1]
            FN = cm[1][0]
            TN = cm[0][0]

            process_lines = [
                f"TP = {TP}, FP = {FP}, FN = {FN}, TN = {TN}"
            ]
            metrics_df, precision, recall, f1 = detailed_metric_table(TP, FP, FN)

            explanations = [
                f"Precision: {precision:.2f} → 예측이 맞을 확률",
                f"Recall: {recall:.2f} → 실제 정답을 놓치지 않을 확률",
                f"F1 Score: {f1:.2f} → Precision과 Recall의 조화 평균"
            ]

            st.subheader("📊 Metric Table")
            st.dataframe(metrics_df)

            chart_paths = []
            cm_path = tempfile.mktemp(suffix=".png")
            fig_cm.savefig(cm_path)
            chart_paths.append((cm_path, "🔍 Confusion Matrix는 예측 vs 실제를 비교하는 표입니다."))

            try:
                y_score = model.predict_proba(X_test)[:, 1]
                fig_roc, roc_auc = plot_roc(y_test, y_score)
                st.subheader("📈 ROC Curve")
                st.pyplot(fig_roc)
                st.markdown(f"*🎯 AUC = {roc_auc:.2f}. 높을수록 구분 능력이 좋습니다.*")
                roc_path = tempfile.mktemp(suffix=".png")
                fig_roc.savefig(roc_path)
                chart_paths.append((roc_path, f"🎯 ROC Curve: AUC = {roc_auc:.2f}. 높은 AUC는 우수한 분류기를 의미합니다."))
                plt.close(fig_roc)
            except Exception as e:
                chart_paths.append(("", f"[!] ROC Curve 예외 발생: {e}"))

            formula_path = generate_formula_image_fixed()
            st.image(formula_path, caption="정밀도, 재현율, F1 수식")

            # Explainability (fallback without SHAP)
            st.subheader("🧠 예측 설명 대안")
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feat_df = pd.DataFrame({
                    "Feature": X_test.columns,
                    "Importance": importances
                }).sort_values(by="Importance", ascending=False)
                st.bar_chart(feat_df.set_index("Feature"))
                st.markdown("*이 모델은 SHAP 없이 feature_importances_로 예측 근거를 제공합니다.*")
            else:
                st.markdown("이 모델은 feature importance 정보를 제공하지 않습니다.")

            if st.button("📄 Generate Detailed PDF Report"):
                pdf_path = generate_pdf(
                    metrics_df=metrics_df,
                    process_lines=process_lines,
                    explanations=explanations,
                    chart_paths=chart_paths,
                    formula_image=formula_path,
                    viz_explanations=[desc for _, desc in chart_paths if desc]
                )
                with open(pdf_path, "rb") as f:
                    st.download_button("📥 Download Report", f, file_name="detailed_evaluation_report.pdf")

    except Exception as e:
        st.error(f"[ERROR] 모델 평가 중 문제가 발생했습니다: {e}")
