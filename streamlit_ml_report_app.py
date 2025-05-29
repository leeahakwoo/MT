
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from fpdf import FPDF
import tempfile
import os

# Generate formula image
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

st.title("🧠 Model Evaluation with Tabular Summary")

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

def summarize_test_process(y_test, y_pred, TP, FP, FN):
    total_cases = len(y_test)
    pred_pos = sum(y_pred)
    actual_pos = sum(y_test)
    lines = [
        f"Total test cases: {total_cases}",
        f"Predicted Positives: {pred_pos}",
        f"Actual Positives: {actual_pos}",
        f"True Positives (TP): {TP}",
        f"False Positives (FP): {FP}",
        f"False Negatives (FN): {FN}",
        f"Precision = TP / (TP + FP) = {TP} / ({TP + FP})",
        f"Recall = TP / (TP + FN) = {TP} / ({TP + FN})"
    ]
    return lines

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

def generate_pdf(metrics_df, process_lines, explanations, chart_paths, formula_image):
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

    for path in chart_paths:
        pdf.add_page()
        pdf.image(path, w=180)

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

            TP = cm[1][1]
            FP = cm[0][1]
            FN = cm[1][0]
            TN = cm[0][0]

            process_lines = summarize_test_process(y_test, y_pred, TP, FP, FN)
            for line in process_lines:
                st.markdown(f"- {line}")

            metrics_df, precision, recall, f1 = detailed_metric_table(TP, FP, FN)

            explanations = [
                f"- A precision of {precision:.2f} indicates that {precision*100:.1f}% of predicted positives were correct.",
                f"- A recall of {recall:.2f} indicates the model captured {recall*100:.1f}% of actual positives.",
                f"- F1 score combines precision and recall. If either is 0, F1 will also be 0.",
            ]

            st.subheader("📊 Metric Table")
            st.dataframe(metrics_df)

            chart_paths = []
            cm_path = tempfile.mktemp(suffix=".png")
            fig_cm.savefig(cm_path)
            chart_paths.append(cm_path)

            try:
                y_score = model.predict_proba(X_test)[:, 1]
                fig_roc, roc_auc = plot_roc(y_test, y_score)
                st.subheader("📈 ROC Curve")
                st.pyplot(fig_roc)
                roc_path = tempfile.mktemp(suffix=".png")
                fig_roc.savefig(roc_path)
                chart_paths.append(roc_path)
                plt.close(fig_roc)
            except:
                roc_auc = 0.0
                explanations.append("ROC Curve not available.")

            formula_path = generate_formula_image_fixed()
            st.image(formula_path, caption="정밀도, 재현율, F1 수식")

            if st.button("📄 Generate Detailed PDF Report"):
                pdf_path = generate_pdf(
                    metrics_df=metrics_df,
                    process_lines=process_lines,
                    explanations=explanations,
                    chart_paths=chart_paths,
                    formula_image=formula_path
                )
                with open(pdf_path, "rb") as f:
                    st.download_button("📥 Download Report", f, file_name="detailed_evaluation_report.pdf")

    except Exception as e:
        st.error(f"Error during evaluation: {e}")
