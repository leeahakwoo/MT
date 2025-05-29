
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from fpdf import FPDF
import tempfile
import os

st.title("🧠 Model Evaluation with Test Process Summary")

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
    lines = []
    try:
        precision = TP / (TP + FP)
        lines.append(f"Precision = TP / (TP + FP) = {TP} / ({TP + FP}) = {precision:.3f}")
    except ZeroDivisionError:
        precision = 0
        lines.append("Precision = TP / (TP + FP) = undefined (division by zero) → treated as 0.0")
    try:
        recall = TP / (TP + FN)
        lines.append(f"Recall = TP / (TP + FN) = {TP} / ({TP + FN}) = {recall:.3f}")
    except ZeroDivisionError:
        recall = 0
        lines.append("Recall = TP / (TP + FN) = undefined (division by zero) → treated as 0.0")
    try:
        f1 = 2 * precision * recall / (precision + recall)
        lines.append(f"F1 Score = 2 * P * R / (P + R) = 2 * {precision:.3f} * {recall:.3f} / ({precision:.3f} + {recall:.3f}) = {f1:.3f}")
    except ZeroDivisionError:
        f1 = 0
        lines.append("F1 Score = undefined (division by zero) → treated as 0.0")
    return lines, precision, recall, f1

def generate_pdf(metrics_lines, table_lines, process_lines, explanations, chart_paths):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Model Evaluation Report (Detailed)", ln=True)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Test Process Summary:", ln=True)
    pdf.set_font("Arial", "", 11)
    for line in process_lines:
        pdf.multi_cell(0, 8, line)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Metrics Calculation:", ln=True)
    pdf.set_font("Arial", "", 11)
    for line in metrics_lines:
        pdf.multi_cell(0, 8, line)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Confusion Matrix Breakdown:", ln=True)
    for line in table_lines:
        pdf.multi_cell(0, 8, line)

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

            table_lines = [
                f"TP (True Positives): {TP}",
                f"FP (False Positives): {FP}",
                f"FN (False Negatives): {FN}",
                f"TN (True Negatives): {TN}"
            ]

            process_lines = summarize_test_process(y_test, y_pred, TP, FP, FN)
            for line in process_lines:
                st.markdown(f"- {line}")

            metric_lines, precision, recall, f1 = detailed_metric_table(TP, FP, FN)

            explanations = [
                f"- A precision of {precision:.2f} indicates that {precision*100:.1f}% of predicted positives were correct.",
                f"- A recall of {recall:.2f} indicates the model captured {recall*100:.1f}% of actual positives.",
                f"- F1 score combines precision and recall. If either is 0, F1 will also be 0.",
            ]

            st.subheader("📘 Detailed Metric Explanation")
            for line in metric_lines + table_lines + explanations:
                st.markdown(f"- {line}")

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

            if st.button("📄 Generate Detailed PDF Report"):
                pdf_path = generate_pdf(
                    metrics_lines=metric_lines,
                    table_lines=table_lines,
                    process_lines=process_lines,
                    explanations=explanations,
                    chart_paths=chart_paths
                )
                with open(pdf_path, "rb") as f:
                    st.download_button("📥 Download Report", f, file_name="detailed_evaluation_report.pdf")

    except Exception as e:
        st.error(f"Error during evaluation: {e}")
