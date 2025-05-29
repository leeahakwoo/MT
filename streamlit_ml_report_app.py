# streamlit_eval_report_with_graphs.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import tempfile
import os
import numpy as np
from fpdf import FPDF

# Streamlit UI Setup
st.set_page_config(page_title="Model Evaluation with PDF", layout="wide")
st.title("🧠 Model Evaluation with Full Report")

uploaded_model = st.file_uploader("Upload your trained model (.pkl or .joblib)", type=["pkl", "joblib"])
uploaded_test_data = st.file_uploader("Upload your test data (.csv with 'target' column)", type="csv")

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 14)
        self.cell(0, 10, "Model Evaluation Report", ln=True, align="C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", 'I', 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def save_plot(fig):
    path = tempfile.mktemp(suffix=".png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def generate_pdf(metrics, interpretations, cm, chart_paths):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", '', 12)

    pdf.cell(0, 10, "Evaluation Metrics:", ln=True)
    for name, value in metrics.items():
        pdf.cell(0, 10, f"{name}: {value:.3f}", ln=True)

    pdf.ln(5)
    pdf.cell(0, 10, "Interpretations:", ln=True)
    for interp in interpretations:
        pdf.multi_cell(0, 8, interp)

    pdf.ln(5)
    pdf.cell(0, 10, "Confusion Matrix:", ln=True)
    for row in cm:
        pdf.cell(0, 10, ' '.join(map(str, row)), ln=True)

    for path in chart_paths:
        pdf.add_page()
        pdf.image(path, w=180)

    path = tempfile.mktemp(suffix=".pdf")
    pdf.output(path)
    return path

def interpret_metric(name, value):
    if name == "AUC":
        return f"AUC of {value:.2f} implies {'excellent' if value >= 0.9 else 'good' if value >= 0.75 else 'poor'} model discrimination."
    elif name == "Precision":
        return f"Precision of {value:.2f} means {value*100:.0f}% of positive predictions were correct."
    elif name == "Recall":
        return f"Recall of {value:.2f} shows the model captured {value*100:.0f}% of all actual positives."
    elif name == "F1 Score":
        return "F1 Score reflects the balance between precision and recall."
    return ""

if uploaded_model and uploaded_test_data:
    model = joblib.load(uploaded_model)
    df = pd.read_csv(uploaded_test_data)

    if 'target' not in df.columns:
        st.error("The dataset must include a 'target' column.")
    else:
        X = df.drop(columns=['target'])
        y = df['target']
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else np.zeros_like(y_pred)

        cm = confusion_matrix(y, y_pred)
        TP, FP, FN, TN = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        st.subheader("📊 Metrics")
        st.metric("Precision", f"{precision:.2f}")
        st.metric("Recall", f"{recall:.2f}")
        st.metric("F1 Score", f"{f1:.2f}")

        interpretations = [
            interpret_metric("Precision", precision),
            interpret_metric("Recall", recall),
            interpret_metric("F1 Score", f1)
        ]

        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig_cm)
        cm_path = save_plot(fig_cm)

        fig_roc, ax = plt.subplots()
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"AUC = {auc_val:.2f}")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig_roc)
        roc_path = save_plot(fig_roc)
        interpretations.append(interpret_metric("AUC", auc_val))

        fig_pr, ax = plt.subplots()
        precision_vals, recall_vals, _ = precision_recall_curve(y, y_prob)
        ax.plot(recall_vals, precision_vals)
        ax.set_title("Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        st.pyplot(fig_pr)
        pr_path = save_plot(fig_pr)

        fig_hist, ax = plt.subplots()
        ax.hist(y_prob, bins=20, color='skyblue', edgecolor='black')
        ax.set_title("Prediction Probability Histogram")
        st.pyplot(fig_hist)
        hist_path = save_plot(fig_hist)

        chart_paths = [cm_path, roc_path, pr_path, hist_path]

        if st.button("📄 Generate PDF Report"):
            pdf_path = generate_pdf(
                metrics={"Precision": precision, "Recall": recall, "F1 Score": f1, "AUC": auc_val},
                interpretations=interpretations,
                cm=cm,
                chart_paths=chart_paths
            )
            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download Report", f, file_name="evaluation_report.pdf")
