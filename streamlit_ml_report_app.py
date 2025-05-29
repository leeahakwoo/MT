# streamlit_eval_report_with_full_interpretation_and_pdf.py
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

# Page settings
st.set_page_config(page_title="Model Evaluation with PDF Report", layout="wide")
st.title("🧠 Model Evaluation with Interpretations & PDF Report")

# File Upload
uploaded_model = st.file_uploader("Upload your trained model (.pkl or .joblib)", type=["pkl", "joblib"])
uploaded_test_data = st.file_uploader("Upload your test data (.csv with 'target' column)", type="csv")

# PDF Class
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Model Evaluation Report", ln=True, align="C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 10)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def generate_pdf_report(metrics: dict, interpretations: dict):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "Evaluation Metrics", ln=True)
    for metric, value in metrics.items():
        pdf.cell(0, 10, f"{metric}: {value:.3f}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Interpretations", ln=True)
    pdf.set_font("Arial", size=12)
    for metric, explanation in interpretations.items():
        pdf.multi_cell(0, 10, f"{metric}: {explanation}")

    pdf_path = tempfile.mktemp(suffix=".pdf")
    pdf.output(pdf_path)
    return pdf_path

def generate_interpretation(metric_name, value):
    if metric_name == "AUC":
        if value >= 0.9:
            return "Excellent model performance (AUC ≥ 0.9)."
        elif value >= 0.75:
            return "Good discrimination capability (AUC ≥ 0.75)."
        else:
            return "Poor discrimination capability. Consider improving the model."
    elif metric_name == "Precision":
        return f"Precision of {value:.2f} indicates that {value*100:.0f}% of predicted positives are true positives."
    elif metric_name == "Recall":
        return f"Recall of {value:.2f} shows the model captures {value*100:.0f}% of actual positives."
    elif metric_name == "F1 Score":
        return "F1 Score balances precision and recall. Closer to 1 is better."
    return "No interpretation available."

if uploaded_model and uploaded_test_data:
    model = joblib.load(uploaded_model)
    df = pd.read_csv(uploaded_test_data)

    if 'target' not in df.columns:
        st.error("The dataset must include a 'target' column.")
    else:
        X = df.drop(columns=['target'])
        y = df['target']
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else np.zeros_like(y_pred)

        cm = confusion_matrix(y, y_pred)
        TP, FP, FN, TN = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]

        precision = TP / (TP + FP) if TP + FP > 0 else 0.0
        recall = TP / (TP + FN) if TP + FN > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        # Metrics Display
        st.subheader("📊 Evaluation Metrics")
        st.metric("Precision", f"{precision:.2f}")
        st.markdown(f"📝 {generate_interpretation('Precision', precision)}")
        st.metric("Recall", f"{recall:.2f}")
        st.markdown(f"📝 {generate_interpretation('Recall', recall)}")
        st.metric("F1 Score", f"{f1:.2f}")
        st.markdown(f"📝 {generate_interpretation('F1 Score', f1)}")

        st.subheader("🔲 Confusion Matrix")
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig_cm)
        st.markdown("📝 Interpretation: High FN or FP values may indicate class imbalance or poor threshold tuning.")

        st.subheader("📈 ROC Curve")
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc_score = auc(fpr, tpr)
        fig_roc, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        st.pyplot(fig_roc)
        st.markdown(f"📝 Interpretation: {generate_interpretation('AUC', auc_score)}")

        st.subheader("📉 Precision-Recall Curve")
        precision_vals, recall_vals, _ = precision_recall_curve(y, y_prob)
        fig_pr, ax = plt.subplots()
        ax.plot(recall_vals, precision_vals)
        ax.set_title("Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        st.pyplot(fig_pr)
        st.markdown("📝 Interpretation: Observe the balance between precision and recall across thresholds.")

        st.subheader("📊 Prediction Probability Histogram")
        fig_hist, ax = plt.subplots()
        ax.hist(y_prob, bins=20, color='skyblue', edgecolor='black')
        ax.set_title("Histogram of Prediction Probabilities")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Frequency")
        st.pyplot(fig_hist)
        st.markdown("📝 Interpretation: Clustering near 0 or 1 indicates confident predictions; uniform distribution suggests uncertainty.")

        # Feature Importance
        if hasattr(model, "feature_importances_"):
            st.subheader("📌 Feature Importance")
            importances = model.feature_importances_
            fig_fi, ax = plt.subplots()
            sns.barplot(x=importances, y=X.columns, ax=ax)
            ax.set_title("Feature Importances")
            st.pyplot(fig_fi)
            top_feature = X.columns[np.argmax(importances)]
            st.markdown(f"📝 Interpretation: Feature '{top_feature}' contributes most significantly to the model's decision.")

        # PDF Report Generation
        if st.button("📄 Generate PDF Report"):
            metrics = {
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "AUC": auc_score
            }
            interpretations = {
                k: generate_interpretation(k, v) for k, v in metrics.items()
            }
            pdf_path = generate_pdf_report(metrics, interpretations)
            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download PDF Report", f, file_name="evaluation_report.pdf")
