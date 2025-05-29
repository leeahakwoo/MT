
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve
from fpdf import FPDF
import tempfile
import os
import numpy as np

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "", 14)
        self.cell(0, 10, "Model Evaluation Report", ln=True, align="C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "", 10)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

st.set_page_config(page_title="Model Evaluation", layout="wide")
st.title("🧠 Model Evaluation App")

uploaded_model = st.file_uploader("Upload trained model (.pkl, .joblib)", type=["pkl", "joblib"])
uploaded_test_data = st.file_uploader("Upload test dataset (.csv)", type=["csv"])

def save_plot(fig, name="plot"):
    path = tempfile.mktemp(suffix=".png", prefix=name)
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path

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
    return fig

def plot_precision_recall(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    return fig

def plot_probability_histogram(y_score):
    fig, ax = plt.subplots()
    ax.hist(y_score, bins=20, color="skyblue", edgecolor="black")
    ax.set_title("Prediction Probability Histogram")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Frequency")
    return fig

def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=feature_names, ax=ax)
        ax.set_title("Feature Importance")
        return fig
    return None

def draw_formula_image():
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis("off")
    formula = (
        r"$\mathrm{Precision} = \frac{TP}{TP + FP} \quad "
        r"\mathrm{Recall} = \frac{TP}{TP + FN} \quad "
        r"F1 = 2 \cdot \frac{\mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}$"
    )
    ax.text(0.5, 0.5, formula, fontsize=16, ha="center", va="center")
    img_path = "formula_viz.png"
    plt.savefig(img_path, bbox_inches="tight", dpi=200)
    plt.close()
    return img_path

def generate_pdf(metrics, explanations, confusion_text, chart_paths):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", "", 12)

    for key, val in metrics.items():
        pdf.cell(0, 10, f"{key}: {val:.2f}", ln=True)

    pdf.ln(5)
    pdf.multi_cell(0, 8, confusion_text)
    for ex in explanations:
        pdf.multi_cell(0, 8, ex)

    for path in chart_paths:
        pdf.add_page()
        pdf.image(path, w=180)

    report_path = tempfile.mktemp(suffix=".pdf")
    pdf.output(report_path)
    return report_path

if uploaded_model and uploaded_test_data:
    try:
        model = joblib.load(uploaded_model)
        df = pd.read_csv(uploaded_test_data)

        if "target" not in df.columns:
            st.error("⚠️ 'target' column is required.")
        else:
            X_test = df.drop(columns=["target"])
            y_test = df["target"]
            y_pred = model.predict(X_test)
            y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros_like(y_pred)

            TP = ((y_pred == 1) & (y_test == 1)).sum()
            FP = ((y_pred == 1) & (y_test == 0)).sum()
            FN = ((y_pred == 0) & (y_test == 1)).sum()
            TN = ((y_pred == 0) & (y_test == 0)).sum()

            precision = TP / (TP + FP) if TP + FP > 0 else 0.0
            recall = TP / (TP + FN) if TP + FN > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics = {
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            }

            st.subheader("📊 Metrics")
            for k, v in metrics.items():
                st.metric(k, f"{v:.2f}")

            st.subheader("📘 Explanation")
            explanations = [
                f"Precision = TP / (TP + FP) = {TP} / ({TP} + {FP}) = {precision:.2f}",
                f"Recall = TP / (TP + FN) = {TP} / ({TP} + {FN}) = {recall:.2f}",
                f"F1 Score = 2 * P * R / (P + R) = {f1:.2f}"
            ]
            for ex in explanations:
                st.markdown(f"- {ex}")

            chart_paths = []

            fig_cm, _ = plot_confusion_matrix(y_test, y_pred)
            st.pyplot(fig_cm)
            chart_paths.append(save_plot(fig_cm, "confusion"))

            fig_roc = plot_roc(y_test, y_score)
            st.pyplot(fig_roc)
            chart_paths.append(save_plot(fig_roc, "roc"))

            fig_pr = plot_precision_recall(y_test, y_score)
            st.pyplot(fig_pr)
            chart_paths.append(save_plot(fig_pr, "pr"))

            fig_hist = plot_probability_histogram(y_score)
            st.pyplot(fig_hist)
            chart_paths.append(save_plot(fig_hist, "hist"))

            fig_importance = plot_feature_importance(model, X_test.columns)
            if fig_importance:
                st.pyplot(fig_importance)
                chart_paths.append(save_plot(fig_importance, "importance"))

            confusion_text = f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}"
            st.markdown(f"**Confusion Matrix Detail:** {confusion_text}")

            if st.button("📄 Generate Full PDF Report"):
                pdf_path = generate_pdf(metrics, explanations, confusion_text, chart_paths)
                with open(pdf_path, "rb") as f:
                    st.download_button("📥 Download PDF", f, file_name="model_eval_extended.pdf")

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")
