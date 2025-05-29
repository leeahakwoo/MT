
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from fpdf import FPDF
import tempfile
import os

st.title("🧠 Model Evaluation with PDF Report and Justification")

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

def generate_pdf(metrics, explanations, matrix_data, chart_paths):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Model Evaluation Report", ln=True)

    for metric, value in metrics.items():
        pdf.cell(0, 10, f"{metric}: {value:.3f}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Justification Table", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, matrix_data)

    pdf.ln(5)
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

            report = classification_report(y_test, y_pred, output_dict=True)
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1 = report['weighted avg']['f1-score']

            st.subheader("📊 Classification Report")
            st.dataframe(pd.DataFrame(report).transpose().round(3))

            fig_cm, cm = plot_confusion_matrix(y_test, y_pred)
            st.subheader("📌 Confusion Matrix")
            st.pyplot(fig_cm)

            TP = cm[1][1]
            FP = cm[0][1]
            FN = cm[1][0]
            TN = cm[0][0]

            matrix_comment = (
                f"TP (True Positive): {TP}\n"
                f"FP (False Positive): {FP}\n"
                f"FN (False Negative): {FN}\n"
                f"TN (True Negative): {TN}\n"
                f"Precision = TP / (TP + FP) = {TP} / ({TP + FP})\n"
                f"Recall = TP / (TP + FN) = {TP} / ({TP + FN})\n"
                f"F1 Score = 2 * (P * R) / (P + R)"
            )

            st.subheader("📘 Explanation & Justification")
            st.text(matrix_comment)

            explanations = [
                f"Precision of {precision:.2f} means that out of all positive predictions, {precision * 100:.1f}% were correct.",
                f"Recall of {recall:.2f} means the model captured {recall * 100:.1f}% of all actual positives.",
                f"F1 Score balances both precision and recall, useful when classes are imbalanced."
            ]

            st.write("📝 Interpretations")
            for exp in explanations:
                st.markdown(f"- {exp}")

            cm_path = tempfile.mktemp(suffix=".png")
            fig_cm.savefig(cm_path)
            plt.close(fig_cm)

            chart_paths = [cm_path]

            try:
                y_score = model.predict_proba(X_test)[:, 1]
                fig_roc, roc_auc = plot_roc(y_test, y_score)
                st.subheader("📈 ROC Curve")
                st.pyplot(fig_roc)

                roc_path = tempfile.mktemp(suffix=".png")
                fig_roc.savefig(roc_path)
                chart_paths.append(roc_path)
                plt.close(fig_roc)
                explanations.append(f"AUC of {roc_auc:.2f} indicates {'excellent' if roc_auc >= 0.9 else 'good' if roc_auc >= 0.8 else 'decent'} classification performance.")
            except:
                roc_auc = 0.0
                explanations.append("ROC Curve not available (model lacks predict_proba).")

            if st.button("📄 Generate PDF Report"):
                pdf_path = generate_pdf(
                    metrics={"Precision": precision, "Recall": recall, "F1 Score": f1, "AUC": roc_auc},
                    explanations=explanations,
                    matrix_data=matrix_comment,
                    chart_paths=chart_paths
                )
                with open(pdf_path, "rb") as f:
                    st.download_button("📥 Download Report", f, file_name="evaluation_report.pdf")

    except Exception as e:
        st.error(f"Error during evaluation: {e}")
