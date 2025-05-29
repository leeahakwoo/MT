# streamlit_eval_report_stable.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np

# Streamlit settings
st.set_page_config(page_title="Model Evaluation (Stable Version)", layout="wide")
st.title("🧠 Stable Model Evaluation")

uploaded_model = st.file_uploader("Upload your trained model (.pkl or .joblib)", type=["pkl", "joblib"])
uploaded_test_data = st.file_uploader("Upload your test data (.csv with 'target' column)", type="csv")

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
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else np.zeros_like(y_pred)

        cm = confusion_matrix(y, y_pred)
        TP, FP, FN, TN = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

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

        if hasattr(model, "feature_importances_"):
            st.subheader("📌 Feature Importance")
            importances = model.feature_importances_
            fig_fi, ax = plt.subplots()
            sns.barplot(x=importances, y=X.columns, ax=ax)
            ax.set_title("Feature Importances")
            st.pyplot(fig_fi)
            top_feature = X.columns[np.argmax(importances)]
            st.markdown(f"📝 Interpretation: Feature '{top_feature}' contributes most significantly to the model's decision.")
