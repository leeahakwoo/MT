import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from fpdf import FPDF
import tempfile
import os


def is_supervised(data):
    return 'target' in data.columns


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap='Blues')
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig


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


def generate_pdf_report(metrics, explanations, charts):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="모델 품질 평가 보고서", ln=True, align='C')

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


st.title("AI/ML 품질 자동 평가 리포트 앱")

uploaded_file = st.file_uploader("CSV 데이터 파일을 업로드하세요", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("데이터 미리보기", df.head())

    if is_supervised(df):
        st.success("지도학습으로 분류되었습니다.")
        X = df.drop(columns=['target'])
        y = df['target']

        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.metric("정밀도 (Precision)", f"{precision:.3f}")
        st.metric("재현율 (Recall)", f"{recall:.3f}")
        st.metric("F1 점수", f"{f1:.3f}")

        cm_fig = plot_confusion_matrix(y_test, y_pred)
        st.pyplot(cm_fig)

        roc_fig, roc_auc = plot_roc(y_test, y_score)
        st.pyplot(roc_fig)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as cm_tmp:
            cm_fig.savefig(cm_tmp.name)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as roc_tmp:
            roc_fig.savefig(roc_tmp.name)

        explanations = [
            f"정밀도는 {precision:.2f}로, 모델이 예측한 Positive 중 실제로 맞은 비율을 의미합니다.",
            f"재현율은 {recall:.2f}로, 실제 Positive 중에서 모델이 맞춘 비율입니다.",
            f"F1 점수는 정밀도와 재현율의 조화 평균으로 {f1:.2f}입니다.",
            f"AUC는 {roc_auc:.2f}로, 임계값 변화에 따른 분류 성능을 평가합니다."
        ]

        if st.button("PDF 리포트 생성"):
            report_path = generate_pdf_report(
                metrics={"정밀도": precision, "재현율": recall, "F1 점수": f1, "AUC": roc_auc},
                explanations=explanations,
                charts=[cm_tmp.name, roc_tmp.name]
            )
            with open(report_path, "rb") as f:
                st.download_button("PDF 다운로드", data=f, file_name="model_report.pdf")
    else:
        st.warning("비지도학습 데이터로 분류되었습니다. 분석 모듈은 추후 지원 예정입니다.")
