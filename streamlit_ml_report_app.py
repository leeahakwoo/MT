
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, silhouette_score
from fpdf import FPDF
import os
import warnings

warnings.filterwarnings("ignore")

st.title("🧠 지도학습 + 비지도학습 통합 ML 리포트 앱")

uploaded_data = st.file_uploader("데이터 업로드 (.csv)", type=["csv"])

if uploaded_data:
    df = pd.read_csv(uploaded_data)

    st.subheader("⚙️ 분석 모드 선택")
    mode = st.radio("분석 모드를 선택하세요:", ["지도학습 (정답 라벨 있음)", "비지도학습 (정답 라벨 없음)"])

    if mode == "지도학습 (정답 라벨 있음)":
        feature_names = df.columns[:-1]
        X = pd.DataFrame(df.iloc[:, :-1].values, columns=feature_names)
        y = df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 1. 성능 평가
        st.subheader("📊 성능 지표")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format(precision=3))

        # 2. Confusion Matrix
        st.subheader("📌 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        st.pyplot(fig_cm)

        # 3. ROC Curve (이진 분류)
        try:
            st.subheader("📈 ROC Curve")
            y_score = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)

            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
        except:
            st.warning("ROC Curve는 이진 분류 모델에서만 지원됩니다.")

        # 4. SHAP 설명가능성
        st.subheader("🔍 SHAP 기반 설명가능성 분석")
        with st.spinner("SHAP 계산 중..."):
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)

        st.markdown("**📌 피처 중요도 (Summary Plot)**")
        fig_shap = shap.plots.bar(shap_values, show=False)
        st.pyplot(bbox_inches='tight')

        st.markdown("**🔬 개별 예측 설명 (Force Plot)**")
        shap.initjs()
        force_plot_html = shap.plots.force(explainer.expected_value, shap_values[0], X_test.iloc[0], matplotlib=False)
        st.components.v1.html(shap.getjs() + force_plot_html.html(), height=300)

    elif mode == "비지도학습 (정답 라벨 없음)":
        X = df.copy()
        if 'target' in X.columns:
            X = X.drop(columns=['target'])

        st.subheader("📊 K-Means 클러스터링")
        n_clusters = st.slider("클러스터 수 선택", min_value=2, max_value=10, value=3)
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(X)

        df['Cluster'] = labels
        st.dataframe(df.head())

        st.subheader("📈 Silhouette Score")
        silhouette = silhouette_score(X, labels)
        st.metric("Silhouette Score", f"{silhouette:.4f}")

        st.subheader("📌 클러스터 시각화 (2D)")
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            df_viz = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
            df_viz["Cluster"] = labels

            fig_cluster, ax_cluster = plt.subplots()
            sns.scatterplot(data=df_viz, x="PC1", y="PC2", hue="Cluster", palette="tab10", ax=ax_cluster)
            st.pyplot(fig_cluster)
        except:
            st.warning("시각화를 위해 PCA 변환을 시도했으나 실패했습니다.")

