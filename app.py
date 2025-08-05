import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px



# Load trained IsolationForest model, scaler, and PCA
model = joblib.load('models/isolation_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')
pca = joblib.load('models/pca_model.pkl')

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("🔍 Credit Card Fraud Detection Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your credit card transaction CSV file", type=["csv"], key="file_uploader")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Prepare features for prediction (numeric only, exclude target)
    target_col = 'is_fraud'
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    X = data[numeric_cols]
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    preds = model.predict(X_pca)
    data['Prediction'] = ['Fraudulent' if x == -1 else 'Not Fraudulent' for x in preds]

    # Show filter option
    if st.checkbox("🔍 Show only fraudulent transactions"):
        data = data[data['Prediction'] == 'Fraudulent']

    st.success("✅ Analysis complete. Below is the prediction summary:")

    # Display the first few rows
    st.dataframe(data.head(100), use_container_width=True)

    # Download button
    st.download_button("📥 Download Results CSV", data.to_csv(index=False).encode('utf-8'), "predictions.csv", "text/csv")

    # Summary stats
    fraud_count = data['Prediction'].value_counts().get('Fraudulent', 0)
    total = len(data)
    fraud_pct = (fraud_count / total) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{total:,}")
    col2.metric("Fraudulent Transactions", f"{fraud_count:,}")
    col3.metric("Fraud Rate", f"{fraud_pct:.2f}%")

    st.markdown("---")

    # Pie Chart
    fig1 = px.pie(
        data_frame=data,
        names='Prediction',
        title='📊 Distribution of Fraudulent vs Non-Fraudulent Transactions',
        color_discrete_map={'Fraudulent': 'red', 'Not Fraudulent': 'green'}
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Histogram of Amount
    st.subheader("💰 Transaction Amount Distribution")
    if 'amt' in data.columns:
        fig2 = px.histogram(
            data, x='amt', color='Prediction',
            nbins=50,
            title='Transaction Amount by Fraud Status',
            color_discrete_map={'Fraudulent': 'red', 'Not Fraudulent': 'green'},
            marginal="box"
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No 'amt' column found for transaction amount distribution plot.")

    # Line Plot over Time
    st.subheader("⏱️ Fraudulent Transactions Over Time")
    if 'Time' in data.columns:
        time_df = data[data['Prediction'] == 'Fraudulent'][['Time']]
        if not time_df.empty:
            fig3 = px.histogram(time_df, x='Time', nbins=50, title="Fraudulent Transaction Count Over Time")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No fraudulent transactions found for time-series plot.")

    # Evaluation metrics if ground truth available
    y_true = None
    if 'is_fraud' in data.columns:
        y_true = data['is_fraud'].map({1: 'Fraudulent', 0: 'Not Fraudulent'})
    elif 'Class' in data.columns:
        y_true = data['Class'].map({1: 'Fraudulent', 0: 'Not Fraudulent'})
    if y_true is not None:
        st.subheader("📊 Evaluation Metrics")
        st.text("Classification Report:")
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        st.text(classification_report(y_true, data['Prediction']))
        st.text("Confusion Matrix:")
        cm = confusion_matrix(y_true, data['Prediction'])
        cm_df = pd.DataFrame(cm, index=['Actual Not Fraudulent', 'Actual Fraudulent'], columns=['Predicted Not Fraudulent', 'Predicted Fraudulent'])
        st.write(cm_df)
        # Seaborn confusion matrix plot
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        plt.title('Confusion Matrix (Seaborn)')
        st.pyplot(fig_cm)
        try:
            auc_score = roc_auc_score((y_true == 'Fraudulent').astype(int), (data['Prediction'] == 'Fraudulent').astype(int))
            st.metric("ROC AUC Score", f"{auc_score:.4f}")
        except Exception as e:
            st.warning(f"Could not compute ROC AUC Score: {e}")

    # Explanation
    with st.expander("ℹ️ How are transactions marked as fraud?"):
        st.markdown("""
    The model uses **Isolation Forest**, an unsupervised anomaly detection algorithm.

    - It isolates observations by randomly selecting features and splitting values.
    - Anomalies require fewer splits and are thus easier to isolate.
    - Based on this logic, transactions far from the normal pattern are marked as **fraudulent**.

    ⚙️ Preprocessing includes:
    - **Standard Scaling** of features
    - **PCA** (Principal Component Analysis) to reduce dimensionality

    **Note**: The model was trained on imbalanced real-world data using **SMOTE** to balance it.
    """)