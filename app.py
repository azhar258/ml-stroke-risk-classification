import streamlit as st
import pandas as pd

from model.train_models import train_all_models
from model.evaluation import evaluate_model


# page config
st.set_page_config(
    page_title="Stroke Risk Classification",
    layout="wide"
)

# title and description
st.title("Stroke Risk Classification")
st.write(
    "This application demonstrates multiple machine learning models "
    "for predicting stroke risk using healthcare data."
)

# sidebar
st.sidebar.header("Dataset Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload test dataset (CSV)",
    type=["csv"]
)

# main logic
if uploaded_file is not None:
    st.subheader(" Dataset Preview")
    data = pd.read_csv(uploaded_file)
    st.dataframe(data.head())

    with st.spinner("Training and evaluating models..."):
        results_df, models = train_all_models(data)

    st.subheader("Model Comparison")
    st.dataframe(results_df.style.format(precision=4))

    # Model evaluations
    st.subheader(" Model Evaluation Details")
    model_names = results_df["Model"].tolist()
    selected_model_name = st.selectbox(
        "Select a model to view detailed metrics",
        model_names
    )

    selected_row = results_df[results_df["Model"] == selected_model_name].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{selected_row['Accuracy']:.4f}")
    col2.metric("Precision", f"{selected_row['Precision']:.4f}")
    col3.metric("Recall", f"{selected_row['Recall']:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", f"{selected_row['F1']:.4f}")
    col5.metric("AUC", f"{selected_row['AUC']:.4f}")
    col6.metric("MCC", f"{selected_row['MCC']:.4f}")
else:
    st.info("Please upload a CSV file to train and evaluate the models.")