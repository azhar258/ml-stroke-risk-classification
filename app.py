import streamlit as st
import pandas as pd

from model.train_models import train_all_models

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
    st.subheader("📂 Dataset Preview")
    data = pd.read_csv(uploaded_file)
    st.dataframe(data.head())

    with st.spinner("Training and evaluating models..."):
        results_df, models = train_all_models(uploaded_file)

    st.subheader("📊 Model Comparison")
    st.dataframe(results_df.style.format(precision=4))
else:
    st.info("Please upload a CSV file to train and evaluate the models.")