import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# page config setup
st.set_page_config(page_title="Stroke Risk Classification", layout="centered")

# app title and description
st.title("Stroke Risk Classification")
st.write(
    "This application demonstrates machine learning models "
    "for predicting stroke risk using healthcare data."
)

# load dataset
st.sidebar.header("Input")
st.sidebar.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader")