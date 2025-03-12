import streamlit as st
import pandas as pd

# Streamlit UI
st.set_page_config(page_title='ML Model Deployment', layout='wide')
st.title('ML Model Deployment')
st.markdown("### Upload an Excel or CSV file")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.write("### Data Preview")
        st.dataframe(data)

    except Exception as e:
        st.error(f"An error occurred: {e}")
