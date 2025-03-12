import streamlit as st
import pandas as pd
import joblib

from keras.models import load_model
model = load_model('cnn_model.h5')


# Load the saved model
# model = joblib.load('cnn_model.pkl')

# Streamlit UI
st.set_page_config(page_title='ML Model Deployment', layout='wide')
st.title('ML Model Deployment')
st.markdown("### Upload an Excel or CSV file to get predictions")

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

        # Assuming model expects specific features
        features = data.drop(columns=['target'], errors='ignore')

        if features.empty:
            st.warning("No valid features found for prediction.")
        else:
            # Make predictions
            predictions = model.predict(features)

            # Combine original data with predictions
            results = data.copy()
            results['Prediction'] = predictions

            # Display predictions
            st.write("### Predictions")
            st.dataframe(results)

            # Download predictions
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv',
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")
