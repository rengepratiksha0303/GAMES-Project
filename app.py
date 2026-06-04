import streamlit as st
import pandas as pd
import pickle

st.title("🎮 Game Sales Prediction")

# Load model
with open("knn_model (1).pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.dataframe(df.head())

    if st.button("Predict"):

        try:

            X = scaler.transform(df)

            prediction = model.predict(X)

            df["Predicted_Sales"] = prediction

            st.success("Prediction Completed")

            st.dataframe(df.head())

            csv = df.to_csv(index=False)

            st.download_button(
                "Download Result",
                csv,
                "prediction.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(e)
