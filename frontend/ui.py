import io

import requests
# from requests_toolbelt.multipart.encoder import MultipartEncoder

import streamlit as st

# interact with FastAPI endpoint
backend = "http://fastapi:8000/predict"


# construct UI layout
st.title("santander Transaction Prediction Service")

st.write(
    """Obtain a transaction prediction for each particular entity present in the CSV file. Upload the CSV"""
)  # description and instructions

input_csv = st.file_uploader("Upload CSV", type="csv")  # csv upload widget

if st.button("Get Transaction Prediction"):
    if input_csv:
        api_response = requests.post(backend, files={"file": input_csv})
        # Display the FastAPI response
        if api_response.status_code == 200:
            api_data = api_response.json()
            st.write("FastAPI Response:")
            st.write(api_data)
        else:
            st.write("Error: Failed to call the FastAPI endpoint")
    else:
        # handle case with no CSV
        st.write("Insert an CSV!")