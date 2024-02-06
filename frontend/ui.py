from io import StringIO
import pandas as pd
import requests
# from requests_toolbelt.multipart.encoder import MultipartEncoder
import base64
import streamlit as st

# interact with FastAPI endpoint
backend = "http://host.docker.internal:8000/predict"


# construct UI layout
st.title("santander Transaction Prediction Service")

st.write(
    """Obtain a transaction prediction for each particular entity present in the CSV file. Upload the CSV"""
)  # description and instructions

input_csv = st.file_uploader("Upload CSV", type="csv")  # csv upload widget

if st.button("Get Transaction Prediction"):
    if input_csv:
        content = input_csv.getvalue()
        files = {'file': ('input.csv', content)}
        api_response = requests.post(backend, files=files)
        # Display the FastAPI response
        if api_response.status_code == 200:
            content = api_response.content.decode('utf-8')
            df = pd.read_csv(StringIO(content))
            st.write("Prediction Response:")
            if st.button('Download CSV'):
                csv = df.to_csv(index=False).encode('utf-8')
                b64 = base64.b64encode(csv).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="output.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
        else:
            st.write("Error: Failed to call the FastAPI endpoint")
    else:
        # handle case with no CSV
        st.write("Insert an CSV!")