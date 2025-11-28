import streamlit as st
import requests
from PIL import Image

SERVER = st.text_input("Inference server URL", value="http://localhost:5000")
st.title("Plant Disease Detector - Demo UI")

uploaded = st.file_uploader("Upload leaf image", type=['jpg','jpeg','png'])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption='Uploaded image', use_column_width=True)

    if st.button('Send for prediction'):

        # Correct multipart upload
        files = {
            "file": (
                uploaded.name,
                uploaded.getvalue(),
                uploaded.type   # "image/jpeg" etc.
            )
        }

        try:
            r = requests.post(
                SERVER.rstrip('/') + "/predict",
                files=files
            )

            if r.status_code == 200:
                resp = r.json()
                st.write("**Predicted:**", resp["predicted_class"])
                st.json(resp["probabilities"])
            else:
                st.error(f"Server responded: {r.status_code} - {r.text}")

        except Exception as e:
            st.error(f"Request error: {e}")
