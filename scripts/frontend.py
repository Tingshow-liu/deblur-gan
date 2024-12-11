import streamlit as st
import requests
from PIL import Image
import io

# Streamlit App Configuration
st.title("DeblurGAN Application")
st.write("Upload an image to deblur it using a trained DeblurGAN model.")

# API URL for Inference
API_URL = "http://127.0.0.1:8000/deblur/"


# # Backend Health Check
# st.subheader("Backend Health Check")
# try:
#     health_response = requests.get(API_URL.replace("/deblur/", ""))
#     if health_response.status_code == 200:
#         st.success("Backend is running and healthy!")
#     else:
#         st.warning(f"Backend returned a non-OK status: {health_response.status_code}")
# except requests.exceptions.ConnectionError:
#     st.error("Failed to connect to the backend API. Please check if it's running.")


# Image Upload Section
st.subheader("Upload Image")
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Deblur"):
        with st.spinner("Processing..."):
            try:
                # Convert the uploaded image to bytes
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                image_bytes = buffered.getvalue()

                # Send the request to the backend
                files = {"file": ("uploaded_image.png", image_bytes, "image/png")}
                response = requests.post(API_URL, files=files)
                response.raise_for_status()

                # Parse the response and display the deblurred image
                result = response.json()
                deblurred_image_path = result["output_path"]

                st.success(f"Image deblurred successfully and saved at {deblurred_image_path}!")
                deblurred_image = Image.open(deblurred_image_path)
                st.image(deblurred_image, caption="Deblurred Image", use_column_width=True)

            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {e}")