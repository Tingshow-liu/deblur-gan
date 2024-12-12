import streamlit as st
import requests
from PIL import Image, ImageOps
import io
import os

# Streamlit App Configuration
st.title("DeblurGAN Application")
st.write("Upload an image to deblur it using a trained DeblurGAN model.")

# API URL for Inference
API_URL = os.getenv("API_URL", "http://deblurgan-backend:8000/deblur/")

# Image Upload Section
st.subheader("Upload Image")
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)

    # Correct the image orientation based on EXIF metadata
    try:
        image = ImageOps.exif_transpose(image)
    except AttributeError:
        pass

    st.image(image, caption="Uploaded Image", use_container_width=True)

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

                # Load the image from the response
                deblurred_image = Image.open(io.BytesIO(response.content))
                st.success("Image deblurred successfully!")
                st.image(
                    deblurred_image, caption="Deblurred Image", use_container_width=True
                )

                # Provide a download link
                buffered_output = io.BytesIO()
                deblurred_image.save(buffered_output, format="PNG")
                buffered_output.seek(0)
                st.download_button(
                    label="Download Deblurred Image",
                    data=buffered_output,
                    file_name="deblurred_image.png",
                    mime="image/png",
                )

            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {e}")
