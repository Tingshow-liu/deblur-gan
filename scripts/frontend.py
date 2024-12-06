import streamlit as st
import requests
from PIL import Image
import base64
import io

# Streamlit App Configuration
st.title("DCGAN Image Generation")
st.write("Generate images using a trained DCGAN model.")

# API URL for Inference
API_URL = "http://127.0.0.1:8000/generate/"

# Backend Health Check
st.subheader("Backend Health Check")
try:
    health_response = requests.get(API_URL.replace("/generate/", ""))
    if health_response.status_code == 200:
        st.success("Backend is running and healthy!")
    else:
        st.warning(f"Backend returned a non-OK status: {health_response.status_code}")
except requests.exceptions.ConnectionError:
    st.error("Failed to connect to the backend API. Please check if it's running.")

# Image Generation Section
st.subheader("Generate Images")

# Number of images to generate
num_images = st.slider("Number of images to generate", 1, 10, 1)
seed = st.number_input(
    "Seed for reproducibility (optional)", value=None, step=1, format="%d"
)

if st.button("Generate"):
    with st.spinner("Generating images..."):
        try:
            # Request payload
            payload = {"num_images": num_images}
            if seed:
                payload["seed"] = int(seed)

            # Call the backend API
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()

            # Parse response and decode images
            results = response.json()
            st.success(f"Generated {len(results['images'])} images successfully!")
            for idx, img_data in enumerate(results["images"]):
                img_bytes = base64.b64decode(img_data)
                image = Image.open(io.BytesIO(img_bytes))
                st.image(
                    image,
                    caption=f"Generated Image {idx + 1}",
                    use_container_width=True,  # Updated here
                )

        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")
