import streamlit as st
import numpy as np
from PIL import Image
import cv2
from generator import generator_model 

# Loading GAN model
model=generator_model()
model.load_weights(r"weights\generator.h5")

             
def deblur(model, input, original_size):
    image = np.array(input)
    image_resized = cv2.resize(image, (256, 256))  # Resize image to match model input size
    image_resized = np.expand_dims(image_resized, axis=0)  # Add batch dimension
    image_resized = (image_resized / 127.5) - 1  # Normalize image to [-1, 1]
    deblurred = model.predict(image_resized)
    deblurred = (deblurred + 1) * 127.5  # De-normalize image to [0, 255]
    deblurred = deblurred.astype(np.uint8)
    deblurred = np.squeeze(deblurred, axis=0)  # Remove batch dimension
    deblurred_resized = cv2.resize(deblurred, original_size)  # Resize back to original size
    return deblurred_resized


def main():
    st.title("Image Deblurring Application")
    st.write("Upload an image to deblur it using a GAN model.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        original_size = image.size  # Save the original size
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Deblurring...")
        
        deblurred_image = deblur(model, image, original_size)
        st.image(deblurred_image, caption='Deblurred Image.', use_column_width=True)

if __name__ == "__main__":
    main()
