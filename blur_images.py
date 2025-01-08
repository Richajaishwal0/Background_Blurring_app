import streamlit as st
from PIL import Image, ImageOps, ImageFilter
from transformers import pipeline
import torch
from typing import Tuple, Optional
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackgroundBlur:
    def __init__(self, model_path: str):
        # Initialize the BackgroundBlur class with model path
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()

    def load_model(self) -> None:
        # Load the segmentation model
        try:
            self.model = pipeline("image-segmentation", model=self.model_path, device=self.device)
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            st.error("Failed to load the segmentation model. Please check the model path.")

    def refine_mask(self, mask: Image.Image) -> Image.Image:
        # Convert the mask to numpy array
        mask_array = np.array(mask)
        
        # Apply the threshold to make the mask more decisive
        threshold = 128
        mask_array = np.where(mask_array > threshold, 255, 0).astype(np.uint8)
        
        # Convert back to PIL image
        refined_mask = Image.fromarray(mask_array)
        
        # Apply a slight blur to anti-alias edges
        refined_mask = refined_mask.filter(ImageFilter.GaussianBlur(radius=0.5))
        return refined_mask

    def process_image(self, image: Image.Image, blur_level: int) -> Tuple[Image.Image, Optional[str]]:
        # Process the image and apply background blur with a clean foreground
        try:
            # Get segmentation mask
            result = self.model(images=image)
            mask = result[0]['mask']
            
            # Refine the mask
            refined_mask = self.refine_mask(mask)

            # Create inverted mask for the foreground
            mask_inverted = ImageOps.invert(refined_mask)

            # Create blurred background
            background=image.copy()
            background = background.filter(ImageFilter.GaussianBlur(radius=blur_level))

            # Combine foreground and background using alpha compositing
            final_image = Image.composite(image, background, mask_inverted)
            
            return final_image, None
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return None, str(e)

    @staticmethod
    def main():
        # Set up Streamlit configuration
        st.set_page_config(page_title="Background Blur App", layout="wide")
        
        # Add custom CSS for better UI
        st.markdown("""
                    <style>
                    .stButton>button{
                    width:100%;
                    }
                    .stImageP{
                    display: flex;
                    justify-content:center;
                    }
                    </style>
                    """, unsafe_allow_html=True)

        # Initialize session state for processing image
        if 'processed_image' not in st.session_state:
            st.session_state.processed_image = None

        # App title and description
        st.title("Welcome to Background Blur App")
        st.markdown("Upload an image to blur its background while keeping the subject in focus.")
        
        # Model path configuration
        model_path = "mattmdjaga/segformer_b2_clothes"
        # Initialize the background blur processor
        processor = BackgroundBlur(model_path)

        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Image")
            uploaded_image = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])
            if uploaded_image:
                original_image = Image.open(uploaded_image)
                st.image(original_image, caption="Original Image", use_column_width=True)

        with col2:
            st.subheader("Processed Image")
            if uploaded_image:
                blur_level = st.slider("Blur Intensity", min_value=0, max_value=30, value=15, step=1, help="Adjust the intensity of the background blur")
                process_button = st.button("Process Image")

                if process_button or st.session_state.processed_image is None:
                    with st.spinner("Processing Image..."):
                        final_image, error = processor.process_image(
                            original_image,
                            blur_level)
                        if error:
                            st.error(f"Error processing image: {error}")
                        else:
                            st.session_state.processed_image = final_image
                            st.image(final_image, caption="Processed Image", use_column_width=True)

                            # Add download button
                            if final_image:
                                # Convert the image to bytes
                                import io
                                buf = io.BytesIO()
                                final_image.save(buf, format="PNG")
                                byte_im = buf.getvalue()
                                st.download_button(
                                    label="Download processed image",
                                    data=byte_im,
                                    file_name="processed_image.png",
                                    mime="image/png"
                                )

if __name__ == "__main__":
    BackgroundBlur.main()
