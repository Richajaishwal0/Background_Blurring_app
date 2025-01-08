# Background Blur App

Welcome to the Background Blur App! This Streamlit app allows users to upload an image and apply a blur effect to the background while keeping the subject in focus.

## Features
- **Image Upload**: Upload an image to apply the background blur effect.
- **Adjustable Blur Intensity**: Use a slider to control the intensity of the background blur.
- **Download Processed Image**: Download the final image with the blurred background.
  
## Technologies Used
- **Streamlit**: For building the app interface.
- **Pillow**: For image manipulation and processing.
- **Transformers (Hugging Face)**: For utilizing a pre-trained image segmentation model.
- **Torch**: For running the model on GPU if available.
  
## Requirements

To run the app locally, you'll need to install the following dependencies:

```bash
pip install -r requirements.txt
```

### Requirements List:

- **Streamlit**:
- **Torch**
- **Transformers**
- **Pillow**
- **Numpy**
## Usage

### 1. Clone the repository

```bash
git clone https://github.com/Richajaishwal0/Background_Blurring_app
cd Background-Blurring-app
```

### 2. Run the Streamlit App

```bash
streamlit run blur_images.py
```

This will start a local server, and you can access the app by visiting `http://localhost:8501`.

## Deployment

You can view the deployed app at [Streamlit Cloud Deployment](https://backgroundblurringapp-vbrp2zklj8adv3oaj2fgyd.streamlit.app/).

## Model

This app uses the **SegFormer** model (`mattmdjaga/segformer_b2_clothes`) from Hugging Face for image segmentation, which is used to separate the subject from the background.

### Model Path Configuration:
The model used for background segmentation can be configured as follows:

```python
model_path = "mattmdjaga/segformer_b2_clothes"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The app uses the **SegFormer** model from Hugging Face, which provides excellent segmentation results.
- Special thanks to **Streamlit** for making app development so easy and interactive.
  
## Troubleshooting

- If you encounter issues with image processing, ensure that you have the correct version of Python installed and all the dependencies mentioned in the `requirements.txt`.
- If the model fails to load, verify that the model path is correctly configured and the model is available.

### Key Sections:
1. **Introduction**: Provides a summary of the app’s features.
2. **Technologies Used**: Lists the main libraries and technologies.
3. **Requirements**: Specifies the dependencies needed to run the app locally.
4. **Usage**: Explains how to clone the repo, install dependencies, and run the app.
5. **Deployment**: Includes the link to your deployed Streamlit app.
6. **Model**: Information about the model used for image segmentation.
7. **License & Acknowledgements**: Credits and license info.
8. **Troubleshooting**: Common troubleshooting tips.
