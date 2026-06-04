# Cat vs Dog Classifier

This project is a simple web app that classifies an uploaded image as either a cat or a dog.

The app was built with Streamlit and uses a trained PyTorch CNN model saved as `model_0.pth`.

## Files

- `app.py` - runs the Streamlit web app
- `model_0.pth` - saved trained model weights
- `catdogclassifier.ipynb` - notebook used to train and save the model

## How It Works

1. The user uploads an image.
2. The image is converted to RGB.
3. The image is resized to `64x64` pixels.
4. The image is converted into a PyTorch tensor.
5. The trained model predicts whether the image is a cat or a dog.
6. The app displays the prediction and confidence score.

## Requirements

Install the required packages:

```bash
pip install streamlit torch torchvision pillow


