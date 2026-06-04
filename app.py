import streamlit as st
from PIL import Image
import torch
from torch import nn
from torchvision import transforms


class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 13 * 13, output_shape),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return self.classifier(x)


CLASS_NAMES = ["cat", "dog"]
MODEL_PATH = "model_0.pth"


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyVGG(input_shape=3, hidden_units=10, output_shape=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device


transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]
)

st.title("Cat vs Dog Classifier")

uploaded_file = st.file_uploader(
    "Upload an image of a cat or dog",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    model_0, device = load_model()

    with torch.inference_mode():
        img = transform(image).unsqueeze(0).to(device)
        logits = model_0(img)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        predicted_idx = int(probabilities.argmax().item())

    label = CLASS_NAMES[predicted_idx]
    confidence = float(probabilities[predicted_idx].item())

    st.subheader("Prediction")
    st.success(f"{label.title()} ({confidence:.1%} confidence)")
