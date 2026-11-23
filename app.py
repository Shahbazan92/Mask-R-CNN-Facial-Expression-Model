from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "resnet18_expression_model.pth"
DEVICE = torch.device("cpu")

st.set_page_config(page_title="Facial Expression Recognition", page_icon="🙂", layout="centered")
st.title("Facial Expression Recognition")
st.write("Upload a face image to classify its facial expression.")

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file is missing: {MODEL_PATH}. "
            "Train the model first and upload the checkpoint to the repository."
        )

    # weights_only=True is supported by the pinned PyTorch version.
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 224))
    mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("std", [0.229, 0.224, 0.225])

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    preprocessing = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return model, class_names, preprocessing


uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        model, class_names, preprocessing = load_model()
        input_tensor = preprocessing(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            probabilities = torch.softmax(model(input_tensor), dim=1)[0]

        predicted_index = int(torch.argmax(probabilities).item())
        st.subheader(f"Prediction: {class_names[predicted_index]}")
        st.metric("Confidence", f"{probabilities[predicted_index].item() * 100:.2f}%")

        st.write("Class probabilities")
        probability_data = {
            class_names[i]: float(probabilities[i].item())
            for i in range(len(class_names))
        }
        st.bar_chart(probability_data)
    except FileNotFoundError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Could not process this image: {error}")
else:
    st.info("Upload an image to get a prediction.")

st.caption("This is an image-level ResNet18 classifier, not a Mask R-CNN segmentation model.")
