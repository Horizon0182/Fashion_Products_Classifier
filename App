import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

st.set_page_config(page_title="Fashion Image Classifier", page_icon="👕")

MODEL_PATH = "Albatrosszzz/Fashion_Clothes_Image_Classifier"

@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return processor, model, device

def predict_topk(image: Image.Image, processor, model, device, k=5):
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {kk: vv.to(device) for kk, vv in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    topk = torch.topk(probs, k)

    results = []
    for idx, score in zip(topk.indices, topk.values):
        results.append(
            (model.config.id2label[idx.item()], float(score.item()))
        )
    return results

st.title("Fashion Product Image Classifier")
st.write("Upload a product image to predict its category.")

processor, model, device = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    results = predict_topk(image, processor, model, device, k=5)

    st.subheader("Top-5 predictions")
    for label, score in results:
        st.write(f"**{label}** — {score:.4f}")
