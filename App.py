import streamlit as st
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    BlipProcessor,
    BlipForConditionalGeneration,
)

st.set_page_config(page_title="Fashion Products Classifier", page_icon="👕")

# 分类模型
CLASSIFIER_MODEL_PATH = "Albatrosszzz/Fashion_Image_Classifier_Subcategory"

# image-to-text 模型
CAPTION_MODEL_PATH = "Salesforce/blip-image-captioning-base"


@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 分类模型
    classifier_processor = AutoImageProcessor.from_pretrained(CLASSIFIER_MODEL_PATH)
    classifier_model = AutoModelForImageClassification.from_pretrained(CLASSIFIER_MODEL_PATH)
    classifier_model.to(device)
    classifier_model.eval()

    # 描述生成模型
    caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL_PATH)
    caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_PATH)
    caption_model.to(device)
    caption_model.eval()

    return (
        classifier_processor,
        classifier_model,
        caption_processor,
        caption_model,
        device,
    )


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


def generate_description(image: Image.Image, subcategory: str, caption_processor, caption_model, device):
    prompt = (
        f"a short e-commerce product description of a {subcategory.lower()} fashion item"
    )

    inputs = caption_processor(
        images=image.convert("RGB"),
        text=prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = caption_model.generate(
            **inputs,
            max_new_tokens=30
        )

    description = caption_processor.decode(output_ids[0], skip_special_tokens=True)
    return description


st.title("Fashion Product Image Classifier")
st.write("Upload a product image to predict its subcategory and generate a short product description.")

classifier_processor, classifier_model, caption_processor, caption_model, device = load_models()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    results = predict_topk(image, classifier_processor, classifier_model, device, k=5)

    subcategory_pred, subcategory_score = results[0]

    description = generate_description(
        image=image,
        subcategory=subcategory_pred,
        caption_processor=caption_processor,
        caption_model=caption_model,
        device=device,
    )

    st.subheader("Predicted Subcategory")
    st.write(f"**{subcategory_pred}** ({subcategory_score:.4f})")

    st.subheader("Top-5 predictions")
    for label, score in results:
        st.write(f"**{label}** — {score:.4f}")

    st.subheader("Generated Product Description")
    st.write(description)
