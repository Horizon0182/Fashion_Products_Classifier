import streamlit as st
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)

st.set_page_config(page_title="Fashion Products Classifier", page_icon="👕")

# =========================
# Model paths
# =========================
CLASSIFIER_MODEL_PATH = "Albatrosszzz/Fashion_Clothes_Image_Classifier"
CAPTION_MODEL_PATH = "Salesforce/blip-image-captioning-base"
LLM_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"


# =========================
# Load models
# =========================
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Image classification model (subcategory)
    classifier_processor = AutoImageProcessor.from_pretrained(CLASSIFIER_MODEL_PATH)
    classifier_model = AutoModelForImageClassification.from_pretrained(CLASSIFIER_MODEL_PATH)
    classifier_model.to(device)
    classifier_model.eval()

    # 2) Image caption model
    caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL_PATH)
    caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_PATH)
    caption_model.to(device)
    caption_model.eval()

    # 3) LLM for product description generation
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
    )
    llm_model.eval()

    return {
        "device": device,
        "classifier_processor": classifier_processor,
        "classifier_model": classifier_model,
        "caption_processor": caption_processor,
        "caption_model": caption_model,
        "llm_tokenizer": llm_tokenizer,
        "llm_model": llm_model,
    }


# =========================
# Prediction helpers
# =========================
def predict_topk(image: Image.Image, processor, model, device, k=5):
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {kk: vv.to(device) for kk, vv in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    topk = torch.topk(probs, k)

    results = []
    for idx, score in zip(topk.indices, topk.values):
        results.append((model.config.id2label[idx.item()], float(score.item())))
    return results


def generate_caption(image: Image.Image, caption_processor, caption_model, device):
    prompt = "a fashion product photo of"
    inputs = caption_processor(
        images=image.convert("RGB"),
        text=prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output_ids = caption_model.generate(
            **inputs,
            max_new_tokens=25
        )

    caption = caption_processor.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption


def build_description_prompt(subcategory: str, caption: str) -> str:
    return f"""
You are a professional e-commerce copywriter for a fashion store.

Write a short and attractive product description based on the given information.

Predicted subcategory: {subcategory}
Image caption: {caption}

Requirements:
- Write 2 sentences only
- Sound natural, stylish, and suitable for an online fashion store
- Keep it concise
- Focus on visible appearance and general usage
- Do not invent brand, price, material, or exact technical details
- Make sure the description is consistent with the predicted subcategory
"""


def generate_product_description(subcategory, caption, tokenizer, model):
    prompt = build_description_prompt(subcategory, caption)

    messages = [
        {"role": "system", "content": "You are a helpful fashion e-commerce assistant."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()

    return response


# =========================
# UI
# =========================
st.title("Fashion Product Image Classifier")
st.write("Upload a fashion product image to predict its subcategory and generate a product description.")

models = load_models()
device = models["device"]

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    # 1) Predict subcategory
    results = predict_topk(
        image,
        models["classifier_processor"],
        models["classifier_model"],
        device,
        k=5,
    )
    subcategory_pred, subcategory_score = results[0]

    # 2) Generate caption
    caption = generate_caption(
        image,
        models["caption_processor"],
        models["caption_model"],
        device,
    )

    # 3) Generate product description with Qwen
    description = generate_product_description(
        subcategory=subcategory_pred,
        caption=caption,
        tokenizer=models["llm_tokenizer"],
        model=models["llm_model"],
    )

    # Display results
    st.subheader("Predicted Subcategory")
    st.write(f"**{subcategory_pred}** ({subcategory_score:.4f})")

    st.subheader("Top-5 Predictions")
    for label, score in results:
        st.write(f"**{label}** — {score:.4f}")

    st.subheader("Image Caption")
    st.write(caption)

    st.subheader("Generated Product Description")
    st.write(description)
