import streamlit as st
import torch
import time
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)

st.set_page_config(
    page_title="Fashion Product Classifier",
    page_icon="👗",
    layout="wide"
)

# =========================
# Model paths
# =========================
CLASSIFIER_MODEL_PATH = "Albatrosszzz/Fashion-Product-Classify-Vit"
CAPTION_MODEL_PATH = "Salesforce/blip-image-captioning-base"
LLM_MODEL_PATH = "Qwen/Qwen3-0.6B"


# =========================
# Load models
# =========================
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Image classification model
    classifier_processor = AutoImageProcessor.from_pretrained(CLASSIFIER_MODEL_PATH)
    classifier_model = AutoModelForImageClassification.from_pretrained(CLASSIFIER_MODEL_PATH)
    classifier_model.to(device)
    classifier_model.eval()

    # 2) Image caption model
    caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL_PATH)
    caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_PATH)
    caption_model.to(device)
    caption_model.eval()

    # 3) LLM for description generation
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
def predict_top1(image: Image.Image, processor, model, device):
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {kk: vv.to(device) for kk, vv in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred_id = torch.argmax(probs).item()
    pred_label = model.config.id2label[pred_id]
    pred_score = float(probs[pred_id].item())

    return pred_label, pred_score


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
- Write 1 to 2 sentences only
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
        {"role": "user", "content": prompt + "\n/no_think"},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        text = f"System: You are a helpful fashion e-commerce assistant.\nUser: {prompt}\nAssistant:"

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=100,
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

    if "</think>" in response:
        response = response.split("</think>", 1)[1].strip()

    response = response.replace("<think>", "").replace("</think>", "").strip()

    return response


# =========================
# UI styles
# =========================
st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .sub-text {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 14px;
        border: 1px solid #e6e6e6;
        margin-top: 0.8rem;
    }
    .label-box {
        background-color: #eef4ff;
        padding: 0.9rem 1rem;
        border-radius: 12px;
        border: 1px solid #d6e4ff;
        margin-bottom: 1rem;
    }
    .label-title {
        font-size: 0.95rem;
        color: #555;
        margin-bottom: 0.2rem;
    }
    .label-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #111;
    }
    .desc-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Fashion Product Image Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">Upload a fashion product image to predict its subcategory and generate a polished product description.</div>',
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("About")
    st.write(
        "This demo uses an image classification model to predict the product subcategory, "
        "then combines image captioning and a language model to generate a short e-commerce description."
    )

    st.header("Supported formats")
    st.write("JPG, JPEG, PNG, WEBP")


# =========================
# Main app
# =========================
models = load_models()
device = models["device"]

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is None:
    st.info("Please upload a fashion product image to begin.")
else:
    image = Image.open(uploaded_file)

    with st.spinner("Analyzing image and generating description..."):
        # 1) Predict subcategory
        subcategory_pred, subcategory_score = predict_top1(
            image,
            models["classifier_processor"],
            models["classifier_model"],
            device,
        )

        # 2) Generate caption + measure runtime
        caption_start_time = time.perf_counter()
        caption = generate_caption(
            image,
            models["caption_processor"],
            models["caption_model"],
            device,
        )
        caption_time = time.perf_counter() - caption_start_time

        # 3) Generate LLM description + measure runtime
        llm_start_time = time.perf_counter()
        description = generate_product_description(
            subcategory=subcategory_pred,
            caption=caption,
            tokenizer=models["llm_tokenizer"],
            model=models["llm_model"],
        )
        llm_time = time.perf_counter() - llm_start_time

    left_col, right_col = st.columns([1, 1.1], gap="large")

    with left_col:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with right_col:
        st.markdown("""
            <div class="label-box">
                <div class="label-title">Predicted Subcategory</div>
                <div class="label-value">{}</div>
            </div>
        """.format(subcategory_pred), unsafe_allow_html=True)

        st.progress(min(float(subcategory_score), 1.0))
        st.caption(f"Confidence score: {subcategory_score:.4f}")

        st.markdown("""
            <div class="result-card">
                <div class="desc-title">Generated Product Description</div>
                <div>{}</div>
            </div>
        """.format(description), unsafe_allow_html=True)

    with st.expander("See technical details"):
        st.write(f"**Predicted subcategory:** {subcategory_pred}")
        st.write(f"**Confidence score:** {subcategory_score:.4f}")
        st.write(f"**Internal image caption:** {caption}")
        st.write(f"**Image caption runtime:** {caption_time:.4f} seconds")
        st.write(f"**LLM generation runtime:** {llm_time:.4f} seconds")
