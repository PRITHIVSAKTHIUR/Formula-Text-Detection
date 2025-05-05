import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Formula-Text-Detection"  # Replace with your model path if different
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Label mapping
id2label = {
    "0": "formula",
    "1": "text"
}

def classify_formula_or_text(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {
        id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
    }

    return prediction

# Gradio Interface
iface = gr.Interface(
    fn=classify_formula_or_text,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=2, label="Formula or Text"),
    title="Formula-Text-Detection",
    description="Upload an image region to classify whether it contains a mathematical formula or natural text."
)

if __name__ == "__main__":
    iface.launch()
