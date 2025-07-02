import gradio as gr
import torch
from model import UNetGenerator
from utils import process_image, output_to_image
from PIL import Image
import numpy as np


# Load model
model = UNetGenerator()
model.load_state_dict(torch.load("focusgan_weights.pth", map_location="cpu"))
model.eval()


def generate_saliency(input_img):
    # Convert NumPy array to PIL Image
    if isinstance(input_img, np.ndarray):
        input_img = Image.fromarray(input_img.astype("uint8")).convert("RGB")

    img_tensor = process_image(input_img)
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
    return output_to_image(output.squeeze(0))

demo = gr.Interface(
    fn=generate_saliency,
    inputs="image",
    outputs="image",
    title="Do You and FocusGAN Focus on the Same Thing?",
    description="Upload an image. Let FocusGAN show you what it thinks matters most."
)

if __name__ == "__main__":
    demo.launch(share=True)
