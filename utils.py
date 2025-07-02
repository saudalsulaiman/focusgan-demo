from torchvision import transforms
from PIL import Image

image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def process_image(pil_image):
    return image_transform(pil_image)

def output_to_image(tensor):
    import matplotlib.cm as cm
    import numpy as np
    from PIL import Image

    arr = tensor.squeeze().detach().cpu().numpy()
    arr = (arr * 255).astype("uint8")
    gray = Image.fromarray(arr).convert("L")
    colormap = cm.jet(np.array(gray))
    color_img = Image.fromarray((colormap * 255).astype(np.uint8)).convert("RGB")
    return color_img
