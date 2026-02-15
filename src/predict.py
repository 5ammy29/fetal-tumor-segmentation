from PIL import Image
from model import UNet
import numpy as np
import os
from scipy.ndimage import label
import torch
from torchvision import transforms
from tqdm import tqdm

test_dir = "/Users/sasha/Documents/fetal-tumor-segmentation/data/dataset/test/images"
pred_dir = "/Users/sasha/Documents/fetal-tumor-segmentation/results/predictions"
model_path = "/Users/sasha/Documents/fetal-tumor-segmentation/results/unet.pth"
threshold = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(pred_dir, exist_ok=True)

model = UNet()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully")

transform = transforms.Compose([
    transforms.ToTensor(),
])

def keep_largest_component(binary_mask):
    labeled_mask, num_features = label(binary_mask)

    if num_features == 0:
        return binary_mask

    largest_component = 0
    largest_size = 0

    for component_id in range(1, num_features + 1):
        size = np.sum(labeled_mask == component_id)
        if size > largest_size:
            largest_size = size
            largest_component = component_id

    return (labeled_mask == largest_component).astype(np.uint8)

image_files = sorted(
    f for f in os.listdir(test_dir)
    if f.endswith(".png")
)

with torch.no_grad():
    for img_name in tqdm(image_files):

        img_path = os.path.join(test_dir, img_name)
        image = Image.open(img_path).convert("L")

        input_tensor = transform(image).unsqueeze(0).to(device)

        logits = model(input_tensor)
        probs = torch.sigmoid(logits)

        pred_mask = (probs > threshold).float()
        pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)

        pred_mask = keep_largest_component(pred_mask)

        pred_mask = (pred_mask * 255).astype(np.uint8)

        base_name = os.path.splitext(img_name)[0]
        save_name = f"{base_name}_pred.png"
        save_path = os.path.join(pred_dir, save_name)

        Image.fromarray(pred_mask).save(save_path)

print("Inference completed")
