import modal

app = modal.App("fetal-tumor-segmentation")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "pillow",
        "matplotlib",
        "scipy"
    )
)

root = "/data"

volume = modal.Volume.from_name(
    "dataset",
    create_if_missing=True,
)

@app.function(
    image=image,
    gpu="A10G",          
    cpu=4,              
    memory=32_000,      
    timeout=60 * 60 * 10,
    volumes={root: volume},
)

def train():

    from PIL import Image
    import numpy as np
    import os
    import random
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    device = torch.device("cuda")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset_dirs = {
        "train": {
            "image_dir": f"{root}/train/images",
            "mask_dir":  f"{root}/train/filled_masks",
        },
        "valid": {
            "image_dir": f"{root}/valid/images",
            "mask_dir":  f"{root}/valid/filled_masks",
        },
    }

    def load_grayscale(path):

        return np.array(Image.open(path).convert("L"))

    def normalize_image(img):

        return img.astype(np.float32) / 255.0

    def binarize_mask(mask):

        return (mask > 0).astype(np.uint8)

    def to_tensor(arr):

        return torch.tensor(arr).unsqueeze(0)

    def dice_loss(logits, targets, eps=1e-6):

        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum()
        den = probs.sum() + targets.sum() + eps
        return 1 - num / den

    class FetalTumorDataset(Dataset):

        def __init__(self, image_dir, mask_dir):

            self.image_dir = image_dir
            self.mask_dir = mask_dir
            self.files = sorted(
                f for f in os.listdir(image_dir) if f.endswith(".png")
            )

        def __len__(self):

            return len(self.files)

        def __getitem__(self, idx):

            fname = self.files[idx]

            img_path = os.path.join(self.image_dir, fname)

            stem, ext = os.path.splitext(fname)

            parts = stem.split("_")

            if parts[-1].lstrip("-").isdigit():
                base = "_".join(parts[:-1])
                angle = parts[-1]
                mask_name = f"{base}_Annotation_{angle}{ext}"
            else:
                mask_name = f"{stem}_Annotation{ext}"

            mask_path = os.path.join(self.mask_dir, mask_name)

            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Missing mask: {mask_path}")

            img = normalize_image(load_grayscale(img_path))
            mask = binarize_mask(load_grayscale(mask_path))

            img = to_tensor(img).float()
            mask = to_tensor(mask).float()

            return img, mask

    train_loader = DataLoader(
        FetalTumorDataset(**dataset_dirs["train"]),
        batch_size=4,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

    valid_loader = DataLoader(
        FetalTumorDataset(**dataset_dirs["valid"]),
        batch_size=4,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    class ConvBlock(nn.Module):

        def __init__(self, in_channels, out_channels):

            super().__init__()

            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.conv(x)

    class DownBlock(nn.Module):

        def __init__(self, in_channels, out_channels):

            super().__init__()
            self.conv = ConvBlock(in_channels, out_channels)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x):

            x_conv = self.conv(x)
            return x_conv, self.pool(x_conv)

    class UpBlock(nn.Module):

        def __init__(self, in_channels, skip_channels, out_channels):

            super().__init__()
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = ConvBlock(
                in_channels=out_channels + skip_channels, 
                out_channels=out_channels)

        def forward(self, x, skip):

            x = self.up(x)
            x = torch.cat([skip, x], dim=1)
            return self.conv(x)

    class UNet(nn.Module):

        def __init__(self):

            super().__init__()
            self.down1 = DownBlock(1, 64)
            self.down2 = DownBlock(64, 128)
            self.down3 = DownBlock(128, 256)
            self.down4 = DownBlock(256, 512)

            self.bottleneck = ConvBlock(512, 1024)

            self.up4 = UpBlock(1024, 512, 512)
            self.up3 = UpBlock(512, 256, 256)
            self.up2 = UpBlock(256, 128, 128)
            self.up1 = UpBlock(128, 64, 64)

            self.out = nn.Conv2d(64, 1, kernel_size=1)

        def forward(self, x):

            s1, x = self.down1(x)
            s2, x = self.down2(x)
            s3, x = self.down3(x)
            s4, x = self.down4(x)

            x = self.bottleneck(x)

            x = self.up4(x, s4)
            x = self.up3(x, s3)
            x = self.up2(x, s2)
            x = self.up1(x, s1)

            return self.out(x)

    model = UNet().to(device)

    fg_pixels, total_pixels = 0, 0
    for i, (_, mask) in enumerate(train_loader):
        if i >= 20:  
            break
        fg_pixels += mask.sum().item()
        total_pixels += mask.numel()

    fg_ratio = fg_pixels / total_pixels
    print("Foreground ratio (sampled):", fg_ratio)

    pos_weight = torch.tensor([(1 - fg_ratio) / fg_ratio], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    def evaluate():
        model.eval()
        loss_sum = 0
        with torch.no_grad():
            for img, mask in valid_loader:
                img = img.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                logits = model(img)
                loss = dice_loss(logits, mask) + 0.2 * bce(logits, mask)
                loss_sum += loss.item()
        return loss_sum / len(valid_loader)

    epochs = 35
    best_val = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for img, mask in train_loader:
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(img)
            loss = dice_loss(logits, mask) + 0.2 * bce(logits, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss = evaluate()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{root}/unet.pth")
            volume.commit()

        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f}"
        )

@app.local_entrypoint()
def main():
    train.remote()
