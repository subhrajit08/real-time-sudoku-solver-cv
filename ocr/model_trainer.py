import sys
import random
import argparse
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
except ImportError:
    sys.exit("pip install Pillow")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split
except ImportError:
    sys.exit("pip install torch")

# Config

IMG_SIZE  = 28
N_CLASSES = 9          # digits 1-9
DEFAULT_SAMPLES_PER_DIGIT_PER_FONT = 200
DEFAULT_EPOCHS = 30
BATCH_SIZE = 128

SCRIPT_DIR = Path(__file__).parent
FONT_DIR   = SCRIPT_DIR / "fonts"
MODEL_PATH = SCRIPT_DIR / "model.pt"

FONT_FILES = [
    "arial.ttf", "calibri.ttf", "Cambria.ttf",
    "FranklinGothic.ttf", "futur.ttf", "Garamond.ttf",
    "Helvetica 400.ttf", "rock.ttf", "times.ttf", "verdana.ttf",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset generation

def render_digit(digit: int, font_path: str | None, augment: bool = True) -> np.ndarray:
    size = IMG_SIZE * 4   # render large, then downscale for anti-aliasing
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)

    font_size_px = random.randint(int(size * 0.5), int(size * 0.85))
    try:
        f = ImageFont.truetype(font_path, font_size_px) if font_path else ImageFont.load_default()
    except Exception:
        f = ImageFont.load_default()

    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) // 2 - bbox[0]
    y = (size - th) // 2 - bbox[1]

    if augment:
        x += random.randint(-size // 10, size // 10)
        y += random.randint(-size // 10, size // 10)

    draw.text((x, y), text, fill=0, font=f)

    if augment:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, fillcolor=255)
        if random.random() < 0.4:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))

    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = 1.0 - arr   # invert: ink = 1.0 (matches thresholded cells at inference time)

    if augment:
        arr += np.random.normal(0, 0.05, arr.shape).astype(np.float32)
        arr = np.clip(arr, 0, 1)

    return arr


class DigitFontDataset(Dataset):

    def __init__(self, font_paths: list, samples_per_digit_per_font: int):
        self.samples = []
        for digit in range(1, 10):
            for font_path in font_paths:
                for _ in range(samples_per_digit_per_font):
                    self.samples.append((digit, font_path))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        digit, font_path = self.samples[idx]
        arr = render_digit(digit, font_path, augment=True)
        x = torch.from_numpy(arr).unsqueeze(0)        # (1, 28, 28)
        y = digit - 1                                   # classes 0-8
        return x, y


def discover_fonts() -> list:
    paths = []
    if FONT_DIR.exists():
        for p in FONT_DIR.iterdir():
            if p.suffix.lower() in (".ttf", ".ttc"):
                paths.append(str(p))
    if paths:
        print(f"  Found {len(paths)} font file(s) in {FONT_DIR}")
    else:
        print("  No fonts found - falling back to PIL default font (low diversity).")
        paths = [None]
    return paths


# Model architecture
class DigitCNN(nn.Module):

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        # 28 -> 14 -> 7
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return self.fc2(x)   # raw logits


# Training loop

def train(epochs: int, samples_per_font: int, patience: int = 5):
    print(f"Device: {DEVICE}")
    print("=== Building dataset from fonts ===")
    font_paths = discover_fonts()
    dataset = DigitFontDataset(font_paths, samples_per_font)
    print(f"  Generated {len(dataset)} training samples from {len(font_paths)} font(s)")

    n_val = int(0.1 * len(dataset))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"  Train: {n_train}  Val: {n_val}")

    model = DigitCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    print("\n=== Training CNN ===")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_correct += (out.argmax(1) == y).sum().item()
            train_total += x.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += x.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        print(f"Epoch {epoch:2d}/{epochs} - "
              f"loss: {train_loss/train_total:.4f} acc: {train_acc:.4f} - "
              f"val_loss: {val_loss/val_total:.4f} val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping (no improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({
        "model_state": model.state_dict(),
        "n_classes": N_CLASSES,
        "img_size": IMG_SIZE,
        "architecture": "DigitCNN_v1",
    }, MODEL_PATH)

    print(f"\nSaved -> {MODEL_PATH}")
    print(f"Best validation accuracy: {best_val_acc * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--samples-per-font", type=int, default=DEFAULT_SAMPLES_PER_DIGIT_PER_FONT)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    train(args.epochs, args.samples_per_font, args.patience)
