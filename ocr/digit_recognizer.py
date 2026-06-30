import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = Path(__file__).parent / "model.pt"
IMG_SIZE = 28
EMPTY_THRESHOLD = 0.04    # mean activation of center region below this -> empty
CONFIDENCE_MIN  = 0.55    # minimum softmax confidence to accept a prediction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DigitCNN(nn.Module):

    def __init__(self, n_classes: int = 9):
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
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return self.fc2(x)


class DigitRecognizer:

    def __init__(self, model_path = MODEL_PATH, device: torch.device = DEVICE):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {path}.\n"
                f"Run `python ocr/model_trainer.py` first."
            )
        checkpoint = torch.load(path, map_location=device)
        self.device = device
        self.model = DigitCNN(n_classes=checkpoint.get("n_classes", 9)).to(device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    # public API

    @torch.no_grad()
    def predict(self, cell: np.ndarray) -> int:
        """Classify a single 28x28 grayscale cell. Returns 0 for empty, 1-9 for digits."""
        normalized = self._preprocess(cell)
        if self._is_empty(normalized):
            return 0
        x = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(1)
        if conf.item() < CONFIDENCE_MIN:
            return 0
        return int(pred.item()) + 1

    @torch.no_grad()
    def predict_board(self, cells) -> tuple:
        """
        Run inference on a 9x9 list of cell images.
        Returns (board_2d, elapsed_seconds).
        """
        start = time.time()

        flat = [c for row in cells for c in row]
        preprocessed = [self._preprocess(c) for c in flat]
        empty_mask = [self._is_empty(p) for p in preprocessed]

        non_empty_idx = [i for i, e in enumerate(empty_mask) if not e]
        digits = [0] * 81

        if non_empty_idx:
            batch = np.stack([preprocessed[i] for i in non_empty_idx])  # (N, 28, 28)
            x = torch.from_numpy(batch).unsqueeze(1).to(self.device)    # (N, 1, 28, 28)
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            confs, preds = probs.max(1)
            confs = confs.cpu().numpy()
            preds = preds.cpu().numpy()
            for i, idx in enumerate(non_empty_idx):
                if confs[i] >= CONFIDENCE_MIN:
                    digits[idx] = int(preds[i]) + 1
                # else: stays 0 (low confidence -> treat as empty)

        board = [digits[r * 9:(r + 1) * 9] for r in range(9)]
        return board, time.time() - start

    # private helpers

    @staticmethod
    def _preprocess(cell: np.ndarray) -> np.ndarray:
        cell = cv2.resize(cell, (IMG_SIZE, IMG_SIZE))
        arr = cell.astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        return arr

    @staticmethod
    def _is_empty(arr: np.ndarray) -> bool:
        # Focus on the center 18x18 region to ignore border grid lines/leakage
        center = arr[5:23, 5:23]
        return float(center.mean()) < EMPTY_THRESHOLD
