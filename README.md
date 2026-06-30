# Real-Time Sudoku Solver - Computer Vision Pipeline

Real-time computer vision system to detect, extract, and solve Sudoku puzzles from
a live webcam feed or static image, overlaying the solution onto the original
perspective.

**Stack:** OpenCV (grid detection & perspective transform) · PyTorch (CNN digit
recognition) · Python (recursive backtracking solver) · Flask (web app)


<img width="343" height="505" alt="image" src="https://github.com/user-attachments/assets/54b256ce-dbc9-4bbc-b821-9a02645cc943" />

---

## How it works

```
┌──────────────┐    ┌────────────────────┐    ┌───────────────┐    ┌─────────────┐
│  Webcam /    │───▶│  OpenCV            │───▶│  PyTorch CNN  │───▶│  Backtrack  │
│  image input │    │  grid detection +  │    │  digit OCR    │    │  solver     │
│              │    │  perspective warp  │    │  (81 cells)   │    │             │
└──────────────┘    └────────────────────┘    └───────────────┘    └─────────────┘
                              ▲                                            │
                              └──────────── inverse warp & overlay ◀───────┘
```

1. **Grid detection** (`sudoku_locator.py`) - adaptive threshold + morphological
   closing -> largest convex quadrilateral contour -> 4-point perspective
   transform -> crop into 81 cells. Includes temporal smoothing for stable
   real-time tracking.
2. **Digit recognition** (`ocr/`) - a small CNN trained on digits rendered from
   system fonts with heavy augmentation classifies each cell as empty or 1-9.
   Predictions below a confidence threshold (55%) are rejected to prevent
   garbage OCR.
3. **Solving** (`solver.py`) - recursive backtracking with MRV (Minimum
   Remaining Values) heuristic fills in the grid.
4. **Overlay** - the solved digits are warped back through the inverse
   perspective matrix and blended onto the original frame in real-time, so the
   solution appears in the same plane as the photographed/filmed grid.

---

## Project structure

```
sudoku_solver/
├── ocr/
│   ├── fonts/                <- copy .ttf/.ttc font files here (auto-discovered)
│   ├── model_trainer.py      <- generates dataset + trains the CNN (PyTorch)
│   ├── digit_recognizer.py   ← loads model.pt and runs inference
│   └── model.pt               <- produced after training
├── images/                   <- put sample Sudoku photos here
├── static/uploads/           <- Flask upload scratch directory
├── templates/
│   └── index.html            <- Flask front-end (drag & drop UI)
├── solver.py                 <- recursive backtracking + MRV heuristic solver
├── sudoku_locator.py         <- OpenCV grid detection / perspective / cell split
├── sudoku_cv.py              <- CLI: solve a single image
├── sudoku_webcam.py          <- real-time webcam loop (threaded OCR + solving)
├── app.py                    <- Flask server
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create an environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt`:
```
opencv-python>=4.8.0
torch>=2.1.0
numpy>=1.24.0
Pillow>=10.0.0
flask>=3.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

> **GPU (optional):** if you have CUDA available, install the matching PyTorch
> build from [pytorch.org](https://pytorch.org/get-started/locally/) instead of
> the plain `pip install torch` — training will run noticeably faster.

### 2. Add font files

The CNN is trained on digits rendered from **system fonts** rather than a
generic handwriting dataset like MNIST (printed Sudoku puzzles use clean
typefaces, not handwriting, so this generalizes far better).

Copy any `.ttf` or `.ttc` font files into `ocr/fonts/`. The trainer
**auto-discovers** all font files in the directory - no hardcoded list needed.

On **Windows**, you can copy common fonts:
```powershell
copy C:\Windows\Fonts\arial.ttf ocr\fonts\
copy C:\Windows\Fonts\calibri.ttf ocr\fonts\
copy C:\Windows\Fonts\cambria.ttc ocr\fonts\
copy C:\Windows\Fonts\times.ttf ocr\fonts\
copy C:\Windows\Fonts\verdana.ttf ocr\fonts\
copy C:\Windows\Fonts\tahoma.ttf ocr\fonts\
copy C:\Windows\Fonts\georgia.ttf ocr\fonts\
copy C:\Windows\Fonts\consola.ttf ocr\fonts\
```

On **macOS:** `/Library/Fonts/` or `/System/Library/Fonts/`
On **Linux:** `/usr/share/fonts/truetype/` (install `fonts-liberation` /
`ttf-mscorefonts-installer`)

Any 5-10 distinct sans/serif fonts gives good results. Font *diversity*
matters more than specific font names.

### 3. Train the CNN

```bash
python ocr/model_trainer.py
```

This will:
- Auto-discover all `.ttf`/`.ttc` font files in `ocr/fonts/`
- Render digits 1-9 in each font at random sizes/positions (~200 samples per
  digit per font)
- Apply augmentation: random rotation (+/-15 deg), Gaussian noise, occasional blur
- Train a CNN (2 conv blocks + dense head) for up to 30 epochs with early
  stopping and LR scheduling
- Save weights to `ocr/model.pt`

Example output (with 7 system fonts on Windows + CUDA):
```
Device: cuda
=== Building dataset from fonts ===
  Found 7 font file(s) in ocr\fonts
  Generated 12600 training samples from 7 font(s)
  Train: 11340  Val: 1260

=== Training CNN ===
Epoch  1/30 - loss: 0.5765 acc: 0.8331 - val_loss: 0.0604 val_acc: 0.9921
Epoch  2/30 - loss: 0.0765 acc: 0.9807 - val_loss: 0.0233 val_acc: 0.9960
Epoch  3/30 - loss: 0.0488 acc: 0.9872 - val_loss: 0.0157 val_acc: 1.0000
Epoch  4/30 - loss: 0.0382 acc: 0.9899 - val_loss: 0.0038 val_acc: 1.0000
Epoch  5/30 - loss: 0.0216 acc: 0.9938 - val_loss: 0.0017 val_acc: 1.0000
Epoch  6/30 - loss: 0.0204 acc: 0.9938 - val_loss: 0.0122 val_acc: 0.9963
Epoch  7/30 - loss: 0.0204 acc: 0.9944 - val_loss: 0.0009 val_acc: 1.0000
Early stopping (no improvement for 5 epochs)

Saved -> ocr/model.pt
Best validation accuracy: 100.00%
```

### Training accuracy summary

| Metric | Value |
|---|---|
| Best validation accuracy | **100.00%** |
| Final training accuracy | 99.44% |
| Final validation loss | 0.0009 |
| Epochs to converge | 7 / 30 (early stopped) |
| Training samples | 12,600 (7 fonts x 9 digits x 200 samples) |

Tune speed/quality with flags:
```bash
python ocr/model_trainer.py --epochs 20 --samples-per-font 150
```

---

## Usage

### Flask web app

```bash
python app.py
```

Open **http://localhost:5000**, drag and drop a photo of a Sudoku grid, and
the solved puzzle is returned overlaid on the original image, along with
timing stats (OCR time, solve time, total time).

```bash
python app.py --port 8080 --debug
```

### Static image (CLI)

```bash
python sudoku_cv.py images/1.jpeg
python sudoku_cv.py images/2.png --save solved_output.png --no-show
```

Example output:
```
Locating grid in 'images/2.png'…
Recognising digits…
Inference of 23 digits done in 0.041 seconds

Puzzle:
+-------+-------+-------+
|       | 8     | 3     |
| 5     |       |   1   |
|   8   | 6 7   |       |
+-------+-------+-------+
...

Solved sudoku in 0.412 seconds
+-------+-------+-------+
| 7 1 2 | 8 9 5 | 3 4 6 |
| 5 9 6 | 3 4 2 | 7 1 8 |
...
```

### Live webcam

```bash
python sudoku_webcam.py
python sudoku_webcam.py --camera 1     # use a different camera index
python sudoku_webcam.py --debug        # draw detected grid corners
```

**Controls:** `Q` quit - `S` save screenshot - `R` reset cached solution

The webcam runs with a **threaded architecture**:
- **Main thread:** captures frames, runs grid detection (cheap OpenCV), draws
  overlays and HUD at full frame rate - no freezing.
- **Background thread:** runs OCR + solving asynchronously, only when the
  detected puzzle changes.

**Real-time overlay features:**
- Detected clues shown immediately (cyan/blue) before solving completes
- Solved digits overlaid in green directly on the grid
- OCR preview HUD (top-right) shows the detected board as a mini grid
- Status bar (top) shows: detection state, digit count, solve status, FPS

---

## Implementation notes

### Grid detection (`sudoku_locator.py`)

1. Grayscale -> Gaussian blur (7x7) -> adaptive threshold
   (`ADAPTIVE_THRESH_GAUSSIAN_C`, block size 19, inverted binary)
2. **Morphological closing** (dilate x2 + erode x1) to connect broken grid
   lines under varying lighting conditions
3. `cv2.findContours` on the threshold image, sorted by area
4. `cv2.approxPolyDP` with **multiple epsilon values** (0.015-0.05) on the
   top 8 contours to find a 4-point polygon. Each candidate is validated:
   - Must be **convex** (`cv2.isContourConvex`)
   - Must have a roughly square **aspect ratio** (0.4-2.5)
   - Must have sufficient **fill ratio** (area vs bounding box > 50%)
5. **Temporal smoothing**: last valid corners are reused for up to 8 frames
   if detection temporarily fails, preventing flickering
6. Corners are ordered (TL, TR, BR, BL) and fed to
   `cv2.getPerspectiveTransform` + `cv2.warpPerspective` to produce a flat,
   square 450x450 view of the grid
7. The square is sliced into a 9x9 grid of cells, each cropped with a
   generous margin (1/6 of cell size) to remove grid-line bleed, blurred,
   and resized to 28x28 for the CNN
8. The inverse perspective matrix is cached so the solved digits can be
   warped back into the original camera/photo perspective for the overlay

### Digit recognition (`ocr/`)

- **Why fonts, not MNIST:** Sudoku puzzles are printed in regular fonts, not
  handwritten digits - training on MNIST gives poor accuracy on real puzzles
  (frequent 1/7/9 confusion). Generating a synthetic dataset from system
  fonts with augmentation matches the real distribution far better.
- **Auto font discovery:** the trainer scans `ocr/fonts/` for any `.ttf` or
  `.ttc` files automatically - no hardcoded font list needed.
- **Architecture (PyTorch):**
  `Conv(32)->BN->Conv(32)->MaxPool->Dropout -> Conv(64)->BN->Conv(64)->MaxPool->Dropout
  -> FC(256)->Dropout->FC(9)`
- **Augmentation:** random translation, +/-15 deg rotation, Gaussian blur (40%
  chance), Gaussian pixel noise
- **Empty-cell detection:** a cell is classified as empty (0) if its mean
  thresholded-pixel activation in the center 18x18 region is below 4%,
  skipping it from CNN inference entirely
- **Confidence filtering:** predictions are passed through softmax and
  rejected if the confidence is below **55%**. Low-confidence predictions
  are treated as empty cells, preventing garbage OCR from producing
  unsolvable boards.
- **Batched inference:** all non-empty cells in a 9x9 grid are stacked into a
  single tensor and classified in one forward pass

### Solver (`solver.py`)

Recursive backtracking with **MRV heuristic**:
- Find the empty cell with the **fewest valid candidates** (Minimum Remaining
  Values) - this dramatically prunes the search tree
- Try each candidate digit, checking row / column / 3x3-box validity
  (`is_valid`)
- Recurse on a valid placement; undo (backtrack) on dead ends
- **Step limit** (5000) prevents infinite loops from invalid OCR boards
- Includes `is_initial_board_valid()` check to reject conflicting boards
  before solving
- Solves any uniquely-solvable 9x9 puzzle in well under a second on average

Run `python solver.py` directly for a quick self-test on a sample puzzle.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `No Sudoku grid detected` | Improve lighting/contrast, ensure the full grid + border is visible and roughly flat-on |
| Low OCR accuracy | Add more/more-distinct fonts, increase `--samples-per-font`, retrain |
| OCR detecting wrong digits | The model uses 55% confidence threshold - lower `CONFIDENCE_MIN` in `digit_recognizer.py` if too aggressive, raise if too loose |
| `FileNotFoundError: model.pt` | Run `python ocr/model_trainer.py` before using the recognizer |
| Slow training | Reduce `--samples-per-font`, or install a CUDA build of PyTorch if you have a GPU |
| Webcam freezing | Should not happen with threaded architecture - ensure `opencv-python >= 4.8.0` |
| Webcam window doesn't open | Try `--camera 1`, `2`, etc., or check OS camera permissions |
