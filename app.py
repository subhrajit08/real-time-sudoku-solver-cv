import os
import io
import copy
import uuid
import argparse
import base64
import time

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

from sudoku_locator import SudokuLocator
from ocr.digit_recognizer import DigitRecognizer
from solver import solve_timed, print_board

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload limit

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Load model once at startup
try:
    _recognizer = DigitRecognizer()
    print("✓  Digit recogniser loaded")
except FileNotFoundError as e:
    print(f"⚠  {e}")
    _recognizer = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/solve", methods=["POST"])
def solve_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    if _recognizer is None:
        return jsonify({"error": "Model not loaded. Run `python ocr/model_trainer.py` first."}), 503

    # Read image
    img_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    t0 = time.time()

    # Locate grid
    locator = SudokuLocator()
    if not locator.locate(frame):
        return jsonify({"error": "No Sudoku grid detected in the image."}), 422

    # OCR
    puzzle, ocr_time = _recognizer.predict_board(locator.cells)
    clues = sum(cell != 0 for row in puzzle for cell in row)

    # Solve
    solution = copy.deepcopy(puzzle)
    ok, solve_time = solve_timed(solution)
    if not ok:
        return jsonify({"error": "Puzzle has no solution (check image clarity)."}), 422

    # Overlay
    result = locator.overlay_solution(frame, puzzle, solution)

    # Encode as base64 PNG for JSON response
    _, buf = cv2.imencode(".png", result)
    img_b64 = base64.b64encode(buf).decode("utf-8")

    total = time.time() - t0
    return jsonify({
        "success":     True,
        "image_b64":   img_b64,
        "ocr_seconds": round(ocr_time, 3),
        "solve_seconds": round(solve_time, 3),
        "total_seconds": round(total, 3),
        "clues_found": clues,
        "puzzle":      puzzle,
        "solution":    solution,
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
