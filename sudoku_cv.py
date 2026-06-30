import sys
import argparse
import time
import cv2
import numpy as np

from sudoku_locator import SudokuLocator
from ocr.digit_recognizer import DigitRecognizer
from solver import solve_timed, print_board
import copy


def process_image(path: str, save_path: str | None = None, show: bool = True) -> np.ndarray:
    frame = cv2.imread(path)
    if frame is None:
        sys.exit(f"Cannot read image: {path}")

    locator = SudokuLocator()
    print(f"Locating grid in '{path}'…")
    if not locator.locate(frame):
        sys.exit("No Sudoku grid found in image.")

    print("Recognising digits…")
    rec = DigitRecognizer()
    puzzle, ocr_time = rec.predict_board(locator.cells)
    clues = sum(cell != 0 for row in puzzle for cell in row)
    print(f"Inference of {clues} digits done in {ocr_time:.3f} seconds")

    print("\nPuzzle:")
    print_board(puzzle)

    solution = copy.deepcopy(puzzle)
    ok, elapsed = solve_timed(solution)
    if not ok:
        print("\nNo solution found - check OCR accuracy.")
        return frame

    print(f"\nSolved sudoku in {elapsed:.3f} seconds")
    print_board(solution)

    result = locator.overlay_solution(frame, puzzle, solution)

    if save_path:
        cv2.imwrite(save_path, result)
        print(f"\nSaved to '{save_path}'")

    if show:
        cv2.imshow("Solved Sudoku", result)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and solve a Sudoku from an image.")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--save", metavar="OUT", help="Save result image to this path")
    parser.add_argument("--no-show", action="store_true", help="Don't open a display window")
    args = parser.parse_args()

    process_image(args.image, save_path=args.save, show=not args.no_show)
