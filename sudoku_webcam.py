import argparse
import time
import copy
import cv2
import numpy as np
import threading

from sudoku_locator import SudokuLocator
from ocr.digit_recognizer import DigitRecognizer
from solver import solve_timed, is_valid


def boards_equal(a, b):
    return all(a[r][c] == b[r][c] for r in range(9) for c in range(9))


class SolverThread:
    """Runs OCR + solving in a background thread so the UI never freezes."""

    def __init__(self, rec):
        self.rec = rec
        self.lock = threading.Lock()
        self.is_running = False
        self.latest_puzzle = None
        self.cached_puzzle = None
        self.cached_solution = None
        self.solve_status = "idle"  # "idle", "detecting", "solving", "solved", "failed"

    def start_solve(self, cells):
        with self.lock:
            if self.is_running:
                return
            self.is_running = True
            self.solve_status = "detecting"
            cells_copy = [[c.copy() for c in row] for row in cells]

        threading.Thread(target=self._process, args=(cells_copy,), daemon=True).start()

    def _process(self, cells):
        try:
            puzzle, _ = self.rec.predict_board(cells)

            with self.lock:
                self.latest_puzzle = puzzle
                cached = self.cached_puzzle

            # Re-solve only if puzzle changed
            if cached is None or not boards_equal(puzzle, cached):
                # Count detected digits — skip solving if too few clues
                clue_count = sum(1 for row in puzzle for v in row if v != 0)
                if clue_count < 12:
                    with self.lock:
                        self.solve_status = "detecting"
                        self.is_running = False
                    return

                with self.lock:
                    self.solve_status = "solving"

                solution = copy.deepcopy(puzzle)
                ok, _ = solve_timed(solution)

                with self.lock:
                    if ok:
                        self.cached_puzzle = puzzle
                        self.cached_solution = solution
                        self.solve_status = "solved"
                    else:
                        self.solve_status = "failed"
                    self.is_running = False
            else:
                with self.lock:
                    self.solve_status = "solved" if self.cached_solution else "detecting"
                    self.is_running = False
        except Exception:
            with self.lock:
                self.solve_status = "failed"
                self.is_running = False

    def get_state(self):
        with self.lock:
            return (self.latest_puzzle, self.cached_puzzle,
                    self.cached_solution, self.solve_status)

    def reset(self):
        with self.lock:
            self.cached_puzzle = None
            self.cached_solution = None
            self.latest_puzzle = None
            self.solve_status = "idle"


def draw_status_hud(frame, fps, grid_found, clue_count, status):
    """Draw a compact status bar at the top-left of the frame."""
    h, w = frame.shape[:2]

    # Background bar
    bar_h = 45
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Status color and text
    status_colors = {
        "idle":      ((180, 180, 180), "Searching..."),
        "detecting": ((0, 200, 255),   f"Detecting ({clue_count} digits)"),
        "solving":   ((0, 255, 255),   "Solving..."),
        "solved":    ((0, 255, 0),     f"SOLVED ({clue_count} clues)"),
        "failed":    ((0, 100, 255),   "OCR Error - adjusting..."),
    }
    color, text = status_colors.get(status, ((180, 180, 180), status))

    # Status indicator dot
    cv2.circle(frame, (20, bar_h // 2), 8, color, -1)

    # Status text
    cv2.putText(frame, text, (38, bar_h // 2 + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    # Grid indicator
    grid_text = "Grid: YES" if grid_found else "Grid: ---"
    grid_color = (0, 255, 0) if grid_found else (100, 100, 100)
    cv2.putText(frame, grid_text, (w - 280, bar_h // 2 + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, grid_color, 1, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 100, bar_h // 2 + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # Hotkey hints at bottom
    hint = "Q=Quit  S=Screenshot  R=Reset"
    cv2.putText(frame, hint, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

    return frame


def run_webcam(camera_idx: int = 0, debug: bool = False):
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"Cannot open camera {camera_idx}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    rec = DigitRecognizer()
    locator = SudokuLocator(debug=debug)
    solver_thread = SolverThread(rec)

    last_detect_time = 0
    DETECT_INTERVAL  = 0.4   # limit spawning solve thread

    # FPS tracking
    fps = 0.0
    frame_times = []

    print("Sudoku Webcam Solver started.")
    print("Press  Q = quit  |  S = screenshot  |  R = reset cache")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        frame_times.append(now)
        # Keep only last 30 frame timestamps for FPS calculation
        frame_times = [t for t in frame_times if now - t < 1.0]
        fps = len(frame_times)

        display = frame.copy()
        grid_found = False

        # -- Locate grid every frame (cheap) --
        if locator.locate(frame):
            grid_found = True
            display = locator.draw_debug(display)

            # -- OCR + solve on interval in background --
            if now - last_detect_time > DETECT_INTERVAL:
                last_detect_time = now
                solver_thread.start_solve(locator.cells)

            latest_puzzle, cached_puzzle, cached_solution, status = solver_thread.get_state()

            # -- Overlay solution and OCR clues --
            if cached_solution is not None and cached_puzzle is not None:
                display = locator.overlay_solution(
                    display, cached_puzzle, cached_solution, show_clues=True)
            elif latest_puzzle is not None:
                # Show detected clues immediately even before solve completes
                display = locator.overlay_solution(
                    display, latest_puzzle, latest_puzzle, show_clues=True)
        else:
            # Still get state to draw HUD even if locator temporarily loses track
            latest_puzzle, cached_puzzle, cached_solution, status = solver_thread.get_state()

        # -- Draw OCR HUD (always, when we have puzzle data) --
        display_puzzle = cached_puzzle if cached_puzzle is not None else latest_puzzle
        if display_puzzle is not None:
            is_solved = cached_solution is not None
            display = locator.draw_ocr_hud(display, display_puzzle, solved=is_solved)

        # -- Status HUD --
        clue_count = 0
        if display_puzzle is not None:
            clue_count = sum(1 for row in display_puzzle for v in row if v != 0)
        display = draw_status_hud(display, fps, grid_found, clue_count, status)

        cv2.imshow("Sudoku Solver", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(fname, display)
            print(f"Saved {fname}")
        elif key == ord("r"):
            solver_thread.reset()
            print("Cache reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()
    run_webcam(args.camera, args.debug)
