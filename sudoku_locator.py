import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four corner points as [top-left, top-right, bottom-right, bottom-left].
    Works regardless of how the contour returned them.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]      # TL  (smallest x+y)
    rect[2] = pts[np.argmax(s)]      # BR  (largest  x+y)
    rect[1] = pts[np.argmin(diff)]   # TR  (smallest x-y)
    rect[3] = pts[np.argmax(diff)]   # BL  (largest  x-y)
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Warp the quadrilateral defined by `pts` into a square bird's-eye view."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width/height of the destination square
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_side = int(max(width_a, width_b, height_a, height_b))

    dst = np.array(
        [[0, 0], [max_side - 1, 0], [max_side - 1, max_side - 1], [0, max_side - 1]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_side, max_side))
    return warped, M, rect


class SudokuLocator:
    """
    Detects the Sudoku grid in an image and provides:
      - `warped`      : perspective-corrected square image of the grid
      - `cells`       : 9x9 list of cell images (grayscale, 28x28)
      - `transform_M` : perspective matrix (to warp solution back)
      - `corners`     : original grid corner points (for overlay)
    """

    CELL_SIZE = 28          # CNN input size
    WARP_SIZE = 450         # internal working resolution
    SMOOTH_FRAMES = 8       # how many frames to keep last-good corners

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.warped: np.ndarray | None = None
        self.cells: list[list[np.ndarray]] | None = None
        self.transform_M: np.ndarray | None = None
        self.inv_M: np.ndarray | None = None
        self.corners: np.ndarray | None = None
        # Temporal smoothing
        self._frames_since_detect = 999
        self._last_good_corners: np.ndarray | None = None

    def locate(self, frame: np.ndarray) -> bool:
        """
        Run the full detection pipeline on `frame`.
        Returns True if a valid Sudoku grid was found.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        contour = self._find_grid_contour(gray)

        # Temporal smoothing: if detection fails, reuse last good corners
        # for a few frames to avoid flickering
        if contour is None:
            self._frames_since_detect += 1
            if (self._last_good_corners is not None
                    and self._frames_since_detect < self.SMOOTH_FRAMES):
                contour = self._last_good_corners
            else:
                return False
        else:
            self._frames_since_detect = 0
            self._last_good_corners = contour.copy()

        warped_gray, M, corners = four_point_transform(gray, contour)
        warped_color, _, _ = four_point_transform(
            frame if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR),
            contour,
        )

        # Resize to fixed working resolution
        warped_gray = cv2.resize(warped_gray, (self.WARP_SIZE, self.WARP_SIZE))
        self.warped = cv2.resize(warped_color, (self.WARP_SIZE, self.WARP_SIZE))
        self.corners = corners
        self.transform_M = M

        # Compute inverse transform for overlaying solution
        dst_corners = np.array(
            [[0, 0], [self.WARP_SIZE - 1, 0],
             [self.WARP_SIZE - 1, self.WARP_SIZE - 1], [0, self.WARP_SIZE - 1]],
            dtype="float32",
        )
        self.inv_M = cv2.getPerspectiveTransform(dst_corners, corners)

        self.cells = self._extract_cells(warped_gray)
        return True

    # ── Overlay ──────────────────────────────────────────────────────────────

    def overlay_solution(
        self,
        frame: np.ndarray,
        puzzle: list[list[int]],
        solution: list[list[int]],
        show_clues: bool = True,
    ) -> np.ndarray:
        """
        Draw the solved digits and optionally the detected clues back onto
        `frame` with the correct perspective, including a subtle grid overlay.
        """
        if self.inv_M is None:
            return frame

        h, w = frame.shape[:2]
        cell_px = self.WARP_SIZE // 9

        # Create a canvas in warped (bird's-eye) space
        canvas = np.zeros((self.WARP_SIZE, self.WARP_SIZE, 3), dtype=np.uint8)

        # Draw subtle grid lines on the canvas
        for i in range(10):
            thick = 2 if i % 3 == 0 else 1
            color = (80, 80, 80) if i % 3 == 0 else (40, 40, 40)
            pos = i * cell_px
            cv2.line(canvas, (pos, 0), (pos, self.WARP_SIZE), color, thick)
            cv2.line(canvas, (0, pos), (self.WARP_SIZE, pos), color, thick)

        # Draw digits
        font = cv2.FONT_HERSHEY_SIMPLEX
        for r in range(9):
            for c in range(9):
                # Center the text in each cell
                text_size = cv2.getTextSize("8", font, 1.1, 2)[0]
                cx = c * cell_px + (cell_px - text_size[0]) // 2
                cy = r * cell_px + (cell_px + text_size[1]) // 2

                if puzzle[r][c] == 0 and solution[r][c] != 0:
                    # Solved digit -> bright green
                    cv2.putText(canvas, str(solution[r][c]), (cx, cy),
                                font, 1.1, (0, 255, 0), 2, cv2.LINE_AA)
                elif show_clues and puzzle[r][c] != 0:
                    # Detected clue -> cyan/blue
                    cv2.putText(canvas, str(puzzle[r][c]), (cx, cy),
                                font, 1.1, (255, 180, 0), 2, cv2.LINE_AA)

        # Warp the canvas back to original perspective
        overlay_warped = cv2.warpPerspective(canvas, self.inv_M, (w, h))

        # Blend: use alpha for the grid lines, opaque for digits
        result = frame.copy()
        gray_overlay = cv2.cvtColor(overlay_warped, cv2.COLOR_BGR2GRAY)

        # Digits (bright pixels) — overlay opaquely
        digit_mask = gray_overlay > 60
        result[digit_mask] = overlay_warped[digit_mask]

        # Grid lines (dim pixels) — blend with alpha
        grid_mask = (gray_overlay > 10) & (~digit_mask)
        if np.any(grid_mask):
            result[grid_mask] = cv2.addWeighted(
                frame, 0.5, overlay_warped, 0.5, 0
            )[grid_mask]

        return result

    def draw_ocr_hud(self, frame: np.ndarray, puzzle: list[list[int]],
                     solved: bool = False) -> np.ndarray:
        """
        Draw a semi-transparent 2D grid HUD in the top-right corner
        showing what the OCR engine currently detects.
        """
        h, w = frame.shape[:2]

        # Larger HUD for better visibility
        box_size = 216  # 24 * 9
        margin = 15
        box_x0 = w - box_size - margin
        box_y0 = margin + 20  # space for label
        box_x1 = w - margin
        box_y1 = box_y0 + box_size

        # Clamp to frame bounds
        box_x0 = max(0, box_x0)
        box_y0 = max(0, box_y0)
        box_x1 = min(w, box_x1)
        box_y1 = min(h, box_y1)

        # Draw semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x0 - 5, box_y0 - 25), (box_x1 + 5, box_y1 + 5),
                      (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Draw HUD label
        clue_count = sum(1 for row in puzzle for v in row if v != 0)
        label = f"OCR ({clue_count} digits)"
        if solved:
            label = f"SOLVED ({clue_count} clues)"
        cv2.putText(
            frame, label, (box_x0, box_y0 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA
        )

        cell_px = box_size // 9
        # Draw grid lines
        for i in range(10):
            thick = 2 if i % 3 == 0 else 1
            color = (140, 140, 140) if i % 3 == 0 else (70, 70, 70)
            px = box_x0 + i * cell_px
            py = box_y0 + i * cell_px
            cv2.line(frame, (px, box_y0), (px, box_y0 + 9 * cell_px), color, thick)
            cv2.line(frame, (box_x0, py), (box_x0 + 9 * cell_px, py), color, thick)

        # Draw digits
        for r in range(9):
            for c in range(9):
                val = puzzle[r][c]
                if val != 0:
                    text = str(val)
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
                    tx = box_x0 + c * cell_px + (cell_px - text_size[0]) // 2
                    ty = box_y0 + r * cell_px + (cell_px + text_size[1]) // 2
                    cv2.putText(
                        frame, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (50, 230, 255), 1, cv2.LINE_AA  # Warm yellow
                    )
        return frame

    # ── Detection internals ──────────────────────────────────────────────────

    def _find_grid_contour(self, gray: np.ndarray) -> np.ndarray | None:
        """
        Return the 4-point approximation of the largest valid quadrilateral.
        Uses morphological operations and multi-pass validation.
        """
        # Pre-process: blur + adaptive threshold
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            19, 3,  # larger block size + higher C for better noise handling
        )

        # Morphological closing to connect broken grid lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        thresh = cv2.erode(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Sort by area descending
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Try multiple epsilon values for polygon approximation
        for cnt in contours[:8]:
            for eps_factor in (0.015, 0.02, 0.03, 0.04, 0.05):
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, eps_factor * peri, True)

                if len(approx) != 4:
                    continue

                area = cv2.contourArea(approx)
                if area < 5_000:
                    continue

                pts = approx.reshape(4, 2).astype("float32")

                # Convexity check
                if not cv2.isContourConvex(approx):
                    continue

                # Aspect ratio check: the quadrilateral should be roughly square
                ordered = order_points(pts)
                w1 = np.linalg.norm(ordered[1] - ordered[0])
                w2 = np.linalg.norm(ordered[2] - ordered[3])
                h1 = np.linalg.norm(ordered[3] - ordered[0])
                h2 = np.linalg.norm(ordered[2] - ordered[1])
                avg_w = (w1 + w2) / 2
                avg_h = (h1 + h2) / 2
                if avg_w < 1 or avg_h < 1:
                    continue
                ratio = avg_w / avg_h
                if ratio < 0.4 or ratio > 2.5:
                    continue

                # Area ratio check: contour area vs bounding rect area
                # A real grid should fill most of its bounding box
                bounding_area = avg_w * avg_h
                fill_ratio = area / bounding_area if bounding_area > 0 else 0
                if fill_ratio < 0.5:
                    continue

                return pts

        return None

    def _extract_cells(self, warped_gray: np.ndarray) -> list[list[np.ndarray]]:
        """
        Divide the warped grid into 81 cells.
        Returns a 9x9 list of CELL_SIZE x CELL_SIZE grayscale images.
        """
        cells = []
        step = self.WARP_SIZE // 9

        # Blur slightly before thresholding to reduce noise
        blurred = cv2.GaussianBlur(warped_gray, (3, 3), 0)

        # Re-threshold for cleaner cells
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 3,
        )

        for r in range(9):
            row_cells = []
            for c in range(9):
                y0, y1 = r * step, (r + 1) * step
                x0, x1 = c * step, (c + 1) * step
                cell = thresh[y0:y1, x0:x1]
                # Larger margin to better crop out grid lines
                margin = step // 6
                cell = cell[margin:-margin, margin:-margin]
                cell = cv2.resize(cell, (self.CELL_SIZE, self.CELL_SIZE))
                row_cells.append(cell)
            cells.append(row_cells)

        return cells

    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        """Draw detected corners on frame for debugging."""
        if self.corners is None:
            return frame
        out = frame.copy()
        pts = self.corners.astype(int)
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)
        for pt in pts:
            cv2.circle(out, tuple(pt), 5, (0, 0, 255), -1)
        return out
