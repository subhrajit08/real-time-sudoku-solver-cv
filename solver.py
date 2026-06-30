import time

def is_valid(board: list[list[int]], row: int, col: int, num: int) -> bool:
    # Row check
    if num in board[row]:
        return False
    # Column check
    if any(board[r][col] == num for r in range(9)):
        return False
    # 3×3 box check
    br, bc = 3 * (row // 3), 3 * (col // 3)
    for r in range(br, br + 3):
        for c in range(bc, bc + 3):
            if board[r][c] == num:
                return False
    return True


def is_initial_board_valid(board: list[list[int]]) -> bool:
    """Check if the starting board configuration has any conflicts (ignoring 0s)."""
    # Rows
    for r in range(9):
        seen = set()
        for val in board[r]:
            if val != 0:
                if val in seen:
                    return False
                seen.add(val)
    # Columns
    for c in range(9):
        seen = set()
        for r in range(9):
            val = board[r][c]
            if val != 0:
                if val in seen:
                    return False
                seen.add(val)
    # Boxes
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            seen = set()
            for r in range(box_row, box_row + 3):
                for c in range(box_col, box_col + 3):
                    val = board[r][c]
                    if val != 0:
                        if val in seen:
                            return False
                        seen.add(val)
    return True


def find_best_empty_cell(board: list[list[int]]) -> tuple[int, int, list[int]] | None:
    """Find the empty cell with the fewest valid candidates (MRV heuristic)."""
    best_cell = None
    min_candidates = 10
    best_candidates = []
    
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                candidates = []
                for num in range(1, 10):
                    if is_valid(board, r, c, num):
                        candidates.append(num)
                num_candidates = len(candidates)
                if num_candidates == 0:
                    # Dead end: empty cell with no possible values
                    return (r, c, [])
                if num_candidates < min_candidates:
                    min_candidates = num_candidates
                    best_cell = (r, c)
                    best_candidates = candidates
                    if min_candidates == 1:
                        return r, c, candidates
    if best_cell is None:
        return None
    return best_cell[0], best_cell[1], best_candidates


def solve(board: list[list[int]], max_steps: int = 5000) -> bool:
    """
    Recursively fill empty cells using backtracking and the MRV heuristic.
    Mutates `board` in-place. Returns True when solved.
    """
    if not is_initial_board_valid(board):
        return False

    steps = 0
    
    def backtrack() -> bool:
        nonlocal steps
        steps += 1
        if steps > max_steps:
            return False  # Prevent thread freeze if limit is exceeded
            
        next_cell = find_best_empty_cell(board)
        if next_cell is None:
            return True  # Solved (no empty cells left)
            
        row, col, candidates = next_cell
        if not candidates:
            return False  # Dead end
            
        for num in candidates:
            board[row][col] = num
            if backtrack():
                return True
            board[row][col] = 0  # Backtrack
            
        return False

    return backtrack()


def solve_timed(board: list[list[int]]) -> tuple[bool, float]:
    """Solve board and return (success, elapsed_seconds)."""
    start = time.time()
    success = solve(board)
    return success, time.time() - start


def print_board(board: list[list[int]]) -> None:
    """Pretty-print the Sudoku board to stdout."""
    for i, row in enumerate(board):
        if i % 3 == 0:
            print("+-------+-------+-------+")
        line = ""
        for j, val in enumerate(row):
            if j % 3 == 0:
                line += "| "
            line += (str(val) if val != 0 else " ") + " "
        print(line + "|")
    print("+-------+-------+-------+")


if __name__ == "__main__":
    puzzle = [
        [0, 0, 0, 8, 0, 0, 3, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 8, 0, 6, 7, 0, 0, 0, 0],
        [0, 0, 3, 0, 8, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 4, 7, 0],
        [0, 0, 1, 2, 3, 0, 6, 0, 0],
        [0, 6, 0, 0, 0, 9, 0, 0, 0],
        [9, 4, 0, 5, 0, 3, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 5, 2],
    ]
    print("Puzzle:")
    print_board(puzzle)
    ok, elapsed = solve_timed(puzzle)
    print(f"\nSolved in {elapsed:.3f}s  (success={ok})")
    print_board(puzzle)
