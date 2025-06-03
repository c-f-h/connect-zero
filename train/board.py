### Functions for board state handling (making moves, checking win condition). ###

import torch
import torch.optim as optim

@torch.jit.script
def make_move(board: torch.Tensor, column: int) -> tuple[torch.Tensor, int]:
    """
    Places a piece on the board for the player 1 in the specified column.

    Args:
        board: The current board state tensor (rows x cols), dtype=torch.int8.
               0=empty, 1=player 1, -1=player 2.
        column: The column index (0 to cols-1) where the player wants to move.

    Returns:
        A tuple containing:
        - The new board state tensor with the piece added.
        - The row index where the piece was placed.

    Raises:
        ValueError: If the chosen column is full or invalid.
    """
    rows, cols = board.shape
    if not 0 <= column < cols:
        raise ValueError(f"Invalid column index: {column}. Must be between 0 and {cols-1}.")

    # Find the lowest empty row (iterate from bottom up)
    for r in range(rows - 1, -1, -1):
        if board[r, column] == 0:
            # Found an empty spot, place the piece
            # Note: Creates a copy to avoid modifying the original tensor directly
            # If in-place modification is desired and safe in your RL loop,
            # you could remove the .clone()
            new_board = board.clone()
            new_board[r, column] = 1
            return new_board, r # Return the modified board and the row

    # If the loop finishes, the column is full
    raise ValueError(f"Column {column} is full.")


def make_moves_batch(boards: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
    """Makes moves on a batch of boards for player 1."""
    num_games = boards.shape[0]
    if num_games == 0:
        return boards

    # Find the lowest empty row for each move
    batch_indices = torch.arange(num_games, device=boards.device)

    # Count pieces in each target column for each board
    pieces_in_col = (boards != 0).sum(dim=1)[batch_indices, moves]
    row_indices = boards.shape[1] - 1 - pieces_in_col       # for each board/move, in which row it will end up

    if (row_indices < 0).any():
        raise ValueError("Attempted move on a full column in make_moves_batch!")

    # Place the pieces
    boards = boards.clone()  # don't modify the original tensor
    boards[batch_indices, row_indices, moves] = 1
    return boards


def is_board_full_batch(boards: torch.Tensor) -> torch.Tensor:
    """For a batch of boards, return a bool tensor indicating which ones of them are completely filled up."""
    first_row_filled = (boards[:, 0, :] != 0)
    return torch.all(first_row_filled, dim=1).bool()


@torch.jit.script
def check_win_after_move(board: torch.Tensor, r: int, c: int) -> bool:
    """
    Checks if the move made by player 1 at (r, c) resulted in a win.
    Optimized to only check lines passing through the last placed piece.

    Args:
        board: The board state tensor *after* the move has been made.
        r: The row index of the last placed piece.
        c: The column index of the last placed piece.

    Returns:
        True if the move resulted in a win for 'player', False otherwise.
    """
    rows, cols = board.shape

    # --- Check Horizontal ---
    count = 1 # Start with the piece just placed
    # Check left
    for i in range(1, 4):
        cc = c - i
        if cc >= 0 and board[r, cc] == 1:
            count += 1
        else:
            break # Stop counting in this direction
    # Check right
    for i in range(1, 4):
        cc = c + i
        if cc < cols and board[r, cc] == 1:
            count += 1
        else:
            break
    if count >= 4: return True

    # --- Check Vertical ---
    # Only need to check downwards as piece is placed at lowest point
    count = 1
    for i in range(1, 4):
        rr = r + i
        if rr < rows and board[rr, c] == 1:
            count += 1
        else:
            break
    if count >= 4: return True

    # --- Check Diagonal (Positive Slope / ) ---
    count = 1
    # Check down-left
    for i in range(1, 4):
        rr, cc = r + i, c - i
        if rr < rows and cc >= 0 and board[rr, cc] == 1:
            count += 1
        else:
            break
    # Check up-right
    for i in range(1, 4):
        rr, cc = r - i, c + i
        if rr >= 0 and cc < cols and board[rr, cc] == 1:
            count += 1
        else:
            break
    if count >= 4: return True

    # --- Check Diagonal (Negative Slope \ ) ---
    count = 1
    # Check down-right
    for i in range(1, 4):
        rr, cc = r + i, c + i
        if rr < rows and cc < cols and board[rr, cc] == 1:
            count += 1
        else:
            break
    # Check up-left
    for i in range(1, 4):
        rr, cc = r - i, c - i
        if rr >= 0 and cc >= 0 and board[rr, cc] == 1:
            count += 1
        else:
            break
    if count >= 4: return True

    # No win found
    return False

@torch.jit.script
def make_move_and_check(board: torch.Tensor, column: int) -> tuple[torch.Tensor, bool]:
    """
    Make a move for player 1 and check if it results in a win.

    Args:
        board: The current board state tensor (rows x cols), dtype=torch.int8.
               0=empty, 1=player 1, -1=player 2.
        column: The column index (0 to cols-1) where the player wants to move.

    Returns:
        A tuple containing:
        - The new board state tensor with the piece added.
        - A boolean indicating if the move resulted in a win.
    """
    new_board, r = make_move(board, column)
    win = check_win_after_move(new_board, r, column)
    return new_board, win

g_win_conv_kernel = None

def check_win_batch_conv(boards: torch.Tensor) -> torch.Tensor:
    """
    Check if the moves made by player 1 in a batch of games resulted in wins.

    Returns a bool tensor.
    """
    global g_win_conv_kernel
    if g_win_conv_kernel is None:
        g_win_conv_kernel = torch.zeros(4, 1, 4, 4, dtype=torch.int8, device=boards.device)
        g_win_conv_kernel[0, 0, 1, :] = 1    # horizontal
        g_win_conv_kernel[1, 0, :, 1] = 1    # vertical
        g_win_conv_kernel[2, 0, ...] = torch.eye(4, dtype=torch.int8)             # diagonal \
        g_win_conv_kernel[3, 0, ...] = torch.eye(4, dtype=torch.int8).fliplr()    # diagonal /
        if boards.is_cuda:
            # PyTorch doesn't seem to support int8 convolution on CUDA
            g_win_conv_kernel = g_win_conv_kernel.float().to(boards.device)
    # convert boards to 1 channel (B, 1, R, C) and convolve with four output channels
    if boards.is_cuda:
        result = torch.nn.functional.conv2d(boards.unsqueeze(1).float(), g_win_conv_kernel, padding=2)
        return torch.amax(result, dim=(1, 2, 3)) >= 3.99        # (B,) - just being paranoid about rounding errors
    else:
        result = torch.nn.functional.conv2d(boards.unsqueeze(1), g_win_conv_kernel, padding=2)
        return torch.amax(result, dim=(1, 2, 3)) == 4           # (B,)


def make_move_and_check_batch(boards: torch.Tensor, moves: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    new_boards = make_moves_batch(boards, moves)
    wins = check_win_batch_conv(new_boards)
    # check for draws among the games that are not won
    iact = torch.where(wins == 0)[0]
    draws = torch.zeros(boards.shape[0], dtype=torch.bool, device=boards.device)
    draws[iact] = is_board_full_batch(new_boards[iact])
    return new_boards, wins, draws

def pretty_print_board(board: torch.Tensor, indent=0):
    """Pretty print the board."""
    board_np = board.cpu().numpy()  # Convert to numpy for easier printing
    symbols = {0: " ", 1: "X", -1: "O"}  # Define symbols for empty, player 1, and player 2
    indentstr = indent * ' '
    for row in range(board_np.shape[0]):
        row_str = " ".join(symbols[board_np[row, col]] for col in range(board_np.shape[1]))
        print(indentstr + f"│{row_str}│")
    print(indentstr + "└─" + "┴─"  * (board_np.shape[1] - 1) + "┘")

def string_to_board_test_format(s):
    # This version expects s to be a single string concatenating all rows,
    # e.g., "XOXOXOXOXOXOXO..." (42 chars for a 6x7 board)
    # where each char is 'X', 'O', or ' '.
    pieces = [c for c in s if c in ' XO'] # Kept for robustness, though 's' should be clean.
    if len(pieces) != 42:
        raise ValueError(f"Input string does not yield 42 pieces after filtering. Got {len(pieces)} pieces. Input: '{s[:100]}...'")

    s2 = []
    # The characters in 'pieces' are the actual pieces for the board, row by row.
    # No need for slicing like s[0:13:2] which assumed original spacing.
    # We just need to convert them.
    # The loop structure was incorrect for a pre-filtered list of pieces.
    # s2 should just be 'pieces'.
    charmap = {' ': 0, 'X': 1, 'O': -1}
    # Ensure all characters in 'pieces' are valid before mapping
    for p_char in pieces:
        if p_char not in charmap:
            raise ValueError(f"Invalid character '{p_char}' in input string pieces. Valid are 'X', 'O', ' '.")

    board_pieces = [charmap[c] for c in pieces]
    return torch.tensor(board_pieces, dtype=torch.int8).reshape(6, 7)

# Original string_to_board, designed to parse pretty_print_board style strings
def string_to_board(s):
    s_filtered = [c for c in s if c in ' XO'] # Use a different variable name to avoid confusion
    s2 = []
    # This logic assumes 's_filtered' still contains pieces grouped by lines of 13 (7 pieces + 6 spaces)
    # This was the original bug if s_filtered was flat.
    # The original intent seems to be that 's' (unfiltered) is passed to the loop logic.
    # Let's re-evaluate: The original function took the raw string from pretty_print_board (with | etc.)
    # then filtered it to 'X', 'O', ' '. This flat list was then processed by the s[0:13:2] logic.
    # This implies the input string 's' to this function is one line of the pretty_print_board
    # or a structure where this slicing makes sense.
    # If 's' is the *entire* multi-line pretty_print string:
    # Example line: "│  O   O      │" -> filtered: [' ', ' ', 'O', ' ', ' ', ' ', 'O', ' ', ' ', ' ', ' ', ' ']
    # s[0:13:2] would take from this filtered list of one line.
    # The original main block did: `boardstrings = [''.join(lines[i:i+7]) ...]`
    # `b = string_to_board(boardstrings[0])` -> `s` is a 7-line string block.
    # So, the initial filter `s = [c for c in s if c in ' XO']` makes a flat list of ALL X,O,spaces from 7 lines.
    # Then `s2.extend(s[0:13:2]); s = s[13:]` processes this flat list.
    # This means it expects 13 characters *from the already filtered flat list* for the first row's pieces,
    # which is 7 pieces. This is correct if the original pretty_print output for a row has 7 pieces and 6 spaces.
    # e.g. " X O X O X O " (13 chars). Filtered is the same. s[0:13:2] gets X,X,X,X.
    # This is the "every other character" logic.

    # Reverting to the exact original code provided in prompt for this function:
    s_processed = [c for c in s if c in ' XO'] # Renamed to avoid modifying 's' if it's used later by mistake
    s_for_slicing = s_processed # Use the filtered list for slicing.
    for i in range(6): # 6 rows
        s2.extend(s_for_slicing[0:13:2]) # Get 7 pieces by taking every other char from a 13-char segment
        s_for_slicing = s_for_slicing[13:] # Advance to the next 13-char segment
    charmap = {' ': 0, 'X': 1, 'O': -1}
    return torch.tensor([charmap[c] for c in s2], dtype=torch.int8).reshape(6, 7)

def print_board_info(model, board):
    """
    Print the board state and the model's output.
    """
    pretty_print_board(board)
    with torch.no_grad():
        logits, value = model(board)
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        print("Probs:", " ".join([f"{p:.3f}" for p in probs]))
        print("Value:", value.item())

if __name__ == '__main__':
    import sys
    from model import load_frozen_model

    boardstrings = r"""
│             │
│             │
│  O   O      │
│  O   X X O  │
│  X   O X O  │
│  X O X O X X│
└─┴─┴─┴─┴─┴─┴─┘
│             │
│             │
│X O          │
│X X          │
│O X O O      │
│X X O X O O  │
└─┴─┴─┴─┴─┴─┴─┘
│             │
│             │
│             │
│             │
│    O        │
│    X X   O  │
└─┴─┴─┴─┴─┴─┴─┘
│             │
│             │
│             │
│  X          │
│  X   O      │
│  X O X   O O│
└─┴─┴─┴─┴─┴─┴─┘
│             │
│    X        │
│    X        │
│    X   O O  │
│    O X O X O│
│    X O X X O│
└─┴─┴─┴─┴─┴─┴─┘
│             │
│             │
│             │
│    O        │
│O X O       X│
│O X O X X X O│
└─┴─┴─┴─┴─┴─┴─┘
│             │
│             │
│      O      │
│      X   X  │
│    O X O X  │
│O X X O O X O│
└─┴─┴─┴─┴─┴─┴─┘
│             │
│      X      │
│    X X O    │
│  X O O X O  │
│  O X X O O  │
│O X O X X O  │
└─┴─┴─┴─┴─┴─┴─┘
"""

    lines = boardstrings.strip().split('\n')
    boardstrings = [''.join(lines[i:i+7]) for i in range(0, len(lines), 7)]
    
    b = string_to_board(boardstrings[0]) # This should use the original pretty_print_board parser

    model = load_frozen_model(sys.argv[1])
    print_board_info(model, b)
    print_board_info(model, -b)
