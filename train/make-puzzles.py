import torch
import time # For basic timing comparison

# Define board dimensions (optional, but good practice)
ROWS = 6
COLS = 7

@torch.jit.script
def make_move(board: torch.Tensor, column: int) -> tuple[torch.Tensor, int]:
    """
    Places a piece on the board for the given player in the specified column.

    Args:
        board: The current board state tensor (rows x cols), dtype=torch.int8.
               0=empty, 1=player 1, -1=player 2.
        column: The column index (0 to cols-1) where the player wants to move.
        player: The player identifier (+1 or -1). Here assumed to be +1 as per user request.

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


@torch.jit.script
def check_win_after_move(board: torch.Tensor, r: int, c: int) -> bool:
    """
    Checks if the move made by 'player' at (r, c) resulted in a win.
    Optimized to only check lines passing through the last placed piece.

    Args:
        board: The board state tensor *after* the move has been made.
        player: The player identifier (+1 or -1) who just made the move.
        r: The row index of the last placed piece.
        c: The column index of the last placed piece.

    Returns:
        True if the move resulted in a win for 'player', False otherwise.
    """
    rows, cols = board.shape
    target = 1  # The piece we are looking for runs of

    # --- Check Horizontal ---
    count = 1 # Start with the piece just placed
    # Check left
    for i in range(1, 4):
        cc = c - i
        if cc >= 0 and board[r, cc] == target:
            count += 1
        else:
            break # Stop counting in this direction
    # Check right
    for i in range(1, 4):
        cc = c + i
        if cc < cols and board[r, cc] == target:
            count += 1
        else:
            break
    if count >= 4: return True

    # --- Check Vertical ---
    # Only need to check downwards as piece is placed at lowest point
    count = 1
    for i in range(1, 4):
        rr = r + i
        if rr < rows and board[rr, c] == target:
            count += 1
        else:
            break
    if count >= 4: return True

    # --- Check Diagonal (Positive Slope / ) ---
    count = 1
    # Check down-left
    for i in range(1, 4):
        rr, cc = r + i, c - i
        if rr < rows and cc >= 0 and board[rr, cc] == target:
            count += 1
        else:
            break
    # Check up-right
    for i in range(1, 4):
        rr, cc = r - i, c + i
        if rr >= 0 and cc < cols and board[rr, cc] == target:
            count += 1
        else:
            break
    if count >= 4: return True

    # --- Check Diagonal (Negative Slope \ ) ---
    count = 1
    # Check down-right
    for i in range(1, 4):
        rr, cc = r + i, c + i
        if rr < rows and cc < cols and board[rr, cc] == target:
            count += 1
        else:
            break
    # Check up-left
    for i in range(1, 4):
        rr, cc = r - i, c - i
        if rr >= 0 and cc >= 0 and board[rr, cc] == target:
            count += 1
        else:
            break
    if count >= 4: return True

    # No win found
    return False


def make_move_and_check(board, c):
    new_board, r = make_move(board, c)
    return new_board, check_win_after_move(new_board, r, c)


def print_board(b):
    b = b.cpu().numpy()
    symbols = {0: ' ', 1: 'X', -1: 'O'}
    for i in range(b.shape[0]):
        print('|' + ''.join(symbols[c] for c in b[i]) + '|')

def random_move(board):
    valid_moves = torch.nonzero(board[0, :] == 0)
    choice = torch.randint(len(valid_moves), (1,)).item()
    return valid_moves[choice].item()

def find_winning_moves(board):
    wins = []
    for c in range(board.shape[1]):
        if board[0, c] == 0:
            _, win = make_move_and_check(board, c)
            if win:
                wins.append(c)
    return wins

def multi_hot(moves, cols):
    output = torch.zeros((cols,), dtype=torch.int8)
    output[torch.tensor(moves)] = 1
    return output

def play():
    board = torch.zeros((ROWS, COLS), dtype=torch.int8)
    player = 0
    
    # collect all tactical puzzles found during play
    puzzles = []

    while True:
        win_moves = find_winning_moves(board)
        if win_moves:
            # found at least one winning move
            puzzles.append((board, multi_hot(win_moves, COLS), 1))
            
            # play a random winning move
            #print('O' if player else 'X', 'Winning:', multi_hot(win_moves, COLS))
            choice = torch.randint(len(win_moves), (1,)).item()
            move = win_moves[choice]
        else:
            block_moves = find_winning_moves(-board)
            if block_moves:
                # valid puzzles are only those where there is a single thread to block
                if len(block_moves) == 1:
                    puzzles.append((board, multi_hot(block_moves, COLS), 0))
                    
                #print('O' if player else 'X', 'Blocking:', multi_hot(block_moves, COLS))
                choice = torch.randint(len(block_moves), (1,)).item()
                move = block_moves[choice]
            else:
                # neither wins nor blocks available, play random move
                move = random_move(board)
        
        board, win = make_move_and_check(board, move)
        #print_board(board if player==0 else -board)
        #print()
        
        draw = torch.all(board[0, :] != 0)
        
        if win or draw:
            return puzzles
            
        board = -board
        player = 1 - player

def remove_duplicates(puzzles):
    used = set()
    result = []
    for p in puzzles:
        h = hash(tuple(p[0].cpu().numpy().ravel()))
        if h not in used:
            used.add(h)
            result.append(p)
    return result

def collect_puzzles():
    puzzles = []
    while len(puzzles) < 10000:
        puzzles.extend(play())
        print(f'\r{len(puzzles)} puzzles  ', end='')
    puzzles = remove_duplicates(puzzles)
    print(f'\n{len(puzzles)} puzzles after deduplication')

    boards = torch.stack(tuple(p[0] for p in puzzles))
    moves = torch.stack(tuple(p[1] for p in puzzles))
    goals = torch.tensor(tuple(p[2] for p in puzzles), dtype=torch.int8)
    torch.save((boards, moves, goals), 'puzzles.dat')
    
if __name__ == '__main__':
    #collect_puzzles()
    pass