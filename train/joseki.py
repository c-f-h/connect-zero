import sys
import torch
import numpy as np
from collections import Counter
import click

from globals import ROWS, COLS, init_device, get_device
from model import load_frozen_model
from board import pretty_print_board, format_board
from play import play_parallel_to_depth

def board_to_hashable(board: torch.Tensor) -> bytes:
    # Convert a board tensor to a hashable bytes object for counting
    return board.cpu().numpy().tobytes()

@click.command()
@click.argument('model_names', nargs=-1)
@click.option('-n', '--num-games', default=100, help='Number of games to play.')
@click.option('-d', '--depth', default=5, help='Number of moves to play (depth).')
@click.option("--cuda", is_flag=True, help="Use CUDA if available.")
def main(model_names, num_games, depth, cuda):
    device = init_device(cuda)
    if len(model_names) == 1:
        model_names = [model_names[0], model_names[0]]
    elif len(model_names) != 2:
        print("Please provide one or two model names.")
        return
    model1 = load_frozen_model(model_names[0]).to(device)
    model2 = load_frozen_model(model_names[1]).to(device)

    boards = play_parallel_to_depth(model1, model2, num_games, depth)
    if depth % 2 == 1:
        boards = -boards
    # Flatten boards to hashable representations
    hashes = [board_to_hashable(b) for b in boards]
    counter = Counter(hashes)

    # Sort by frequency, descending
    most_common = counter.most_common()

    print(f"Top {len(most_common)} resulting states after {depth} moves (out of {num_games} games):")
    total = 0
    cutoff = int(num_games * 0.95)

    # Prepare formatted boards and frequencies
    formatted = []
    for h, freq in most_common:
        arr = np.frombuffer(h, dtype=np.int8).reshape(ROWS, COLS)
        board_tensor = torch.tensor(arr, dtype=torch.int8)
        board_lines = format_board(board_tensor)
        formatted.append((board_lines, freq))

    # Print up to 4 boards per row, side by side
    max_per_row = 6
    i = 0
    shown = 0
    while i < len(formatted) and total < cutoff:
        row_boards = formatted[i:i+max_per_row]
        max_lines = max(len(b[0]) for b in row_boards)
        # Pad boards to same height
        padded_boards = []
        for lines, freq in row_boards:
            pad = [' ' * len(lines[0])] * (max_lines - len(lines))
            padded_boards.append(lines + pad)
        # Print boards line by line
        for l in range(max_lines):
            print("  ".join(padded_boards[j][l] for j in range(len(row_boards))))
        # Print frequency labels below the boards, aligned
        label_line = "  ".join(f"{row_boards[j][1]}/{num_games}".center(len(padded_boards[j][0])) for j in range(len(row_boards)))
        print(label_line)
        print()
        # Update counters
        for _, freq in row_boards:
            total += freq
            shown += 1
            if total >= cutoff:
                break
        i += max_per_row

if __name__ == "__main__":
    main()
