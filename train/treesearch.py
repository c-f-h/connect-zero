import torch
import torch.nn as nn

from board import make_move_and_check_batch
from play import sample_moves

def multiple_rollouts(initial_boards: torch.Tensor, model: nn.Module, width: int, depth: int, temperature: float = 1.0) -> torch.Tensor:
    from main import REWARD_DISCOUNT

    model.eval()
    with torch.no_grad():
        device = initial_boards.device
        n_inputs = initial_boards.shape[0]
        valid_moves = (initial_boards[:, 0] == 0)
        n_moves = valid_moves.sum(dim=1)        # Number of valid moves for each board

        # Compute 1D tensor for the valid moves per board
        moves = torch.where(valid_moves)[1]
        n_boardmoves = moves.shape[0]  # Total number of valid moves across all boards

        initial_boards = initial_boards.repeat_interleave(n_moves, dim=0)  # (n_boardmoves, R, C) - repeat each board for each valid move
        initial_boards, wins, draws = make_move_and_check_batch(initial_boards, moves)

        initial_values = torch.full((n_boardmoves,), 0.0, dtype=torch.float32, device=device)  # (n_boardmoves,)
        initial_values[wins] = 1.0

        # Repeat the boards after the initial move for each rollout width
        # Note: board is already flipped for the next player
        boards = (-initial_boards).repeat_interleave(width, dim=0) # (n_boardmoves*width, ROWS, COLS)
        finished = (wins | draws).repeat_interleave(width, dim=0)  # (n_boardmoves*width,)
        values = initial_values.repeat_interleave(width)           # (n_boardmoves*width,)

        actidxs = torch.where(~finished)[0]  # Global indices of active games

        # ply = 0 is opponent's move, ply = 1 is our move, and so on
        for ply in range(depth):
            # Only continue unfinished games
            if actidxs.numel() == 0:
                break
            # Sample and play moves for all active boards
            moves = sample_moves(model, boards[actidxs], temperature=temperature)
            next_boards, wins, draws = make_move_and_check_batch(boards[actidxs], moves)
            boards[actidxs] = -next_boards       # Flip the board for the next player

            # Update finished mask and values
            just_finished = wins | draws
            finished[actidxs[just_finished]] = True  # Update finished for the global indices
            values[actidxs[wins]] = (REWARD_DISCOUNT**(ply + 1)) * (1.0 if (ply % 2 == 1) else -1.0)  # Update values for winning moves

            actidxs = actidxs[~just_finished]  # Keep only active games for the next iteration

        # For unfinished games, use value head to estimate value of final position
        unfinished = ~finished
        if unfinished.any():
            factor = (REWARD_DISCOUNT**(depth + 1)) * (1.0 if (depth % 2 == 1) else -1.0)  # Sign depends on whether we are at our turn
            _, v = model(boards[unfinished])
            values[unfinished] = factor * v

        # Average values for each move
        valid_move_values = values.view((n_boardmoves, width)).mean(dim=1)  # (n_boardmoves,)

        all_values = torch.full((n_inputs, boards.shape[2]), -1e12, device=device)
        all_values[valid_moves] = valid_move_values
        return all_values


def estimate_move_values_from_rollout(board: torch.Tensor, model: nn.Module, width: int, depth: int, temperature: float = 1.0) -> torch.Tensor:
    """
    Estimate the value of each move by simulating rollouts using the model's policy and value head.

    Args:
        board: (ROWS, COLS) tensor, the current board state.
        model: policy/value model.
        width: number of rollouts per move.
        depth: number of plies to simulate before bootstrapping with value head.

    Returns:
        Tensor with estimated values for each move.
    """
    values = multiple_rollouts(board.unsqueeze(0), model, width, depth, temperature=temperature)
    return values.squeeze(0)  # (COLS,)


if __name__ == '__main__':
    from board import string_to_board, pretty_print_board
    from model import load_frozen_model, RolloutModel

    boardstrings = r"""
│  X   O O    │
│O O   X O    │
│X X   X O    │
│X O   O X    │
│X O X O X   X│
│O X O X O X O│
└─┴─┴─┴─┴─┴─┴─┘
│             │
│  O O     O  │
│  X O   X X  │
│  X X   O X  │
│  X X O X O  │
│O O O X O X  │
└─┴─┴─┴─┴─┴─┴─┘
"""

    lines = boardstrings.strip().split('\n')
    boardstrings = [''.join(lines[i:i+7]) for i in range(0, len(lines), 7)]

    b = string_to_board(boardstrings[1])
    pretty_print_board(b)

    #model = RolloutModel(load_frozen_model('CNN-Mk4:mk4-ts1.pth'), width=4, depth=4)
    basemodel = load_frozen_model('CNN-Mk4:selfsolo/0104.pth')
    model = RolloutModel(basemodel, width=4, depth=4)
    with torch.no_grad():
        logits, value = model(b)
        print('Rollout model:')
        print(logits)
        print(value)

        logits, value = basemodel(b)
        print('\nBase model:')
        print(logits)
        print(value)
