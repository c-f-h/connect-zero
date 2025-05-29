import torch
import torch.nn as nn

from board import make_move_and_check_batch
from play import sample_moves

def multiple_rollouts(boards: torch.Tensor, model: nn.Module, depth: int):
    pass

def estimate_move_values_from_rollout(board: torch.Tensor, model: nn.Module, width: int, depth: int) -> torch.Tensor:
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
    model.eval()
    with torch.no_grad():
        device = board.device
        valid_moves = torch.where(board[0] == 0)[0]  # Get valid moves (columns that are not full)
        n_moves = valid_moves.shape[0]

        initial_boards = board.unsqueeze(0).repeat(n_moves, 1, 1)  # (n_moves, ROWS, COLS)
        initial_boards, wins, draws = make_move_and_check_batch(initial_boards, valid_moves)

        initial_values = torch.full((n_moves,), 0.0, dtype=torch.float32, device=device)
        initial_values[wins] = 1.0

        # Repeat the boards after the initial move for each rollout width
        # Note: board is already flipped for the next player
        boards = (-initial_boards).repeat_interleave(width, dim=0) # (n_moves*width, ROWS, COLS)
        finished = (wins | draws).repeat_interleave(width, dim=0)  # (n_moves*width,)
        values = initial_values.repeat_interleave(width)           # (n_moves*width,)

        # ply = 0 is opponent's move, ply = 1 is our move, and so on
        for ply in range(depth):
            # Only continue unfinished games
            active = ~finished
            if not active.any():
                break
            # Sample and play moves for all active boards
            moves = sample_moves(model, boards[active])
            next_boards, wins, draws = make_move_and_check_batch(boards[active], moves)
            boards[active] = -next_boards       # Flip the board for the next player

            # Update finished mask and values
            just_finished = wins | draws
            idxs = torch.where(active)[0]           # get global indices of active games
            for i_global, i_act in zip(idxs[just_finished], torch.where(just_finished)[0]):
                finished[i_global] = True
                if wins[i_act]:
                    # Current player wins: +1 or -1, depending on the ply
                    values[i_global] = 1.0 if (ply % 2 == 1) else -1.0
                # no need to update values for draws, they remain 0.0

        # For unfinished games, use value head to estimate value of final position
        unfinished = ~finished
        if unfinished.any():
            sign = 1.0 if (depth % 2 == 1) else -1.0  # Sign depends on whether we are at our turn
            _, v = model(boards[unfinished])
            values[unfinished] = sign * v

        # Average values for each move
        valid_move_values = values.view((n_moves, width)).mean(dim=1)  # (n_moves,)

        all_values = torch.full((board.shape[1],), -1e12, device=device)
        all_values[valid_moves] = valid_move_values
        return all_values
