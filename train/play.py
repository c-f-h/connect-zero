import torch
import torch.nn.functional as F
import numpy as np

from globals import ROWS, COLS, init_device, get_device
from board import make_move_and_check, make_move_and_check_batch, pretty_print_board


def sample_move(model, board: torch.Tensor, epsilon: float = 0.0, output_probs: bool = False) -> int:
    """Sample a move using the model's output logits."""
    value = None
    if epsilon > 0 and torch.rand(1).item() < epsilon:
        # Epsilon-greedy strategy: choose a random move with probability epsilon
        logits = torch.zeros((COLS,), dtype=torch.float32, device=get_device())
    else:
        logits = model(board)
        if isinstance(logits, tuple):
            logits, value = logits        # model could return (policy, value) or just policy
    illegal_moves = torch.where(board[0, :] == 0, 0.0, -torch.inf)
    logits += illegal_moves                     # Mask out illegal moves
    probs = F.softmax(logits, dim=-1)           # Convert logits to probabilities
    if output_probs:
        np.set_printoptions(precision=3)
        p = probs.cpu().numpy()
        entropy = -(p * np.log(p + 1e-9)).sum()
        print(f"Move probabilities: {p}, Entropy: {entropy:.4f}")
        if value is not None:
            print(f"Board state value estimate: {value.item()}")
    move = torch.multinomial(probs, 1).item()
    return move

def sample_moves(model, boards: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Sample moves for a batch of boards using the model's output logits.

    Returns:
        A tensor of sampled moves, shape (N,).
    """
    num_games = boards.shape[0]
    if num_games == 0:
        return torch.empty(0, dtype=torch.long, device=get_device())

    # Get model logits
    value = None
    logits = model(boards)
    if isinstance(logits, tuple):
        logits, value = logits  # model could return (policy, value) or just policy
    logits /= temperature

    # Mask out illegal moves (columns that are full)
    illegal_moves_mask = (boards[:, 0, :] != 0)
    logits[illegal_moves_mask] = -torch.inf

    probs = F.softmax(logits, dim=-1)
    moves = torch.multinomial(probs, 1).squeeze(-1)
    return moves


def play(model1, model2, output: bool = False) -> int:
    """Have two models play against each other."""
    model1.eval()
    model2.eval()
    winner = 1
    moves = []
    with torch.no_grad():
        board = torch.zeros((ROWS, COLS), dtype=torch.int8, device=get_device())

        while True:
            if output:
                pretty_print_board(board if winner == 1 else -board, indent = (winner - 1) * 20)

            move = sample_move(model1, board, output_probs=output)
            moves.append(move)

            if output:
                print(f"Model {winner} plays: {move}")

            board, win = make_move_and_check(board, move)

            if win:
                print('Move list:', moves)
                return winner

            elif torch.all(board[0, :] != 0):  # Check if the top row is full
                print('Move list:', moves)
                return 0    # draw

            board = -board
            winner = 3 - winner
            model1, model2 = model2, model1 # Swap models for the next turn


def play_parallel(model1, model2, num_games: int, temperature: float = 1.0) -> tuple[int, int, int]:
    """Have two models play num_games against each other in parallel.
    Returns (model 1 wins, model 2 wins, draws)."""
    model1.eval()
    model2.eval()

    device = get_device()

    with torch.no_grad():
        active = torch.ones((num_games,), dtype=torch.int8, device=device)
        winner = torch.ones((num_games,), dtype=torch.int8, device=device)
        board  = torch.zeros((num_games, ROWS, COLS), dtype=torch.int8, device=device)

        iact = torch.where(active)[0]
        while torch.any(active):

            moves = sample_moves(model1, board[iact], temperature=temperature)
            board[iact], wins, draws = make_move_and_check_batch(board[iact], moves)

            active[iact] &= ~(wins | draws) # deactivate games that are won or drawn
            winner[iact[draws]] = 0         # set winner to 0 for drawn games

            board[iact] *= -1
            iact = torch.where(active)[0]
            winner[iact] = 3 - winner[iact]     # change potential winner for next round in active games

            model1, model2 = model2, model1 # Swap models for the next turn

    return (winner == 1).sum().item(), (winner == 2).sum().item(), (winner == 0).sum().item()


def play_parallel_to_depth(
    model1: torch.nn.Module, model2: torch.nn.Module,
    num_games: int, num_moves: int, temperature: float = 1.0
) -> torch.Tensor:
    """
    Play num_games in parallel for exactly num_moves moves (plies), alternating models.
    Returns:
        board: Tensor of shape (num_games, ROWS, COLS) with the resulting board states.
    """
    model1.eval()
    model2.eval()
    device = get_device()
    with torch.no_grad():
        board = torch.zeros((num_games, ROWS, COLS), dtype=torch.int8, device=device)
        active = torch.ones((num_games,), dtype=torch.bool, device=device)
        for move_idx in range(num_moves):
            iact = torch.where(active)[0]
            if iact.numel() == 0:
                break
            moves = sample_moves(model1, board[iact], temperature=temperature)
            board[iact], wins, draws = make_move_and_check_batch(board[iact], moves)
            # Optionally, you could deactivate finished games, but we keep all boards for output
            board[iact] *= -1
            # Swap models for next turn
            model1, model2 = model2, model1
    return board


if __name__ == '__main__':
    import sys
    from model import load_frozen_model

    device = init_device(False)

    model_names = sys.argv[1:]
    if len(model_names) != 2:
        print('Please pass two model names.')
        sys.exit()
    
    models = [load_frozen_model(name).to(device) for name in model_names]
    
    play(models[0], models[1], output=True)
