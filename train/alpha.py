import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from main import RESET_OPTIMIZER, g_stats, mask_invalid_moves_batch, VALUE_LOSS_WEIGHT
from stats import UpdatablePlot
from globals import init_device, get_device, ROWS, COLS
from model import make_move_and_check_batch, RolloutModel, Connect4CNN_Mk4
from play import sample_moves, play
from tournament import win_rate

import os

def play_both_sides(model, num_games, temperature=1.0):
    """Have a model play against itself. Returns all board states, moves, and outcomes."""
    device = get_device()
    model.eval()
    
    all_board_states = []
    all_moves        = []
    all_gameidxs     = []

    with torch.no_grad():
        # iact has the same length as board and stores the game index
        board = torch.zeros((num_games, ROWS, COLS), dtype=torch.int8, device=device)
        iact  = torch.arange(num_games, device=device)
        game_length = torch.zeros((num_games,), dtype=torch.int8, device=device)
        outcome = torch.zeros((num_games,), dtype=torch.int8, device=device)

        curplayer = 0
        num_moves = 0

        while iact.shape[0] > 0:
            # sample moves for all active boards, play them and check results
            moves = sample_moves(model, board, temperature=temperature)

            # Store the board states and chosen moves
            all_board_states.append(board)
            all_moves.append(moves)
            all_gameidxs.append(2 * iact.clone() + curplayer)  # store game index with player info
            num_moves += 1

            # Make the moves and check for win/draw
            board, wins, draws = make_move_and_check_batch(board, moves)
            where_wins  = torch.where(wins)[0]          # indices into iact where wins happened
            where_draws = torch.where(draws)[0]         # indices into iact where draws happened

            outcome[iact[where_wins]] = (1 if curplayer == 0 else -1)
            # outcome for draws remains at 0
            game_length[iact[where_wins]] = num_moves
            game_length[iact[where_draws]] = num_moves

            not_done = torch.where(~(wins | draws))[0]          # which games are still active?
            board = board[not_done]
            iact = iact[not_done]

            # Flip the board state for the next player
            board *= -1
            curplayer = 1 - curplayer       # toggle 0 <-> 1

    all_board_states = torch.concat(all_board_states, dim=0)
    all_moves = torch.cat(all_moves)
    all_gameidxs = torch.cat(all_gameidxs)

    wr = (outcome == 1).sum().item() / num_games

    # interleave outcomes with -outcomes
    all_outcomes = torch.stack([outcome, -outcome], dim=1).view(-1)

    return all_board_states, all_moves, all_outcomes[all_gameidxs], wr


def update_alpha(model, states, actions, outcomes):
    """Update the model using the given board states, actions, and outcomes."""
    device = get_device()
    model.train()

    # Forward pass
    logits, values = model(states)

    # mask out illegal moves and compute logprobs of the actually taken actions
    masked_logits = mask_invalid_moves_batch(states, logits, mask_value=-1e9)  # (B, C)      # use finite mask instead of -inf to avoid nans in entropy
    log_probs = F.log_softmax(masked_logits, dim=1)                # (B, C)
    entropy = -(log_probs * torch.exp(log_probs)).sum(1)           # (B,)

    # choose logprobs of actually taken actions: (B, C) -> (B, 1) -> (B,)
    log_probs_taken = torch.gather(log_probs, dim=1, index=actions.unsqeeze(1)).squeeze(1)

    # Calculate value loss
    value_loss = nn.functional.mse_loss(values, outcomes)

    policy_loss = -log_probs_taken.sum()

    total_loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

    return policy_loss.item(), value_loss.item(), entropy.item()


def train_alpha_mini(model_constructor, model_improver, ref_model, games_per_batch=100, batches_per_epoch=20, learning_rate=1e-3,
                   win_threshold=0.55, fname_prefix="alpha"):
    """Train a model using self-play."""

    cp_file = fname_prefix + "_last.pth"
    best_cp_file = fname_prefix + "_best.pth"
    device = get_device()

    # Create two copies of the model
    model = model_constructor().to(device)
    model_cp = model_constructor().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    if os.path.exists(cp_file):
        print(f"Loading model from {cp_file}")
        checkpoint = torch.load(cp_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_cp.load_state_dict(model.state_dict())  # Copy the model state to the checkpoint model
        if (not RESET_OPTIMIZER) and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if os.path.exists(best_cp_file):
        print(f"Loading best model from {best_cp_file}")
        checkpoint = torch.load(best_cp_file, map_location=get_device())
        model_cp.load_state_dict(checkpoint['model_state_dict'])

    wrplot = UpdatablePlot(labels=[['Win rate (against ref)', 'Entropy', "Returns st.d."],
                                   ['Policy loss', 'Value loss', 'Advantage st.d.']], show_last_n=200)

    # ----------------- MAIN TRAINING LOOP ----------------- #
    epoch = 0
    while True:
        for batchnr in range(batches_per_epoch):

            improved_model = model_improver(model)
            board_states, actions, outcomes, wr = play_both_sides(improved_model, num_games=games_per_batch, temperature=1.0)
            policy_loss, value_loss, entropy = update_alpha(model, board_states, actions, outcomes)
            g_stats.add('policy_loss', policy_loss)
            g_stats.add('value_loss', value_loss)
            g_stats.add('entropy', entropy)

            if batchnr % 20 == min(19, batches_per_epoch - 1):
                if ref_model is not None:
                    wr, dr = win_rate(model, ref_model, num_games=100)
                    g_stats.add('winrate_ref', wr)
                g_stats.aggregate()
                wr_src = 'winrate_ref' if ref_model is not None else 'winrate'
                wrplot.update_from(g_stats, [
                    wr_src, 'entropy', 'rewards_std',
                    'policy_loss', 'value_loss', 'advantage_std'
                ])
                print(f"Batch {batchnr+1} / {batches_per_epoch} done. Avg loss: {g_stats.last('policy_loss'):.4f}. Avg game length: {g_stats.last('game_length'):.2f}. Win rate: {100*g_stats.last('winrate'):.2f}%")
            wrplot.poll()

        # End of epoch; save model and create new checkpoint if performance improved
        epoch += 1

        play(model, model, output=True)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, cp_file)
        #if g_stats.last('winrate') > win_threshold:
        #    torch.save({
        #        'model_state_dict': model.state_dict(),
        #        'optimizer_state_dict': optimizer.state_dict()
        #    }, best_cp_file)
        #    model_cp.load_state_dict(model.state_dict())  # Copy the model state to the checkpoint model
        #    print(f"\nEpoch {epoch} done. New best model saved to {best_cp_file}.")
        #else:
        print(f"\nEpoch {epoch} done.")

if __name__ == "__main__":
    device = init_device(True)

    model_constr = lambda: Connect4CNN_Mk4(value_head=True)
    model_improver = lambda m: RolloutModel(m, width=4, depth=4)
    train_alpha_mini(model_constr, model_improver, ref_model=None,
                      games_per_batch=100, batches_per_epoch=20, learning_rate=1e-6,
                      win_threshold=0.55, fname_prefix="alpha")
    