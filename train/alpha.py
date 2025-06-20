import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from main import g_stats, mask_invalid_moves_batch, augment_symmetry
from stats import UpdatablePlot
from globals import init_device, get_device, ROWS, COLS
from model import make_move_and_check_batch, RolloutModel, Connect4CNN_Mk4, Connect4MLP3, load_frozen_model
from play import sample_moves, play
from tournament import win_rate
from board import pretty_print_board

import os

RESET_OPTIMIZER = False
LEARNING_RATE   = 1e-4
VALUE_LOSS_WEIGHT = 1.5
ROLLOUT_TEMPERATURE = 1.0

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
    g_stats.add('game_length', game_length.sum().item() / num_games)

    # interleave outcomes with -outcomes
    all_outcomes = torch.stack([outcome, -outcome], dim=1).view(-1)

    return all_board_states, all_moves, all_outcomes[all_gameidxs], wr


def update_alpha(model, optimizer, states, actions, outcomes, debug=False):
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
    log_probs_taken = torch.gather(log_probs, dim=1, index=actions.unsqueeze(1)).squeeze(1)

    #print(log_probs_taken)
    #k_worst = torch.argmin(log_probs_taken).item()
    #print(k_worst)
    #pretty_print_board(states[k_worst])
    #print(actions[k_worst])
    #print(logits[k_worst])

    policy_loss = -log_probs_taken.sum()
    value_loss = nn.functional.mse_loss(values, outcomes.float(), reduction='sum')
    total_loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

    optimizer.zero_grad()
    total_loss.backward()
    gradnorm = nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None]).item()
    print(f'norm(grad) = {gradnorm:.4f}')
    optimizer.step()

    if debug:
        import numpy as np
        debug_board = torch.zeros((1, ROWS, COLS), dtype=torch.int8, device=device)  # which board state to debug
        debug_board[0,5,3] = 1
        #debug_board[0,5,0] = -1
        #debug_board[0,4,2] = 1
        debug_board *= -1
        debug_states = torch.where((states == debug_board).all(dim=(1, 2)))[0]  # find indices of all board state to debug
        if debug_states.numel() > 0:
            k = debug_states[0].item()
            pretty_print_board(debug_board[0])
            initial_probs = F.softmax(logits[k], dim=-1).detach().cpu().numpy()
            initial_entropy = -(initial_probs * np.log(initial_probs)).sum()
            np.set_printoptions(precision=3)
            print(f"  Initial probs: {initial_probs}, entropy: {initial_entropy:.4f}           value: {values[k].item():.2f}")
            # count how often each action was taken
            unique_actions, counts = np.unique(actions[debug_states].cpu().numpy(), return_counts=True)
            print("  Actions taken:")
            for a, c in zip(unique_actions, counts):
                print('    ', int(a), ':', int(c))
            final_logits = model(debug_board)[0].detach().cpu()
            final_probs = F.softmax(final_logits[0], dim=-1)
            final_entropy = -(final_probs * torch.log(final_probs)).sum().item()
            print(f"  Final probs: {final_probs.numpy()}, entropy: {final_entropy:.4f}")

    bs = states.shape[0]         # batch size
    return policy_loss.item() / bs, value_loss.item() / bs, entropy.mean().item()      # report mean so as not to vary with batch size


def train_alpha_mini(model_constructor, model_improver, ref_model, games_per_batch=100, batches_per_epoch=20, learning_rate=1e-3,
                   win_threshold=0.55, fname_prefix="alpha"):
    """Train a model using self-play."""

    cp_file = fname_prefix + "_last.pth"
    best_cp_file = fname_prefix + "_best.pth"
    device = get_device()

    # Create two copies of the model
    model = model_constructor().to(device)
    #model_cp = model_constructor().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    if os.path.exists(cp_file):
        print(f"Loading model from {cp_file}")
        checkpoint = torch.load(cp_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        #model_cp.load_state_dict(model.state_dict())  # Copy the model state to the checkpoint model
        if (not RESET_OPTIMIZER) and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #if os.path.exists(best_cp_file):
    #    print(f"Loading best model from {best_cp_file}")
    #    checkpoint = torch.load(best_cp_file, map_location=get_device())
    #    model_cp.load_state_dict(checkpoint['model_state_dict'])

    wrplot = UpdatablePlot(labels=[['Win rate', 'Entropy', "Returns st.d."],
                                   ['Policy loss', 'Value loss', 'Advantage st.d.']], show_last_n=200)

    # ----------------- MAIN TRAINING LOOP ----------------- #
    epoch = 0
    while True:
        for batchnr in range(batches_per_epoch):

            improved_model = model_improver(model)
            board_states, actions, outcomes, wr = play_both_sides(improved_model, num_games=games_per_batch)

            done_dummy = torch.zeros((1,))
            board_states, actions, outcomes, done_dummy = augment_symmetry(board_states, actions, outcomes, done_dummy)

            policy_loss, value_loss, entropy = update_alpha(model, optimizer, board_states, actions, outcomes, debug=False)
            g_stats.add('winrate', wr)
            g_stats.add('policy_loss', policy_loss)
            g_stats.add('value_loss', value_loss)
            g_stats.add('entropy', entropy)

            print(f'Batch size: {board_states.shape[0]}, win rate (P1): {wr*100:.1f}%')

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

    model_constr = Connect4MLP3
    model_improver = lambda m: RolloutModel(m, width=4, depth=4, temperature=ROLLOUT_TEMPERATURE)
    ref_model = load_frozen_model('CNN-Mk4:model_B_5.pth').to(device)

    train_alpha_mini(model_constr, model_improver, ref_model=ref_model,
                      games_per_batch=100, batches_per_epoch=10, learning_rate=LEARNING_RATE,
                      win_threshold=0.55, fname_prefix="mlpalpha")
