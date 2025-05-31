import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os.path
import numpy as np

from globals import ROWS, COLS, init_device, get_device
from board import *
from model import *
from stats import *
from tournament import run_fast_tournament, win_rate
from league import League
from play import play, sample_move, sample_moves

RESET_OPTIMIZER = False
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0
OPPONENT_TEMPERATURE = 1.0

GRAD_NORM_CLIPPING = None # 1000.0
#ENTROPY_BONUS = 0.030   #0.075-0.08 is good for exploration; lower to improve performance once sufficiently explored
ENTROPY_BONUS = 0.050       #0.075-0.08 is good for exploration; lower to improve performance once sufficiently explored
VALUE_LOSS_WEIGHT = 0.5

NORMALIZE_ADVANTAGE = False       # normalize the advantage estimate per batch
BOOTSTRAP_VALUE   = True        # use Actor-Critic (A2C) for value bootstrapping; if off, use direct Monte Carlo samping
KEEP_DRAWS        = True        # whether drawn games are kept in the training data (reward 0) or discarded

REWARD_DISCOUNT = 0.98

PPO_CLIP_EPSILON = 0.2
PPO_EPOCHS = 5
PPO_TARGET_KL = 0.01

ALGORITHM = "PPO"

def set_params(
    learning_rate=1e-4,
    weight_decay=0,
    opponent_temperature=1.0,
    grad_norm_clipping=None,
    entropy_bonus=0.05,
    value_loss_weight=0.5,
    bootstrap_value=False,
    keep_draws=True,
    reward_discount=0.90,
    ppo_clip_epsilon=0.2,
    ppo_epochs=4,
    ppo_target_kl=0.01,
    algorithm="A2C",
):
    """Allow overriding of default parameters from scripts."""
    global LEARNING_RATE, WEIGHT_DECAY, OPPONENT_TEMPERATURE, GRAD_NORM_CLIPPING, ENTROPY_BONUS, VALUE_LOSS_WEIGHT
    global BOOTSTRAP_VALUE, KEEP_DRAWS, REWARD_DISCOUNT
    global PPO_CLIP_EPSILON, PPO_EPOCHS, PPO_TARGET_KL, ALGORITHM
    LEARNING_RATE = learning_rate
    WEIGHT_DECAY = weight_decay
    OPPONENT_TEMPERATURE = opponent_temperature
    GRAD_NORM_CLIPPING = grad_norm_clipping
    ENTROPY_BONUS = entropy_bonus
    VALUE_LOSS_WEIGHT = value_loss_weight
    BOOTSTRAP_VALUE = bootstrap_value
    KEEP_DRAWS = keep_draws
    REWARD_DISCOUNT = reward_discount
    PPO_CLIP_EPSILON = ppo_clip_epsilon
    PPO_EPOCHS = ppo_epochs
    PPO_TARGET_KL = ppo_target_kl
    ALGORITHM = algorithm


g_stats = BatchStats(["winrate", "game_length", "policy_loss", "value_loss", "entropy", "rewards_std", "advantage_std",
                      "winrate_ref"])


def compute_rewards(num_moves: int, outcome: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    move_nr = torch.arange(num_moves, device=device)
    is_done = (move_nr == (num_moves - 1))
    if BOOTSTRAP_VALUE:
        # bootstrapping: sparse rewards, only assigned for the winning move
        return (is_done * outcome).float(), is_done
    else:
        # Monte Carlo sampling: rewards are discounted over the game
        return (outcome * REWARD_DISCOUNT**move_nr).flip(dims=(0,)), is_done


def play_parallel_with_results(model1, model2, track_player, num_games, opponent_temperature=1.0):
    """Have two models play against each other. Returns all board states, moves, and rewards for player `track_player`."""
    device = get_device()
    model1.eval()
    model2.eval()
    wr = 0.0            # win rate for this batch
    
    all_board_states = []
    all_moves        = []
    all_gameidxs     = []

    all_rewards      = [ None for _ in range(num_games) ]
    all_done         = [ None for _ in range(num_games) ]

    with torch.no_grad():
        # iact has the same length as board and stores the game index
        board = torch.zeros((num_games, ROWS, COLS), dtype=torch.int8, device=device)
        iact  = torch.arange(num_games, device=device)

        curplayer = 0
        num_moves = 0

        while iact.shape[0] > 0:
            # sample moves for all active boards, play them and check results
            moves = sample_moves(model1, board, temperature=opponent_temperature if curplayer != track_player else 1.0)

            if curplayer == track_player:
                # Store the board states and chosen moves for the player we are tracking
                all_board_states.append(board)
                all_moves.append(moves)
                all_gameidxs.append(iact.clone())
                num_moves += 1          # count only our moves

            # Make the moves and check for win/draw
            board, wins, draws = make_move_and_check_batch(board, moves)
            where_wins  = torch.where(wins)[0]          # indices into iact where wins happened
            where_draws = torch.where(draws)[0]         # indices into iact where draws happened

            # Bookkeeping for won and drawn games
            for i in iact[where_wins].cpu():
                # Game over - win
                all_rewards[i], all_done[i] = compute_rewards(num_moves, 1 if (curplayer == track_player) else -1, device)
                wr += 1 if (curplayer == track_player) else 0
                g_stats.add('winrate', 1 if (curplayer == track_player) else 0)
                g_stats.add('game_length', num_moves)

            for i in iact[where_draws].cpu():
                # Game over - draw
                if KEEP_DRAWS:
                    all_rewards[i], all_done[i] = compute_rewards(num_moves, 0, device)
                else:
                    raise Exception('KEEP_DRAWS == False currently not supported')

                wr += 0.5
                g_stats.add('winrate', 0.5)
                g_stats.add('game_length', num_moves)
            
            not_done = torch.where(~(wins | draws))[0]          # which games are still active?
            board = board[not_done]
            iact = iact[not_done]

            # Flip the board state for the next player
            board *= -1
            model1, model2 = model2, model1 # Swap models for the next turn
            curplayer = 1 - curplayer       # toggle 0 <-> 1

    all_board_states = torch.concat(all_board_states, dim=0)
    all_moves = torch.cat(all_moves)
    all_gameidxs = torch.cat(all_gameidxs)

    # order all game states and moves by the game index to which they belong
    reorder = torch.argsort(all_gameidxs, stable=True)

    return all_board_states[reorder], all_moves[reorder], torch.cat(all_rewards), torch.cat(all_done), wr / num_games


def play_multiple_against_model(model, opponent, num_games: int, opponent_temperature=1.0):
    b1, m1, r1, d1, wr1 = play_parallel_with_results(model, opponent, track_player=0, num_games=num_games//2, opponent_temperature=opponent_temperature)
    b2, m2, r2, d2, wr2 = play_parallel_with_results(opponent, model, track_player=1, num_games=num_games//2, opponent_temperature=opponent_temperature)
    return torch.cat((b1, b2)), torch.cat((m1, m2)), torch.cat((r1, r2)), torch.cat((d1, d2)), (wr1 + wr2) / 2.0


def dump_move_info(model, board_states, moves, rewards):
    for i in range(len(board_states)):
        pretty_print_board(board_states[i])
        print(move_entropy(model, board_states[i]))
        print(f"Action: {moves[i]}, Reward: {rewards[i]}")

def mask_invalid_moves_batch(boards: torch.Tensor, logits: torch.Tensor, mask_value=-torch.inf):
    illegal_mask = torch.where(boards[:, 0, :] == 0, 0.0, mask_value)     # (B, C)
    return logits + illegal_mask                           # (B, C)


def update_policy(
    model: nn.Module,
    optimizer: optim.Optimizer,
    states:  torch.Tensor,  # (B, R, C) - encountered board positions
    actions: torch.Tensor,  # (B,) - move played in each board position
    returns: torch.Tensor,  # (B,) - contains discounted reward G_t or sparse reward for each step t
    done:    torch.Tensor,  # (B,) - terminal flag for last move in each game
    algorithm: str = "A2C",  # Default to A2C for backward compatibility
    debug: bool = False
) -> tuple:
    """
    Updates the policy model using the REINFORCE or PPO algorithm, assuming the
    'returns' list already contains the calculated return (e.g., G_t) for each step.

    Args:
        model: The policy network (e.g., Connect4MLP instance) to be trained.
        optimizer: The optimizer instance (e.g., Adam) for the model.
        board_states: List of board state tensors for the episode.
        actions: List of actions (column indices) taken during the episode.
        returns: List of pre-calculated returns (e.g., discounted G_t) for each step,
                 expected to be between -1 and 1. Assumes non-zero values for non-draws.

    Returns:
        The calculated loss value for this episode (0.0 if returns list is empty).
    """
    device = next(model.parameters()).device
    
    assert(states.shape[:1] == returns.shape == actions.shape == done.shape)
    assert(states.shape[1:] == (ROWS, COLS))
    actions = actions.unsqueeze(1)              # (B,) -> (B, 1) for gather
    
    # Log standard deviation of returns
    g_stats.add('rewards_std', torch.std(returns).item())
    
    model.train()

    # compute model outputs for the batch
    logits = model(states)
    
    value, value_loss = None, 0.0
    if isinstance(logits, tuple):              # check for presence of value head
        logits, value = logits

    # mask out illegal moves and compute logprobs of the actually taken actions
    masked_logits = mask_invalid_moves_batch(states, logits, mask_value=-1e9)  # (B, C)      # use finite mask instead of -inf to avoid nans in entropy
    log_probs = F.log_softmax(masked_logits, dim=1)                # (B, C)
    entropy = -(log_probs * torch.exp(log_probs)).sum(1)           # (B,)

    # choose logprobs of actually taken actions: (B, C) -> (B, 1) -> (B,)
    log_probs_taken = torch.gather(log_probs, dim=1, index=actions).squeeze(1)

    # Calculate advantage (and v_target for value loss if BOOTSTRAP_VALUE)
    # This is done once before PPO loop or A2C update, using the policy that generated the data
    if value is not None:
        if BOOTSTRAP_VALUE:
            # Actor-Critic (A2C): use bootstrapping to improve value estimate and for advantage calculation
            # next value after 2 ply (only if not terminal)
            V_next = torch.roll(value.detach(), shifts=-1)  # (B,)
            # for terminal states, the value is 0
            V_next[done] = 0.0
            v_target = returns + REWARD_DISCOUNT * V_next # Target for value function
            advantage = (v_target - value).detach()       # Advantage for policy update
        else:
            # REINFORCE with baseline - use Monte Carlo returns as value target
            v_target = returns
            advantage = (returns - value).detach()

        advantage_std, advantage_mean = torch.std_mean(advantage)
        g_stats.add('advantage_std', advantage_std.item())
        if NORMALIZE_ADVANTAGE:
            advantage = (advantage - advantage_mean) / (advantage_std + 1e-8)
    else:
        # No value head (REINFORCE without value function)
        advantage = returns
        v_target = returns

    # Weight for value loss: 2x for terminal states
    weight = torch.ones_like(returns)
    weight[returns == 1]  = 2.0
    weight[returns == -1] = 2.0

    if algorithm == "PPO":
        log_probs_taken_old = log_probs_taken.detach()   # log probs from initial policy
        initial_probs = torch.exp(log_probs.detach())    # initial probabilities for KL divergence calculation

        for epoch in range(PPO_EPOCHS):
            if epoch == 0:
                # use the already computed log_probs, value and entropy from the initial policy; don't recompute
                log_probs_new = log_probs
                log_probs_taken_new = log_probs_taken
                value_new = value
            else:
                logits_new, value_new = model(states)
                masked_logits_new = mask_invalid_moves_batch(states, logits_new, mask_value=-1e9)
                log_probs_new = F.log_softmax(masked_logits_new, dim=1)
                entropy = -(log_probs_new * torch.exp(log_probs_new)).sum(1)
                log_probs_taken_new = torch.gather(log_probs_new, dim=1, index=actions).squeeze(1)

            # Importance sampling: (current probability / original probability) of taken move
            ratio = torch.exp(log_probs_taken_new - log_probs_taken_old)

            # Implement PPO clipping
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPSILON, 1.0 + PPO_CLIP_EPSILON) * advantage
            policy_loss = -torch.min(surr1, surr2).sum()

            value_loss = F.mse_loss(value_new, v_target, weight=weight, reduction='sum')

            total_loss_ppo = policy_loss + VALUE_LOSS_WEIGHT * value_loss - ENTROPY_BONUS * entropy.sum()

            # update the model
            optimizer.zero_grad()
            total_loss_ppo.backward()
            if GRAD_NORM_CLIPPING:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM_CLIPPING)
            optimizer.step()

            if PPO_TARGET_KL is not None:
                # Early stopping based on KL divergence
                with torch.no_grad():
                    kl = (initial_probs * (log_probs - log_probs_new)).sum(dim=1).mean()
                    print(f"  PPO epoch {epoch+1}: KL divergence: {kl.item():.4f}, Policy loss: {(policy_loss / states.shape[0]).item():.4f}")
                    if kl > PPO_TARGET_KL:
                        print(f"Stopping PPO early at epoch {epoch+1} due to KL divergence: {kl.item():.4f} > {PPO_TARGET_KL:.4f}")
                        break


    else:   # A2C or REINFORCE
        value_loss = F.mse_loss(value, v_target, weight=weight, reduction='sum')
        policy_loss = -(advantage * log_probs_taken).sum()
        total_loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss - ENTROPY_BONUS * entropy.sum()

        if debug:
            for i in range(states.shape[0]):
                pretty_print_board(states[i])
                print(f"terminal: {done[i].item()}, move: {actions[i,0].item()}, return: {returns[i].item()}, V_next: {V_next[i].item():.4f}, v_target: {v_target[i].item():.4f}, value: {value[i].item():.4f}, advantage: {advantage[i].item():.4f}, value_loss: {(v_target[i] - value[i]).item()**2:.4f}, chance: {torch.exp(log_probs_taken[i]):.2f}")
                print()
            import sys
            sys.exit(0)

        optimizer.zero_grad()
        total_loss.backward()

        if GRAD_NORM_CLIPPING:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM_CLIPPING)
        gradnorm = nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None]).item()
        print(f'norm(grad) = {gradnorm:.4f}')

        optimizer.step()            # Update model weights

    if debug and False:
        print(" ------ after update ------ ")
        model.eval()
        with torch.no_grad():
            for i in [-2]: #range(len(board_states)):
                pretty_print_board(in_board_states[i])
                print('Entropy:', move_entropy(model, in_board_states[i]))
                print(f"Action: {in_actions[i]}, Reward: {in_returns[i]}")

    bs = states.shape[0]         # batch size
    if value_loss:
        return policy_loss.item() / bs, value_loss.item() / bs, entropy.mean().item()      # report mean so as not to vary with batch size
    else:
        return policy_loss.item() / bs, 0.0, entropy.mean().item()


def show_winrate(model1, model2, num_games=300):
    wr, dr = win_rate(model1, model2, num_games=num_games)
    print(f"Win rate: {100*wr:.2f}%  Draw rate: {100*dr:.2f}%")
    return


def move_entropy(model, board: torch.Tensor) -> float:
    """Calculate the entropy of the model's move distribution."""
    model.eval()
    with torch.no_grad():
        logits = model(board)
        if isinstance(logits, tuple): logits = logits[0]
        probs = F.softmax(logits, dim=-1)
        print(f"Move logprobs: {torch.log(probs)}")
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)) # Add small value to avoid log(0)
        return entropy.item()



def train_against_opponents(model, opponents, checkpoint_file="best_model.pth",
                            batches_per_epoch=5, games_per_batch=1000, debug=False):
    device = get_device()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"Loading model from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if (not RESET_OPTIMIZER) and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    board = torch.zeros((ROWS, COLS), dtype=torch.int8, device=device)
    print("Starting move entropy:", move_entropy(model, board))

    wrplot = UpdatablePlot(labels=[['Win rate', 'Entropy', 'Rewards st.d.'],
                                   ['Policy loss', 'Value loss', 'Advantage st.d.']], show_last_n=200)

    estimated_wr = 0.5 * np.ones((len(opponents),), dtype=np.float32)
    opponent_weights = np.ones((len(opponents),), dtype=np.float32) / len(opponents)
    nprng = np.random.default_rng()

    # ----------------- MAIN TRAINING LOOP ----------------- #
    #for epoch in range(40):
    while True:
        for i in range(batches_per_epoch):
            #random_opponent = torch.randint(0, len(opponents), (1,)).item()
            random_opponent = nprng.choice(len(opponents), p=opponent_weights)
            board_states, actions, rewards, done, wr = play_multiple_against_model(model, opponents[random_opponent], num_games=games_per_batch, opponent_temperature=OPPONENT_TEMPERATURE)
            print(f"opp: {random_opponent} wr: {wr*100:2.0f}% ")
            estimated_wr[random_opponent] = estimated_wr[random_opponent] * 0.90 + wr * 0.10

            policy_loss, value_loss, entropy = update_policy(model, optimizer, board_states, actions, rewards, done, algorithm=ALGORITHM, debug=debug)
            g_stats.add('policy_loss', policy_loss)
            g_stats.add('value_loss', value_loss)
            g_stats.add('entropy', entropy)

            if i % 20 == min(19, batches_per_epoch - 1):
                show_winrate(model, opponents[0], num_games=300)
                opponent_weights = np.clip(1.0 - estimated_wr, 0.05, 1.0)
                opponent_weights /= opponent_weights.sum()
                print(f"Estimated win rates:  {estimated_wr}")
                print(f"New opponent weights: {opponent_weights}")
                g_stats.aggregate()
                wrplot.update_from(g_stats, [
                    'winrate', 'entropy', 'rewards_std',
                    'policy_loss', 'value_loss', 'advantage_std']
                )
                print(f"Batch {i+1} / {batches_per_epoch} done. Avg loss: {g_stats.last('policy_loss'):.4f}. Avg game length: {g_stats.last('game_length'):.2f}. Win rate: {100*g_stats.last('winrate'):.2f}%")
            wrplot.poll()

        if checkpoint_file:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_file)
            print(f'Model saved to {checkpoint_file}')

        play(opponents[0], model, output=True)
    #wrplot.save("final_training_plot.png")



def self_play_with_league(model: nn.Module, league: League, win_threshold=0.75, model_improver=None,
        batches_per_epoch=100, games_per_batch=100):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    cp_file = os.path.join(league.dir, "cur.pth")
    if cp_file and os.path.exists(cp_file):
        print(f"Loading model from {cp_file}")
        checkpoint = torch.load(cp_file, map_location=get_device())
        model.load_state_dict(checkpoint['model_state_dict'])
        if (not RESET_OPTIMIZER) and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    wrplot = UpdatablePlot(labels=[['Win rate', 'Entropy', 'Rewards st.d.'],
                                   ['Policy loss', 'Value loss', 'Advantage st.d.']], show_last_n=200)

    # ----------------- MAIN TRAINING LOOP ----------------- #
    while True:
        print(" ====================== CURRENT LEAGUE ======================= ")
        sort_order = np.argsort(league.estimated_wr)
        for k in range(len(league.models)):
            i = sort_order[k]
            print(f"  {i}: {league.model_names[i]:<40} Win rate: {league.estimated_wr[i]*100:.1f}%")
        print(" ============================================================= ")

        for i in range(batches_per_epoch):
            random_opponent = league.choose_opponent()
            opp_model = league.models[random_opponent]
            if model_improver is not None:
                opp_model = model_improver(opp_model)
            board_states, actions, rewards, done, wr = play_multiple_against_model(model, opp_model, num_games=games_per_batch, opponent_temperature=OPPONENT_TEMPERATURE)
            print(f"opp: {random_opponent} wr: {wr*100:2.0f}% ", end='')
            league.update_winrate(random_opponent, wr)

            policy_loss, value_loss, entropy = update_policy(model, optimizer, board_states, actions, rewards, done, algorithm=ALGORITHM)
            g_stats.add('policy_loss', policy_loss)
            g_stats.add('value_loss', value_loss)
            g_stats.add('entropy', entropy)

            if i % 20 == 19:
                league.update_opponent_weights()
                print(f"Estimated win rates:  {league.estimated_wr}")
                print(f"New opponent weights: {league.opponent_weights}")
                g_stats.aggregate()
                wrplot.update_from(g_stats, [
                    'winrate', 'entropy', 'rewards_std',
                    'policy_loss', 'value_loss', 'advantage_std']
                )
                print(f"Batch {i+1} / {batches_per_epoch} done. Avg loss: {g_stats.last('policy_loss'):.4f}. Avg game length: {g_stats.last('game_length'):.2f}. Win rate: {100*g_stats.last('winrate'):.2f}%")
            wrplot.poll()

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, cp_file)
        print(f'Model saved to {cp_file}')

        play(model, model, output=True)

        if league.estimated_wr.min() > win_threshold:
            league.add_model(model)
        league.save()


def self_play_loop(model_constructor, ref_model, games_per_batch=50, batches_per_epoch=100, learning_rate=1e-3,
                   win_threshold=0.52, fname_prefix="self"):
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

            board_states, actions, returns, done, wr = play_multiple_against_model(model, model_cp, num_games=games_per_batch)

            policy_loss, value_loss, entropy = update_policy(model, optimizer, board_states, actions, returns, done, algorithm=ALGORITHM)
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

        play(model_cp, model, output=True)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, cp_file)
        if g_stats.last('winrate') > win_threshold:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, best_cp_file)
            model_cp.load_state_dict(model.state_dict())  # Copy the model state to the checkpoint model
            print(f"\nEpoch {epoch} done. New best model saved to {best_cp_file}.")
        else:
            print(f"\nEpoch {epoch} done.")

        #show_winrate(model, ref_model, num_games=300)


if __name__ == "__main__":
    device = init_device(True)

    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True

    #ref_model = load_frozen_model('CNN-Mk4:pretrain1-a2c.pth').to(DEVICE)
    #constr = lambda: Connect4CNN_Mk4(value_head=True)
    #self_play_loop(constr, ref_model, games_per_batch=1000, batches_per_epoch=5, learning_rate=1e-3,
    #                   win_threshold=0.60, fname_prefix="self")

    #with profile(
    #    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #    record_shapes=True,
    #    profile_memory=True,
    #    with_stack=False) as prof:
    #    for batchnr in range(1):
    #        board_states, actions, returns, done, wr = play_multiple_against_model(model, ref_model, num_games=1000)
    #        policy_loss, value_loss, entropy = update_policy(model, optimizer, board_states, actions, returns, done, debug=False)
    #prof.export_chrome_trace("my_trace.json")


    import sys
    debug = len(sys.argv) > 1 and sys.argv[1] == 'debug'

    model = Connect4CNN_Mk4(value_head=True).to(device)
    opponents = [
        RolloutModel(load_frozen_model('CNN-Mk4:mk4-ts11.pth').to(device), width=3, depth=6, temperature=1.0),
    ]

    train_against_opponents(model, opponents, batches_per_epoch=10, games_per_batch=250, debug=debug)

    #model = Connect4CNN_Mk4(value_head=True)
    #league = League(model_names=None, dir="selfplay", model_string="CNN-Mk4", device=DEVICE)
    #self_play_with_league(model, league, win_threshold=0.75)
