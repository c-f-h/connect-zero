import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os.path
import numpy as np
import matplotlib.pyplot as plt

from typing import List

from board import *
from model import *
from stats import *
from tournament import run_fast_tournament, win_rate
from league import League

# Define board dimensions
ROWS = 6
COLS = 7

RESET_OPTIMIZER = True
LEARNING_RATE = 3e-7
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
PPO_EPOCHS = 4
PPO_TARGET_KL = 0.01

ALGORITHM = "A2C"

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


DEVICE = None
g_stats = BatchStats(["winrate", "game_length", "policy_loss", "value_loss", "entropy", "rewards_std", "advantage_std",
                      "winrate_ref"])


def init_device(allow_cuda):
    global DEVICE
    if allow_cuda:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device("cpu")        # cuda is just slower currently
    print(f"Using device: {DEVICE}")
    return DEVICE


def compute_rewards(num_moves: int, outcome: int) -> torch.Tensor:
    move_nr = torch.arange(num_moves, device=DEVICE)
    is_done = (move_nr == (num_moves - 1))
    if BOOTSTRAP_VALUE:
        # bootstrapping: sparse rewards, only assigned for the winning move
        return (is_done * outcome).float(), is_done
    else:
        # Monte Carlo sampling: rewards are discounted over the game
        return (outcome * REWARD_DISCOUNT**move_nr).flip(dims=(0,)), is_done


def sample_move(model, board: torch.Tensor, epsilon=0.0, output_probs=False) -> int:
    """Sample a move using the model's output logits."""
    value = None
    if epsilon > 0 and torch.rand(1).item() < epsilon:
        # Epsilon-greedy strategy: choose a random move with probability epsilon
        logits = torch.zeros((COLS,), dtype=torch.float32, device=DEVICE)
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
        return torch.empty(0, dtype=torch.long, device=DEVICE)

    # Get model logits
    value = None
    logits = model(boards)
    if isinstance(logits, tuple):
        logits, value = logits  # model could return (policy, value) or just policy
    logits /= temperature

    # Mask out illegal moves (columns that are full)
    illegal_moves_mask = (boards[:, 0, :] != 0)
    logits[illegal_moves_mask] = -torch.inf

    all_illegal = (logits == -torch.inf).all(dim=-1)
    if all_illegal.any():
        raise ValueError(f"Board(s) with no legal moves found: {torch.where(all_illegal)[0].tolist()}")

    probs = F.softmax(logits, dim=-1)
    moves = torch.multinomial(probs, 1).squeeze(-1)
    return moves


def sample_random_move(board: torch.Tensor) -> int:
    """Sample a random move from the available columns."""
    available_moves = torch.where(board[0, :] == 0)[0]
    if available_moves.numel() == 0:
        raise ValueError("No valid moves available.")
    move = torch.randint(0, available_moves.numel(), (1,)).item()
    return available_moves[move].item()


def play_self(model, reward_discount=0.95, epsilon_greedy=0.0):
    """Have a model play against itself."""
    model.eval()
    board_states, moves = [], []
    curplayer = 1
    with torch.no_grad():
        board = torch.zeros((ROWS, COLS), dtype=torch.int8, device=DEVICE)

        while True:
            move = sample_move(model, board, epsilon=epsilon_greedy)
            board_states.append(board)  # no clone necessary since we do no destructive updates
            moves.append(move)
            board, win = make_move_and_check(board, move)

            if win:
                num_moves = len(board_states)
                move_nr = torch.arange(num_moves)
                rewards = ((-1.0)**move_nr * reward_discount**(move_nr // 2))
                rewards = rewards.flip(dims=(0,))
                g_stats.add('winrate', 1 if curplayer == 1 else 0)      # note: here this simply means "player 1 wins", win rate is meaningless in selfplay
                return board_states, moves, rewards

            elif torch.all(board[0, :] != 0):  # Check if the top row is full - draw
                g_stats.add('winrate', 0.5)
                if KEEP_DRAWS:
                    return board_states, moves, torch.zeros((len(board_states),))
                else:
                    return [], [], []

            board = -board
            curplayer = 3 - curplayer


def play_against_model(model, opponentmodel, reward_discount=0.95, epsilon_greedy=0.0):
    """Have a model play against a frozen opponent model."""
    model.eval()
    opponentmodel.eval()
    board_states, moves = [], []

    curplayer = torch.randint(1, 3, (1,)).item() # Randomly choose starting player

    with torch.no_grad():
        board = torch.zeros((ROWS, COLS), dtype=torch.int8, device=DEVICE)

        while True:
            if curplayer == 1:
                move = sample_move(model, board, epsilon=epsilon_greedy)
                board_states.append(board)  # no clone necessary since we do no destructive updates
                moves.append(move)
            else:
                move = sample_move(opponentmodel, board)
            board, win = make_move_and_check(board, move)

            if win:
                num_moves = len(board_states)
                move_nr = torch.arange(num_moves)
                R = 1.0 if curplayer == 1 else -1.0
                rewards = R * reward_discount**move_nr
                rewards = rewards.flip(dims=(0,))
                g_stats.add('winrate', 1 if curplayer == 1 else 0)
                return board_states, moves, rewards

            elif torch.all(board[0, :] != 0):  # Check if the top row is full - draw
                g_stats.add('winrate', 0.5)
                if KEEP_DRAWS:
                    return board_states, moves, torch.zeros((len(board_states),))
                else:
                    return [], [], []

            board = -board
            curplayer = 3 - curplayer # Switch players


def play_parallel_with_results(model1, model2, track_player, num_games, opponent_temperature=1.0):
    """Have two models play against each other. Returns all board states, moves, and rewards for player `track_player`."""
    model1.eval()
    model2.eval()
    wr = 0.0            # win rate for this batch
    
    all_board_states = [ [] for _ in range(num_games) ]
    all_moves        = [ [] for _ in range(num_games) ]
    all_rewards      = [ None for _ in range(num_games) ]
    all_done         = [ None for _ in range(num_games) ]
    
    with torch.no_grad():
        active = torch.ones((num_games,), dtype=torch.int8, device=DEVICE)
        board = torch.zeros((num_games, ROWS, COLS), dtype=torch.int8, device=DEVICE)

        curplayer = 0

        while torch.any(active):
            iact = torch.where(active)[0]

            # sample moves for all active boards, play them and check results
            moves = sample_moves(model1, board[iact], temperature=opponent_temperature if curplayer != track_player else 1.0)

            if curplayer == track_player:
                for k, i in enumerate(iact):
                    # Store the board state and chosen move for the player we are tracking
                    all_board_states[i].append(board[i].clone())
                    all_moves[i].append(moves[k].item())

            # Make the moves and check for win/draw
            board[iact], wins, draws = make_move_and_check_batch(board[iact], moves)

            for k, i in enumerate(iact):
                if wins[k].item():
                    # Game over - win
                    num_moves = len(all_board_states[i])
                    all_rewards[i], all_done[i] = compute_rewards(num_moves, 1 if (curplayer == track_player) else -1)
                    wr += 1 if (curplayer == track_player) else 0
                    g_stats.add('winrate', 1 if (curplayer == track_player) else 0)
                    g_stats.add('game_length', num_moves)

                elif draws[k].item():
                    # Game over - draw
                    if KEEP_DRAWS:
                        num_moves = len(all_board_states[i])
                        all_rewards[i], all_done[i] = compute_rewards(num_moves, 0)
                    else:
                        all_board_states[i] = []
                        all_moves[i] = []
                        all_done[i] = []

                    wr += 0.5
                    g_stats.add('winrate', 0.5)
                    g_stats.add('game_length', num_moves)
            
            active[iact] &= ~(wins | draws) # deactivate games that are won or drawn

            # Flip the board state for the next player
            board[iact] *= -1
            model1, model2 = model2, model1 # Swap models for the next turn
            curplayer = 1 - curplayer       # toggle 0 <-> 1

    return sum(all_board_states, []), sum(all_moves, []), torch.cat([r for r in all_rewards if r is not None]), all_done, wr / num_games


def play_multiple(model, num_games: int, game_func):
    """Have a model play multiple matches. game_func is called to play one match."""
    all_board_states = []
    all_moves = []
    all_rewards = []

    games_played = 0
    game_length = 0

    for _ in range(num_games):
        board_states, moves, rewards = game_func(model)

        if len(board_states) > 0:
            games_played += 1
            game_length += len(board_states)

            all_board_states.extend(board_states)
            all_moves.extend(moves)
            all_rewards.append(rewards)     # list of tensors

    g_stats.add('game_length', game_length, games_played)
    all_rewards = torch.cat(all_rewards, dim=0)
    return all_board_states, all_moves, all_rewards


def play_multiple_against_model(model, opponent, num_games: int, opponent_temperature=1.0):
    b1, m1, r1, d1, wr1 = play_parallel_with_results(model, opponent, track_player=0, num_games=num_games//2, opponent_temperature=opponent_temperature)
    b2, m2, r2, d2, wr2 = play_parallel_with_results(opponent, model, track_player=1, num_games=num_games//2, opponent_temperature=opponent_temperature)
    return b1 + b2, m1 + m2, torch.cat((r1, r2)), d1 + d2, (wr1 + wr2) / 2.0


def dump_move_info(model, board_states, moves, rewards):
    for i in range(len(board_states)):
        pretty_print_board(board_states[i])
        print(move_entropy(model, board_states[i]))
        print(f"Action: {moves[i]}, Reward: {rewards[i]}")

def play(model1, model2, output=False):
    """Have two models play against each other."""
    model1.eval()
    model2.eval()
    winner = 1
    moves = []
    with torch.no_grad():
        board = torch.zeros((ROWS, COLS), dtype=torch.int8, device=DEVICE)

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


def play_parallel(model1, model2, num_games):
    """Have two models play num_games against each other in parallel.
    Returns (model 1 wins, model 2 wins, draws)."""
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        active = torch.ones((num_games,), dtype=torch.int8, device=DEVICE)
        winner = torch.ones((num_games,), dtype=torch.int8, device=DEVICE)
        board = torch.zeros((num_games, ROWS, COLS), dtype=torch.int8, device=DEVICE)

        while torch.any(active):

            moves = sample_moves(model1, board)

            for i in range(num_games):
                if not active[i]:
                    continue
                board[i], win = make_move_and_check(board[i], moves[i].item())
                if win:
                    active[i] = 0
                elif torch.all(board[i, 0, :] != 0):  # Check if the top row is full   
                    active[i] = 0
                    winner[i] = 0

            board = -board      # also flips inactive, but it doesn't matter
            iact = torch.where(active)[0]
            winner[iact] = 3 - winner[iact]     # change potential winner for next round in active games
            
            model1, model2 = model2, model1 # Swap models for the next turn
    
    return (winner == 1).sum().item(), (winner == 2).sum().item(), (winner == 0).sum().item()

def play_parallel2(model1, model2, num_games):
    """Have two models play num_games against each other in parallel.
    Returns (model 1 wins, model 2 wins, draws)."""
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        active = torch.ones((num_games,), dtype=torch.int8, device=DEVICE)
        winner = torch.ones((num_games,), dtype=torch.int8, device=DEVICE)
        board  = torch.zeros((num_games, ROWS, COLS), dtype=torch.int8, device=DEVICE)

        iact = torch.where(active)[0]
        while torch.any(active):

            moves = sample_moves(model1, board[iact])
            board[iact], wins, draws = make_move_and_check_batch(board[iact], moves)

            active[iact] &= ~(wins | draws) # deactivate games that are won or drawn
            winner[iact[draws]] = 0         # set winner to 0 for drawn games

            board[iact] *= -1
            iact = torch.where(active)[0]
            winner[iact] = 3 - winner[iact]     # change potential winner for next round in active games
            
            model1, model2 = model2, model1 # Swap models for the next turn
    
    return (winner == 1).sum().item(), (winner == 2).sum().item(), (winner == 0).sum().item()


def mask_invalid_moves_batch(boards: torch.Tensor, logits: torch.Tensor, mask_value=-torch.inf):
    illegal_mask = torch.where(boards[:, 0, :] == 0, 0.0, mask_value)     # (B, C)
    return logits + illegal_mask                           # (B, C)


def update_policy(
    model: nn.Module,
    optimizer: optim.Optimizer,
    in_board_states: List[torch.Tensor],
    in_actions: List[int],
    in_returns: torch.Tensor, # contains discounted reward G_t or sparse reward for each step t
    in_done: List[torch.Tensor],
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
    num_steps = len(in_board_states)
    if num_steps == 0: return 0.0
    
    device = next(model.parameters()).device
    
    # Convert inputs to tensors
    states = torch.stack(in_board_states).to(device)            # (B, R, C)
    actions = torch.tensor(in_actions, dtype=torch.int64, device=device) # (B,)
    returns = in_returns.to(device)                             # (B,)
    done = torch.cat(in_done).to(device)                        # (B,)
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


def show_tournament(all_models, model_names, num_games=300):
    final_ratings = run_fast_tournament(
        models=all_models,
        num_rounds=num_games,
        model_names=model_names,
    )
    
    if final_ratings:
        print()
        rank = 1
        for name, wins in final_ratings.items():
            print(f" {rank}. {name:<15} -- {wins} wins")
            rank += 1
    else:
        print("No results.")


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



def train_against_opponents(model, opponents, checkpoint_file="best_model.pth", debug=False):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"Loading model from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        if (not RESET_OPTIMIZER) and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    board = torch.zeros((ROWS, COLS), dtype=torch.int8, device=DEVICE)
    print("Starting move entropy:", move_entropy(model, board))

    wrplot = UpdatablePlot(labels=[['Win rate', 'Entropy', 'Rewards st.d.'],
                                   ['Policy loss', 'Value loss', 'Advantage st.d.']], show_last_n=200)

    estimated_wr = 0.5 * np.ones((len(opponents),), dtype=np.float32)
    opponent_weights = np.ones((len(opponents),), dtype=np.float32) / len(opponents)
    nprng = np.random.default_rng()

    # ----------------- MAIN TRAINING LOOP ----------------- #
    #for epoch in range(40):
    while True:
        num_batches = 100
        num_games = 50
        for i in range(num_batches):
            #random_opponent = torch.randint(0, len(opponents), (1,)).item()
            random_opponent = nprng.choice(len(opponents), p=opponent_weights)
            board_states, actions, rewards, done, wr = play_multiple_against_model(model, opponents[random_opponent], num_games=num_games, opponent_temperature=OPPONENT_TEMPERATURE)
            print(f"opp: {random_opponent} wr: {wr*100:2.0f}% ")
            estimated_wr[random_opponent] = estimated_wr[random_opponent] * 0.90 + wr * 0.10

            policy_loss, value_loss, entropy = update_policy(model, optimizer, board_states, actions, rewards, done, algorithm=ALGORITHM, debug=debug)
            g_stats.add('policy_loss', policy_loss)
            g_stats.add('value_loss', value_loss)
            g_stats.add('entropy', entropy)

            if i % 20 == 19:
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
                print(f"Batch {i+1} / {num_batches} done. Avg loss: {g_stats.last('policy_loss'):.4f}. Avg game length: {g_stats.last('game_length'):.2f}. Win rate: {100*g_stats.last('winrate'):.2f}%")
            wrplot.poll()

        if checkpoint_file:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_file)
            print(f'Model saved to {checkpoint_file}')

        play(opponents[0], model, output=True)
    #wrplot.save("final_training_plot.png")


def self_play_with_league(model: nn.Module, league: League, win_threshold=0.75):
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    cp_file = os.path.join(league.dir, "cur.pth")
    if cp_file and os.path.exists(cp_file):
        print(f"Loading model from {cp_file}")
        checkpoint = torch.load(cp_file, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        if (not RESET_OPTIMIZER) and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    wrplot = UpdatablePlot(labels=[['Win rate', 'Entropy', 'Rewards st.d.'],
                                   ['Policy loss', 'Value loss', 'Advantage st.d.']], show_last_n=200)

    # ----------------- MAIN TRAINING LOOP ----------------- #
    while True:
        num_batches = 100
        num_games = 50
        print(" ====================== CURRENT LEAGUE ======================= ")
        sort_order = np.argsort(league.estimated_wr)
        for k in range(len(league.models)):
            i = sort_order[k]
            print(f"  {i}: {league.model_names[i]:<40} Win rate: {league.estimated_wr[i]*100:.1f}%")
        print(" ============================================================= ")

        for i in range(num_batches):
            random_opponent = league.choose_opponent()
            board_states, actions, rewards, done, wr = play_multiple_against_model(model, league.models[random_opponent], num_games=num_games, opponent_temperature=OPPONENT_TEMPERATURE)
            print(f"opp: {random_opponent} wr: {wr*100:2.0f}% ", end='')
            league.update_winrate(random_opponent, wr)

            policy_loss, value_loss, entropy = update_policy(model, optimizer, board_states, actions, rewards, done, debug=debug)
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
                print(f"Batch {i+1} / {num_batches} done. Avg loss: {g_stats.last('policy_loss'):.4f}. Avg game length: {g_stats.last('game_length'):.2f}. Win rate: {100*g_stats.last('winrate'):.2f}%")
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
    global DEVICE
    if DEVICE is None:
        DEVICE = init_device(False)

    # Create two copies of the model
    model = model_constructor().to(DEVICE)
    model_cp = model_constructor().to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    if os.path.exists(cp_file):
        print(f"Loading model from {cp_file}")
        checkpoint = torch.load(cp_file, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model_cp.load_state_dict(model.state_dict())  # Copy the model state to the checkpoint model
        if (not RESET_OPTIMIZER) and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if os.path.exists(best_cp_file):
        print(f"Loading best model from {best_cp_file}")
        checkpoint = torch.load(best_cp_file, map_location=DEVICE)
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

            if batchnr % 20 == 19:
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
    init_device(False)

    #import pyinstrument
    #profiler = pyinstrument.Profiler()
    #profiler.start()

    #ref_model = load_frozen_model('CNN-Mk4:model-mk4-slf3.pth').to(DEVICE)
    #constr = lambda: Connect4CNN_Mk4(value_head=True)
    #self_play_loop(constr, ref_model, games_per_batch=50, batches_per_epoch=100, learning_rate=LEARNING_RATE, win_threshold=0.60)

    import sys
    debug = len(sys.argv) > 1 and sys.argv[1] == 'debug'

    model = Connect4CNN_Mk4(value_head=True)
    #opponents = [
    #    load_frozen_model('CNN-Mk4:model-mk4-a2c-b3.pth').to(DEVICE),
    #    load_frozen_model('CNN-Mk4:model-mk4-a2c-b2.pth').to(DEVICE),
    #    load_frozen_model('CNN-Mk4:model-mk4-slf2.pth').to(DEVICE),
    #    load_frozen_model('CNN-Mk4:model-mk4-a2c-cp22.pth').to(DEVICE),
    #    load_frozen_model('CNN-Mk4:model-mk4-a2c-cp16.pth').to(DEVICE),
    #]
    #train_against_opponents(model, opponents, debug=debug)

    league = League(model_names=None, dir="selfplay", model_string="CNN-Mk4", device=DEVICE)
    self_play_with_league(model, league, win_threshold=0.75)

    #if profiler:
    #    profiler.stop()
    #    profiler.output_html()
    #    profiler.open_in_browser()
