import math
import random
from typing import List, Callable, Any, Dict, Tuple
import numpy as np
import click

# --- Configuration ---
INITIAL_ELO = 1200
K_FACTOR = 32
NUM_ROUNDS = 10 # How many times each pair plays against each other (total games = N*(N-1)*NUM_ROUNDS)

# --- Elo Calculation Functions ---

def calculate_expected_score(rating1: float, rating2: float) -> Tuple[float, float]:
    """Calculates the expected scores for player 1 and player 2."""
    expected1 = 1 / (1 + 10**((rating2 - rating1) / 400))
    expected2 = 1 - expected1 # E1 + E2 = 1
    return expected1, expected2

def update_elo(rating1: float, rating2: float, result: int, k_factor: int = K_FACTOR) -> Tuple[float, float]:
    """
    Updates Elo ratings based on the game result.

    Args:
        rating1: Current Elo rating of model 1.
        rating2: Current Elo rating of model 2.
        result: Outcome from play(model1, model2) (0: draw, 1: model1 win, 2: model2 win).
        k_factor: The K-factor to use for the update.

    Returns:
        A tuple containing the new Elo ratings (new_rating1, new_rating2).
    """
    expected1, expected2 = calculate_expected_score(rating1, rating2)

    if result == 1: # Model 1 wins
        score1 = 1.0
        score2 = 0.0
    elif result == 2: # Model 2 wins
        score1 = 0.0
        score2 = 1.0
    elif result == 0: # Draw
        score1 = 0.5
        score2 = 0.5
    else:
        raise ValueError(f"Invalid game result: {result}. Expected 0, 1, or 2.")

    new_rating1 = rating1 + k_factor * (score1 - expected1)
    new_rating2 = rating2 + k_factor * (score2 - expected2)

    return new_rating1, new_rating2

# --- Tournament Runner ---

def run_tournament(
    models: List[Any],
    num_rounds: int = NUM_ROUNDS,
    initial_elo: int = INITIAL_ELO,
    k_factor: int = K_FACTOR,
    model_names: List[str] = None
) -> Dict[str, float]:
    """
    Runs a round-robin tournament and calculates Elo ratings.

    Args:
        models: A list of model objects. These objects will be passed to play_func.
        num_rounds: The number of full round-robins to play. Each pair plays
                    num_rounds * 2 games (once as player 1, once as player 2).
        initial_elo: The starting Elo for all models.
        k_factor: The K-factor for Elo updates.
        model_names: Optional list of names for the models. If None, names
                     like "Model 1", "Model 2" will be generated.

    Returns:
        A dictionary mapping model names to their final Elo ratings, sorted descending.
    """
    from main import play
    
    num_models = len(models)
    if num_models < 2:
        print("Need at least two models to run a tournament.")
        return {}

    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(num_models)]
    elif len(model_names) != num_models:
        raise ValueError("Length of model_names must match the number of models.")

    # Initialize Elo ratings
    elo_ratings = {name: float(initial_elo) for name in model_names}
    model_map = {name: model for name, model in zip(model_names, models)}
    name_list = list(model_names) # Keep a consistent order

    print(f"Starting Tournament: {num_models} models, {num_rounds} round(s)...", end="", flush=True)
    #print(f"Initial Elo: {initial_elo}, K-Factor: {k_factor}")
    #print("-" * 30)

    total_games = 0
    win_table = np.zeros((num_models, num_models), dtype=int)
    for r in range(num_rounds):
        #print(f"--- Round {r + 1} / {num_rounds} ---")
        round_games = 0
        # Iterate through all unique pairs, playing both ways (A vs B and B vs A)
        for i in range(num_models):
            for j in range(num_models):
                if i == j:
                    continue # Models don't play against themselves

                name1 = name_list[i]
                name2 = name_list[j]
                model1 = model_map[name1]
                model2 = model_map[name2]

                rating1 = elo_ratings[name1]
                rating2 = elo_ratings[name2]

                # Play the game
                try:
                    # print(f"  Playing: {name1} ({rating1:.0f}) vs {name2} ({rating2:.0f})")
                    result = play(model1, model2)
                    round_games += 1
                    total_games += 1

                    if result == 1:
                        win_table[i, j] += 1
                        win_table[j, i] -= 1
                    elif result == 2:
                        win_table[i, j] -= 1
                        win_table[j, i] += 1

                    # Update ratings
                    new_rating1, new_rating2 = update_elo(rating1, rating2, result, k_factor)

                    elo_ratings[name1] = new_rating1
                    elo_ratings[name2] = new_rating2

                except Exception as e:
                    print(f"\nERROR during game between {name1} and {name2}: {e}")
                    print("Skipping this game.")

    #print("-" * 30)
    print(f" -- Tournament Finished! Total games played: {total_games}")

    # Print win table
    print("Win Table:")
    header = "     " + " ".join(f"{name[:5]:>5}" for name in name_list)
    print(header)
    for i, name1 in enumerate(name_list):
        row = f"{name1[:5]:>5} " + " ".join(
            f"{win_table[i, j]:>+5}" if i != j else "     " for j in range(num_models)
        )
        print(row)

    # Sort final ratings by Elo score (descending)
    sorted_ratings = dict(sorted(elo_ratings.items(), key=lambda item: item[1], reverse=True))
    return sorted_ratings



def run_fast_tournament(
    models: List[Any],
    num_rounds: int = NUM_ROUNDS,
    model_names: List[str] = None
) -> Dict[str, float]:
    """
    Runs a round-robin tournament.
    """
    from main import play_parallel

    num_models = len(models)
    if num_models < 2:
        print("Need at least two models to run a tournament.")
        return {}

    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(num_models)]
    elif len(model_names) != num_models:
        raise ValueError("Length of model_names must match the number of models.")

    # Initialize Elo ratings
    model_map = {name: model for name, model in zip(model_names, models)}
    name_list = list(model_names) # Keep a consistent order

    print(f"Starting Tournament: {num_models} models, {num_rounds} round(s)...", end="", flush=True)

    total_games = 0
    win_table = np.zeros((num_models, num_models), dtype=int)

    # Iterate through all unique pairs, playing both ways (A vs B and B vs A)
    for i in range(num_models):
        for j in range(num_models):
            if i == j:
                continue # Models don't play against themselves

            name1, name2 = name_list[i], name_list[j]
            model1, model2 = model_map[name1], model_map[name2]

            (w1, w2, draws) = play_parallel(model1, model2, num_rounds)
            total_games += num_rounds

            win_table[i, j] += w1
            win_table[j, i] += w2

    print(f" -- Tournament Finished! Total games played: {total_games}")

    # Print win table
    print("Win Table:")
    header = "     " + " ".join(f"{name[:5]:>5}" for name in name_list)
    print(header)
    for i, name1 in enumerate(name_list):
        row = f"{name1[:5]:>5} " + " ".join(
            f"{win_table[i, j]:>+5}" if i != j else "     " for j in range(num_models)
        )
        print(row)

    # Print win table as percentages
    ng = num_rounds * 2
    print("\nWin Percentages:")
    header = "     " + " ".join(f"{name[:5]:>5}" for name in name_list)
    print(header)
    for i, name1 in enumerate(name_list):
        row = f"{name1[:5]:>5} " + " ".join(
            f"{(win_table[i, j] / ng * 100):>5.1f}%" if i != j else "     " for j in range(num_models)
        )
        print(row)

    total_wins = win_table.sum(axis=1)
    
    model_wins = dict(zip(name_list, total_wins))

    # Sort final ratings by total wins (descending)
    sorted_ratings = dict(sorted(model_wins.items(), key=lambda item: item[1], reverse=True))
    return sorted_ratings


def win_rate(model1, model2, num_games=500):
    """
    Simulates a number of games between two models and returns the win rate of model1.
    """
    from main import play_parallel2

    (w1, w2, dr)    = play_parallel2(model1, model2, num_games // 2)
    (w2b, w1b, drb) = play_parallel2(model2, model1, num_games // 2)
    n = 2 * (num_games // 2)

    return (w1 + w1b) / n, (dr + drb) / n


@click.command()
@click.argument('model_names', nargs=-1)
@click.option('-n', '--num-games', default=300, help='Number of games to play in the tournament.')
def main_run(model_names, num_games=300):
    from model import load_frozen_model
    from main import show_tournament, init_device
    device = init_device(False)
    
    models = [load_frozen_model(name).to(device) for name in model_names]

    for m in models:
        print(f"Model: {m.__class__.__name__} -- {sum(p.numel() for p in m.parameters())} parameters")
        #for p in m.parameters():
        #    print(f"  {p.shape} - {p.dtype} - {p.numel()} parameters")
    
    #import pyinstrument
    #with pyinstrument.profile():
    show_tournament(models, model_names, num_games=num_games)
    #results = run_fast_tournament(models, num_rounds=300, model_names=model_names)

def benchmark_run():
    from model import RandomConnect4
    from main import play_parallel, play_parallel2, init_device
    device = init_device(False)

    model = RandomConnect4()
    
    import pyinstrument
    with pyinstrument.profile():
        print(play_parallel2(model, model, 10000))

if __name__ == '__main__':
    main_run()
    #benchmark_run()