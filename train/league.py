import os
import numpy as np
import torch
import json

from model import load_frozen_model

def find_first_free_model_index(dir):
    i = 1
    os.makedirs(dir, exist_ok=True)
    while True:
        name = os.path.join(dir, f"{i:04d}.pth")
        if not os.path.exists(name):
            return i
        i += 1

class League:
    def __init__(self, model_names, dir, model_string, device, max_league_size=6):
        self.device = device

        leaguedata = None
        if model_names is None:
            with open(os.path.join(dir, "league.json"), "r") as f:
                print("Loading league from league.json")
                leaguedata = json.load(f)
                model_names = leaguedata['model_names']

        self.model_names = model_names
        self.models = [load_frozen_model(name).to(device) for name in model_names]
        self.max_league_size = max_league_size
        if len(model_names) > max_league_size:
            raise ValueError(f"League size exceeds maximum of {max_league_size}.")
        self.dir = dir
        self.model_string = model_string
        self.next_model_index = find_first_free_model_index(dir)

        if leaguedata is not None:
            self.estimated_wr = np.array(leaguedata['estimated_wr'], dtype=np.float32)
            if len(self.estimated_wr) != len(model_names):
                raise ValueError("Mismatch between number of models and estimated win rates.")
        else:
            self.estimated_wr = 0.5 * np.ones((len(self.models),), dtype=np.float32)
        self.update_opponent_weights()
        self.rng = np.random.default_rng()

    def update_winrate(self, opponent_index, winrate):
        """Update the estimated win rate against a specific opponent (exponential moving average)."""
        self.estimated_wr[opponent_index] = self.estimated_wr[opponent_index] * 0.9 + winrate * 0.1

    def update_opponent_weights(self):
        """Update the weights for opponent selection based on estimated win rates."""
        weights = np.clip(1.0 - self.estimated_wr, 0.05, 1.0)
        self.opponent_weights = weights / weights.sum()

    def choose_opponent(self):
        """Choose an opponent, preferring opponents against which we have lower win rates."""
        return self.rng.choice(len(self.models), p=self.opponent_weights)

    def find_opponent_to_eliminate(self):
        from similarity import compute_similarity_matrix

        print("Computing similarity matrix...")
        similarity = compute_similarity_matrix(self.models)
        print(similarity)
        np.fill_diagonal(similarity, np.inf)
        min_index = np.unravel_index(np.argmin(similarity), similarity.shape)
        # Get opponent from the pair with the worst performance
        idx = min_index[0] if self.estimated_wr[min_index[0]] > self.estimated_wr[min_index[1]] else min_index[1]
        print(f"Most similar opponent pair: ({min_index[0]}, {min_index[1]}) with similarity {similarity[min_index]}. Eliminating {idx}.")
        return idx

    def add_model(self, model):
        """Add a new model to the league."""
        fname = os.path.join(self.dir, f"{self.next_model_index:04d}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
        }, fname)
        print(f"Saving model checkpoint to {fname}")
        self.next_model_index += 1

        idx = self.find_opponent_to_eliminate()
        self.models[idx] = load_frozen_model(self.model_names[idx]).to(self.device)
        self.model_names[idx] = self.model_string + ':' + fname
        self.estimated_wr[idx] = 0.5
        self.update_opponent_weights()

    def save(self):
        """Save the league to a JSON file."""
        with open(os.path.join(self.dir, "league.json"), "w") as f:
            json.dump({
                'model_names': self.model_names,
                'estimated_wr': self.estimated_wr.tolist(),
            }, f, indent=2)
