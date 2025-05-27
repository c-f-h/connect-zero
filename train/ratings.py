import glicko2
import math
from collections import defaultdict

from main import play_parallel2
from model import load_frozen_model

# --- Glicko Player Wrapper ---
class GlickoAgent:
    def __init__(self, name, model_obj, initial_rating=1500, initial_rd=350, initial_vol=0.06):
        self.name = name
        self.model = model_obj # The actual NN model object
        self.glicko_player = glicko2.Player(rating=initial_rating, rd=initial_rd, vol=initial_vol)
        self.games_played_in_cycle = 0
        self.opponents_in_cycle = [] # To store (opponent_glicko_player, outcome)

    def get_rating(self): return self.glicko_player.getRating()
    def get_rd(self): return self.glicko_player.getRd()

    def record_game_result(self, opponent_glicko_player_at_start_of_cycle, outcome_score):
        """
        Store game results for batch update at the end of the cycle.
        opponent_glicko_player_at_start_of_cycle: A glicko2.Player object representing opponent's state AT THE START of this cycle.
        outcome_score: 1.0 for win, 0.5 for draw, 0.0 for loss.
        """
        self.opponents_in_cycle.append({
            "opponent_rating": opponent_glicko_player_at_start_of_cycle.getRating(),
            "opponent_rd": opponent_glicko_player_at_start_of_cycle.getRd(),
            "outcome": outcome_score
        })
        self.games_played_in_cycle +=1

    def update_rating_at_end_of_cycle(self):
        """
        Updates the Glicko rating based on all games played in the current cycle.
        This should be called AFTER all games in a cycle are played and results recorded.
        """
        if not self.opponents_in_cycle:
            # If agent didn't play, only its RD increases over time (Glicko-2 handles this implicitly if no games provided)
            self.glicko_player.did_not_compete()
        else:
            ratings = [game["opponent_rating"] for game in self.opponents_in_cycle]
            rds = [game["opponent_rd"] for game in self.opponents_in_cycle]
            outcomes = [game["outcome"] for game in self.opponents_in_cycle]
            self.glicko_player.update_player(ratings, rds, outcomes) # Glicko-2

        # Reset for next cycle
        self.games_played_in_cycle = 0
        self.opponents_in_cycle = []

    def __repr__(self):
        return f"{self.name} (R: {self.get_rating():.0f}, RD: {self.get_rd():.0f})"


# --- Tournament Scheduler ---
class DynamicTournament:
    def __init__(self, models_dict, games_per_batch=10,
                 w_diff=1.0, w_rd=1.0, w_tslp=1.0,
                 diff_scale_factor=200.0, rd_scale_factor=200.0, never_played_bonus_tslp=5.0):
        """
        models_dict: A dictionary {name: model_object}
        """
        self.agents = {name: GlickoAgent(name, model_obj) for name, model_obj in models_dict.items()}
        self.games_per_batch = games_per_batch
        self.current_cycle = 0
        self.last_played_cycle = defaultdict(lambda: -1) # {(name1, name2): cycle_num}, name1 < name2

        # Scoring weights
        self.w_diff = w_diff
        self.w_rd = w_rd
        self.w_tslp = w_tslp
        self.diff_scale_factor = diff_scale_factor # For Score_Diff exponential decay
        self.rd_scale_factor = rd_scale_factor
        self.never_played_bonus_tslp = never_played_bonus_tslp # Added to TSLP score if never played

    def _get_tslp_score(self, agent1_name, agent2_name):
        pair_key = tuple(sorted((agent1_name, agent2_name)))
        last_played = self.last_played_cycle[pair_key]      # may be -1 if never played
        tslp = self.current_cycle - last_played
        return math.log(1 + tslp)

    def _calculate_pair_score(self, agent1, agent2):
        # 1. Rating Proximity Component (Score_Diff)
        delta_mu = abs(agent1.get_rating() - agent2.get_rating())
        score_diff = math.exp(- (delta_mu / self.diff_scale_factor)**2)

        # 2. Combined Uncertainty Component (Score_RD)
        score_rd = (agent1.get_rd() + agent2.get_rd()) / 2.0 # Average RD
        score_rd /= self.rd_scale_factor

        # 3. Freshness Component (Score_TSLP)
        score_tslp = self._get_tslp_score(agent1.name, agent2.name)

        total_score = (self.w_diff * score_diff +
                       self.w_rd * score_rd +
                       self.w_tslp * score_tslp)
        return total_score

    def select_pairs_for_cycle(self, num_pairs_to_select):
        available_agent_names = list(self.agents.keys())

        # Consider all possible unique pairs from available agents
        potential_pairs = []
        for i in range(len(available_agent_names)):
            for j in range(i + 1, len(available_agent_names)):
                name1, name2 = available_agent_names[i], available_agent_names[j]
                agent1, agent2 = self.agents[name1], self.agents[name2]
                score = self._calculate_pair_score(agent1, agent2)
                potential_pairs.append(((name1, name2), score))

        # Sort pairs by score in descending order
        potential_pairs.sort(key=lambda x: x[1], reverse=True)

        # Select top pairs, ensuring no agent plays more than once per cycle
        agents_in_selected_pairs = set()
        final_selected_pairs = [] # List of (agent1_name, agent2_name)

        for pair_names, score in potential_pairs:
            if len(final_selected_pairs) >= num_pairs_to_select:
                break
            name1, name2 = pair_names
            if name1 not in agents_in_selected_pairs and name2 not in agents_in_selected_pairs:
                final_selected_pairs.append(pair_names)
                agents_in_selected_pairs.add(name1)
                agents_in_selected_pairs.add(name2)
        return final_selected_pairs


    def run_cycle(self, num_concurrent_batches, play_fn):
        self.current_cycle += 1
        print(f"\n--- Tournament Cycle: {self.current_cycle} ---")

        # 1. Select pairs
        pairs_to_play_names = self.select_pairs_for_cycle(num_concurrent_batches)
        if not pairs_to_play_names:
            raise Exception("No suitable pairs found to play in this cycle.")

        #print(f"Selected pairs for cycle {self.current_cycle}: {pairs_to_play_names}")

        # 2. Play batches
        for name1, name2 in pairs_to_play_names:
            agent1 = self.agents[name1]
            agent2 = self.agents[name2]

            print(f"  Playing: {name1} vs {name2} ({self.games_per_batch} games)", end="", flush=True)
            model1_obj = agent1.model
            model2_obj = agent2.model

            # Call your actual game playing function
            m1_wins, m2_wins, draws = play_fn(model1_obj, model2_obj, self.games_per_batch)
            print(f" - result: {m1_wins} wins, {m2_wins} losses, {draws} draws")

            for _ in range(m1_wins):
                agent1.record_game_result(agent2.glicko_player, 1.0)
                agent2.record_game_result(agent1.glicko_player, 0.0)
            for _ in range(m2_wins):
                agent1.record_game_result(agent2.glicko_player, 0.0)
                agent2.record_game_result(agent1.glicko_player, 1.0)
            for _ in range(draws):
                agent1.record_game_result(agent2.glicko_player, 0.5)
                agent2.record_game_result(agent1.glicko_player, 0.5)

            # Update last played time
            pair_key = tuple(sorted((name1, name2)))
            self.last_played_cycle[pair_key] = self.current_cycle

        # 3. Update Glicko ratings for ALL agents that participated (or all for RD increase)
        print("\nUpdating Glicko ratings...")
        for agent in self.agents.values():
            # Update the Glicko rating for this agent
            agent.update_rating_at_end_of_cycle()


    def print_rankings(self):
        print("\n--- Current Rankings ---")
        sorted_agents = sorted(self.agents.values(), key=lambda ag: ag.get_rating(), reverse=True)
        for i, agent in enumerate(sorted_agents):
            print(f"{i+1}. {agent.name:<15} Rating: {agent.get_rating():.0f} (RD: {agent.get_rd():.0f})")
        print("------------------------")

    def get_agent_ratings_df(self):
        """Returns a pandas DataFrame with current ratings."""
        import pandas as pd
        data = []
        for name, agent in self.agents.items():
            data.append({
                "name": name,
                "rating": agent.get_rating(),
                "rd": agent.get_rd(),
            })
        return pd.DataFrame(data).sort_values(by="rating", ascending=False).reset_index(drop=True)

# --- Example Usage ---
if __name__ == "__main__":
    model_names = (
                #  [f"CNN-Mk4:model-mk4-slf{i}.pth" for i in range(1, 6) ]
                #+ [f"CNN-Mk4:model-mk4-a2c-cp{i}.pth" for i in range(6, 23)]
                #  [f"CNN-Mk4:model-mk4-a2c-b{i}.pth" for i in range(1, 5)]
                  [f"CNN-Mk4:selfplay/{i:04d}.pth" for i in range(1, 48)]
    )
    print(f"Loading {len(model_names)} models")
    models_map = {name: load_frozen_model(name) for name in model_names}

    # Initialize tournament
    # Weights: RD slightly more important, then TSLP, then Diff
    tournament = DynamicTournament(
        models_map,
        games_per_batch=20, # Number of games in one A vs B match
        w_diff=1.0,         # Weight for rating difference (prefer closer)
        w_rd=1.0,           # Weight for rating deviation (prefer uncertain)
        w_tslp=0.2,         # Weight for time since last played
        diff_scale_factor=150.0, # Characteristic rating difference for exp decay
    )

    num_tournament_cycles = 20
    num_concurrent_batches_per_cycle = len(models_map) // 2 # Maximize participation

    for i in range(num_tournament_cycles):
        tournament.run_cycle(num_concurrent_batches_per_cycle, play_parallel2)
        tournament.print_rankings()
        if i % 5 == 0 or i == num_tournament_cycles - 1:
            df = tournament.get_agent_ratings_df()
            if df is not None:
                print(df)
        # Optional: Check for convergence (e.g., average RD below a threshold)
        #avg_rd = sum(agent.get_rd() for agent in tournament.agents.values()) / len(tournament.agents)
        #print(f"Average RD: {avg_rd:.2f}")
        #if avg_rd < 50 and i > 5: # Example convergence criteria
        #    print("Converged (average RD low).")
        #    # break


    print("\nFinal Rankings:")
    tournament.print_rankings()
    final_df = tournament.get_agent_ratings_df()
    if final_df is not None:
        print(final_df)