import numpy as np
import torch
import random
from typing import List, Dict, Optional
from collections import defaultdict

from main import mask_invalid_moves_batch, play_multiple_against_model
from model import load_frozen_model


def collect_states_from_random_play(
    model_list: List[torch.nn.Module],
    num_pairings: int,
    games_per_pairing: int,
) -> List[torch.Tensor]:
    """
    Collects board states by playing games between randomly chosen pairs of models.

    Args:
        model_list: A list of neural network policy models.
        num_pairings: The number of random model pairings to simulate.
        games_per_pairing: The number of games to play for each chosen pair.

    Returns:
        A list of unique board state tensors collected from all games.
    """
    if not model_list or len(model_list) < 2:
        raise ValueError("Not enough models in model_list to form pairings. Need at least 2.")

    all_collected_states: List[torch.Tensor] = []
    unique_state_hashes = set() # To keep track of unique states efficiently

    print(f"Starting state collection from {num_pairings} random pairings, {games_per_pairing} games each.")

    for i in range(num_pairings):
        # Randomly choose two different models
        model_a_idx, model_b_idx = random.sample(range(len(model_list)), 2)
        model_a = model_list[model_a_idx]
        model_b = model_list[model_b_idx]

        print(f"Pairing {i+1}/{num_pairings}: Model {model_a_idx} vs Model {model_b_idx}")

        # Play games and get board states
        board_states_from_pairing, _, _, _, _ = play_multiple_against_model(
            model_a, model_b, num_games=games_per_pairing
        )

        for board_tensor in board_states_from_pairing:
            # --- Uniqueness Check ---
            board_bytes = board_tensor.numpy().tobytes()

            # Using board_bytes for hashing
            if board_bytes not in unique_state_hashes:
                unique_state_hashes.add(board_bytes)
                all_collected_states.append(board_tensor.clone()) # Store a clone

        print(f"  Collected {len(board_states_from_pairing)} raw states, now {len(all_collected_states)} unique states total.")

    print(f"Finished collection. Total unique states collected: {len(all_collected_states)}")
    return all_collected_states


def get_piece_count(board_tensor: torch.Tensor) -> int:
    return int(abs(board_tensor).sum().item())

def stratified_sample_states(
    collected_states_pool: List[torch.Tensor],
    bin_edges: Optional[List[int]] = None,
    samples_per_stratum: Optional[int] = None,
    target_total_samples: Optional[int] = None,
    allow_fewer_samples: bool = True
) -> List[torch.Tensor]:
    """
    Performs stratified sampling on a list of board states based on piece count.

    Args:
        collected_states_pool: List of unique board state tensors.
        bin_edges: A list of integers defining the boundaries of the strata.
                   e.g., [0, 11, 21, 31, 43] creates strata for piece counts
                   [0-10], [11-20], [21-30], [31-42].
                   The last edge should be exclusive (i.e., > max possible pieces).
        samples_per_stratum: The desired number of samples from each stratum.
        target_total_samples: The desired total number of samples. If provided,
                              samples_per_stratum is ignored. Samples will be
                              distributed as evenly as possible across strata.
        allow_fewer_samples: If True, and a stratum has fewer states than requested,
                             all states from that stratum will be taken. If False,
                             an error might be raised or fewer total samples returned
                             (currently implies returning fewer overall if a stratum is deficient
                              when samples_per_stratum is set).

    Returns:
        A list of sampled board state tensors.
    """
    if not collected_states_pool:
        return []

    if bin_edges is None:
        raise ValueError("bin_edges must be provided.")
    if samples_per_stratum is None and target_total_samples is None:
        raise ValueError("Either samples_per_stratum or target_total_samples must be provided.")
    if samples_per_stratum is not None and target_total_samples is not None:
        print("Warning: Both samples_per_stratum and target_total_samples provided. Using target_total_samples.")
        samples_per_stratum = None

    if not bin_edges or len(bin_edges) < 2:
        raise ValueError("bin_edges must contain at least two elements.")
    if not all(bin_edges[i] < bin_edges[i+1] for i in range(len(bin_edges)-1)):
        raise ValueError("bin_edges must be strictly increasing.")

    # 1. Stratify the states
    stratified_states: Dict[int, List[torch.Tensor]] = defaultdict(list)
    # Map piece counts to stratum index
    # Stratum i corresponds to piece counts in [bin_edges[i], bin_edges[i+1] - 1]
    for board_tensor in collected_states_pool:
        count = get_piece_count(board_tensor)
        assigned_stratum = -1
        for i in range(len(bin_edges) - 1):
            # Bin i includes states with piece_count >= bin_edges[i] and < bin_edges[i+1]
            if bin_edges[i] <= count < bin_edges[i+1]:
                assigned_stratum = i
                break
        if assigned_stratum != -1:
            stratified_states[assigned_stratum].append(board_tensor)
        # else: # State does not fall into any bin (should not happen if bins cover range)
        #     print(f"Warning: State with {count} pieces did not fall into any bin defined by {bin_edges}")


    sampled_states: List[torch.Tensor] = []
    actual_strata_indices = sorted(stratified_states.keys()) # Strata that actually have states

    if not actual_strata_indices:
        print("Warning: No states fell into any defined strata.")
        return []

    # 2. Sample from strata
    if samples_per_stratum is not None:
        print(f"Sampling up to {samples_per_stratum} states per stratum.")
        for stratum_idx in actual_strata_indices:
            states_in_stratum = stratified_states[stratum_idx]
            num_to_sample = min(len(states_in_stratum), samples_per_stratum)

            if num_to_sample < samples_per_stratum and not allow_fewer_samples:
                 print(f"Warning: Stratum {stratum_idx} (pieces {bin_edges[stratum_idx]}-{bin_edges[stratum_idx+1]-1}) has only {len(states_in_stratum)} states, less than requested {samples_per_stratum}.")

            if num_to_sample > 0:
                sampled_states.extend(random.sample(states_in_stratum, num_to_sample))
            print(f"  Stratum {stratum_idx} (pieces {bin_edges[stratum_idx]}-{bin_edges[stratum_idx+1]-1}): has {len(states_in_stratum)} states, sampled {num_to_sample}.")

    elif target_total_samples is not None:
        print(f"Attempting to sample a total of {target_total_samples} states.")
        num_non_empty_strata = len(actual_strata_indices)
        if num_non_empty_strata == 0: return []

        # Calculate base samples per non-empty stratum and remainder
        base_samples_per_actual_stratum = target_total_samples // num_non_empty_strata
        remainder_samples = target_total_samples % num_non_empty_strata

        temp_sampled_counts = {idx: 0 for idx in actual_strata_indices}

        # First pass: sample base_samples_per_actual_stratum or available
        for stratum_idx in actual_strata_indices:
            states_in_stratum = stratified_states[stratum_idx]
            num_to_sample = min(len(states_in_stratum), base_samples_per_actual_stratum)
            if num_to_sample > 0:
                sampled_states.extend(random.sample(states_in_stratum, num_to_sample))
                temp_sampled_counts[stratum_idx] = num_to_sample
            print(f"  Stratum {stratum_idx} (pieces {bin_edges[stratum_idx]}-{bin_edges[stratum_idx+1]-1}): has {len(states_in_stratum)}, base sample {num_to_sample}.")


        # Second pass: distribute remainder samples
        # Give one extra to strata that can provide it, cycling through them
        eligible_strata_for_remainder = [
            idx for idx in actual_strata_indices
            if len(stratified_states[idx]) > temp_sampled_counts[idx]
        ]
        random.shuffle(eligible_strata_for_remainder) # Randomize order for distributing remainder

        for i in range(remainder_samples):
            if not eligible_strata_for_remainder:
                break # No more states or strata to pick from

            stratum_to_add_idx = eligible_strata_for_remainder[i % len(eligible_strata_for_remainder)]

            # Find an unpicked state from this stratum
            current_samples_from_stratum = [s for s in sampled_states if get_piece_count(s) >= bin_edges[stratum_to_add_idx] and get_piece_count(s) < bin_edges[stratum_to_add_idx+1]]

            available_for_remainder = [s for s in stratified_states[stratum_to_add_idx] if not any(torch.equal(s, cs) for cs in current_samples_from_stratum)]

            if available_for_remainder:
                sampled_states.append(random.choice(available_for_remainder))
                temp_sampled_counts[stratum_to_add_idx] += 1
                # If this stratum can no longer provide more, remove it from future consideration in this remainder loop
                if len(available_for_remainder) == 1: # was the last one
                     eligible_strata_for_remainder = [idx for idx in eligible_strata_for_remainder if idx != stratum_to_add_idx or len(stratified_states[idx]) > temp_sampled_counts[idx]]

            else: # Should not happen if eligibility check is correct
                pass

        # If we still haven't met target_total_samples (e.g. pool too small)
        if len(sampled_states) < target_total_samples and allow_fewer_samples:
            print(f"Warning: Could only sample {len(sampled_states)} states, less than target {target_total_samples} due to pool size/distribution.")
        elif len(sampled_states) < target_total_samples and not allow_fewer_samples:
            # This case is tricky; current logic prioritizes getting as close as possible.
            # Strict "fail if not exact" would require different handling.
            print(f"Error/Warning: Could not meet target_total_samples strictly. Sampled {len(sampled_states)}.")


    random.shuffle(sampled_states) # Shuffle the final list
    print(f"Total states sampled: {len(sampled_states)}")
    return sampled_states


def compute_similarity_states(models: List[torch.nn.Module]):
    num_model_pairings = 100    # How many times to pick two random models
    num_games_per_pairing = 20 # Games they play against each other (2 each way)

    collected_states_pool = collect_states_from_random_play(
        model_list=models,
        num_pairings=num_model_pairings,
        games_per_pairing=num_games_per_pairing,
    )

    bin_edges = list(range(0, 45, 4))
    states = stratified_sample_states(
        collected_states_pool=collected_states_pool,
        bin_edges=bin_edges,
        samples_per_stratum=750,
        allow_fewer_samples=True
    )

    print(f"Sampled {len(states)} states from the pool.")
    stacked_states = torch.stack(states)
    print(f"Stacked states shape: {stacked_states.shape}")
    torch.save(stacked_states, "similarity_states.pt")
    return stacked_states


def compute_model_similarity(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    similarity_states: torch.Tensor
) -> float:
    """
    Computes the similarity between two models based on their predictions on a set of states.
    """
    with torch.no_grad():
        logits_a = model_a(similarity_states)        # ignore value head
        logits_b = model_b(similarity_states)
        if isinstance(logits_a, tuple):
            logits_a = logits_a[0]
        if isinstance(logits_b, tuple):
            logits_b = logits_b[0]

    # Compute probabilities using softmax
    logits_a = mask_invalid_moves_batch(similarity_states, logits_a)
    logits_b = mask_invalid_moves_batch(similarity_states, logits_b)
    prob_a = torch.softmax(logits_a, dim=1)
    prob_b = torch.softmax(logits_b, dim=1)

    # Compute the mean distribution (M)
    mean_prob = 0.5 * (prob_a + prob_b)

    # Compute Jensen-Shannon divergence
    kl_a = torch.sum(prob_a * (torch.log(prob_a + 1e-10) - torch.log(mean_prob + 1e-10)), dim=1)
    kl_b = torch.sum(prob_b * (torch.log(prob_b + 1e-10) - torch.log(mean_prob + 1e-10)), dim=1)
    js_divergence = 0.5 * (kl_a + kl_b)

    # Return the mean JS divergence
    return js_divergence.mean().item()

def compute_similarity_matrix(models: List[torch.nn.Module], verbose=False) -> np.ndarray:
    """
    Computes a similarity matrix for a list of models based on their predictions on a set of states.
    """
    num_models = len(models)
    similarity_matrix = np.zeros((num_models, num_models))

    for m in models:
        m.eval()

    similarity_states = torch.load("similarity_states.pt")
    #print(f"Loaded {similarity_states.shape[0]} similarity states.")

    for i in range(num_models):
        for j in range(i + 1, num_models):
            similarity = compute_model_similarity(models[i], models[j], similarity_states)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
            if verbose:
                print(f"Similarity between model {i} and model {j}: {similarity:.4f}")

    return similarity_matrix


def visualize_similarity(dissimilarity_matrix: np.ndarray, node_labels: List[str]):
    import matplotlib.pyplot as plt
    from sklearn.manifold import MDS, TSNE
    import umap
    import networkx as nx

    n_nodes = dissimilarity_matrix.shape[0]

    # --- 2. Apply Dimensionality Reduction ---

    # Method 1: Multidimensional Scaling (MDS)
    mds = MDS(n_components=2, metric=True, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    # For non-metric MDS:
    #mds = MDS(n_components=2, metric=False, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    pos_mds = mds.fit_transform(dissimilarity_matrix)
    print("\nMDS Stress:", mds.stress_) # Lower is better

    # Method 2: t-SNE
    tsne = TSNE(n_components=2, metric='precomputed', random_state=42,
                perplexity=7, learning_rate='auto', init='random')
    pos_tsne = tsne.fit_transform(dissimilarity_matrix)
    print("\nt-SNE KL Divergence:", tsne.kl_divergence_)

    # Method 3: UMAP (if installed)
    reducer = umap.UMAP(metric='precomputed', densmap=True, n_neighbors=5, min_dist=0.1)
        #n_components=2, metric='precomputed', random_state=42,
        #n_neighbors=5, min_dist=0.1)
    pos_umap = reducer.fit_transform(dissimilarity_matrix)

    # Method 4: Force-Directed Layout (using NetworkX)
    # For this, we usually create a graph where edges represent strong similarity
    # Or, use similarities/distances as weights for spring forces
    #G = nx.Graph()
    #for i in range(n_nodes):
    #    G.add_node(i, label=node_labels[i])

    ## Option A: Add edges based on a similarity threshold
    ## threshold = 0.4
    ## for i in range(n_nodes):
    ##     for j in range(i + 1, n_nodes):
    ##         if similarity_matrix[i, j] > threshold:
    ##             G.add_edge(i, j, weight=similarity_matrix[i, j])

    ## Option B: Create a fully connected graph and use distances as 'lengths' for Kamada-Kawai
    ## Or inverse similarities as 'weights' for Fruchterman-Reingold (where higher weight = stronger attraction)
    #for i in range(n_nodes):
    #    for j in range(i + 1, n_nodes):
    #        if dissimilarity_matrix[i,j] < 1: # Avoid adding edges for completely dissimilar nodes if desired
    #            #G.add_edge(i, j, weight=similarity_matrix[i,j]) # F-R uses weight as attraction
    #            # For Kamada-Kawai, it uses 'distance' attribute (or inverse of weight)
    #            G.add_edge(i, j, distance=dissimilarity_matrix[i,j])

    ## pos_spring = nx.spring_layout(G, weight='weight', seed=42) # Fruchterman-Reingold
    #pos_kk = nx.kamada_kawai_layout(G, dist=None, weight=None) # Uses graph distance if dist not given.
    #                                                    # If your `d_ij` are good path distances, you can pass them.
    #                                                    # Often needs a connected graph.

    # For Kamada-Kawai, it's better if we provide the dissimilarity matrix directly if possible,
    # but NetworkX's implementation expects a graph structure.
    # A workaround is to set edge attributes that kamada_kawai_layout can use.
    # Let's make a dense graph where edge 'distance' is our dissimilarity:
    G_kk = nx.Graph()
    dist_dict = {}
    for i in range(n_nodes):
        G_kk.add_node(i, label=node_labels[i])
        dist_dict[i] = {}
        for j in range(n_nodes):
            if i != j:
                # Kamada-Kawai expects shorter distance for closer nodes
                dist_dict[i][j] = dissimilarity_matrix[i,j] if dissimilarity_matrix[i,j] > 0 else 1e-6 # Avoid zero distance
#
    pos_kk_custom_dist = nx.kamada_kawai_layout(G_kk, dist=dist_dict)


    # --- 3. Visualize ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Low-Dimensional Visualization of Node Similarities', fontsize=16)

    plots_data = [
        (pos_mds, "MDS", axs[0, 0]),
        (pos_tsne, "t-SNE", axs[0, 1]),
        (pos_umap, "UMAP", axs[1, 0]) if pos_umap is not None else (None, "", None),
        (pos_kk_custom_dist, "Kamada-Kawai (Custom Dist)", axs[1,1])
    ]
    # if pos_umap is None: # adjust subplot if UMAP is missing
    #    plots_data[2] = (pos_kk_custom_dist, "Kamada-Kawai (Custom Dist)", axs[1,0])
    #    fig.delaxes(axs[1,1]) # remove unused subplot

    for pos, title, ax in plots_data:
        if pos is None or ax is None:
            if ax: fig.delaxes(ax) # remove axis if no plot
            continue
        if isinstance(pos, dict): # NetworkX returns a dict
            x_coords = [p[0] for p in pos.values()]
            y_coords = [p[1] for p in pos.values()]
        else: # Scikit-learn returns an array
            x_coords = pos[:, 0]
            y_coords = pos[:, 1]

        ax.scatter(x_coords, y_coords)
        for i, label in enumerate(node_labels):
            ax.annotate(label, (x_coords[i], y_coords[i]), textcoords="offset points", xytext=(0,5), ha='center')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":

    model_names = [
        f"CNN-Mk4:model-mk4-slf{i}.pth" for i in range(1, 4)
    ] + [f"CNN-Mk4:model-mk4-a2c-cp{i}.pth" for i in range(6, 23)]
    print(f"Loading {len(model_names)} models")
    models = [load_frozen_model(name) for name in model_names]

    #compute_similarity_states(models)

    #similarity_matrix = compute_similarity_matrix(models, verbose=True)
    #torch.save(similarity_matrix, "similarity_matrix.pt")

    similarity_matrix = torch.load("similarity_matrix.pt")

    print("Similarity Matrix:")
    print(similarity_matrix)

    labels = [n.replace('.', '-').split('-')[-2] for n in model_names]

    visualize_similarity(similarity_matrix, labels)
