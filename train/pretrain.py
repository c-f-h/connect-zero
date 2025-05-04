import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import time
from typing import Optional, Tuple

from model import *

def load_puzzleset(filename: str = 'puzzles.dat'):
    if not os.path.exists(filename):
        raise IOError(f"Error: Puzzle file not found at {filename}")

    boards, multi_hot_moves, goals = torch.load(filename)
    print(f"Loaded {boards.shape[0]} puzzles from {filename}")

    multi_hot_moves = multi_hot_moves.float()
    goals = goals.float()

    return TensorDataset(boards, multi_hot_moves, goals)


def pretrain_on_puzzles(
    model: nn.Module,
    optimizer: optim.Optimizer,
    puzzle_file: str,
    epochs: int = 10,
    batch_size: int = 64,
    value_loss_weight: float = 0.5, # Set to 0 to disable value head training
    device: Optional[torch.device] = None,
    ignore_value_head: bool = False # Explicitly disable value head training
):
    """
    Pre-trains a model on a dataset of Connect-4 tactical puzzles.

    Trains the policy head using BCEWithLogitsLoss on multi-hot move targets.
    Optionally trains the value head using MSELoss on goal targets (1=win, 0=block).

    Args:
        model: The PyTorch model (expecting policy_logits output,
               optionally value_estimate as second output).
        optimizer: The PyTorch optimizer configured for the model.
        puzzle_file: Path to the .dat file containing puzzle tensors.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        value_loss_weight: Weight for the value loss component. If 0 or
                           ignore_value_head is True, value head is not trained.
        device: The torch device (e.g., torch.device('cuda') or torch.device('cpu')).
                If None, attempts to auto-detect CUDA.
        ignore_value_head: If True, forces skipping value head training, even if
                           value_loss_weight > 0 and model returns two outputs.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Load Data ---
    dataset = load_puzzleset(puzzle_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # --- 2. Setup ---
    model.to(device)
    model.train() # Set model to training mode

    policy_loss_fn = nn.BCEWithLogitsLoss()
    value_loss_fn = nn.MSELoss()

    # Determine if model has a value head based on its forward output
    # Run a dummy forward pass to check output structure
    train_value_head = False
    if not ignore_value_head and value_loss_weight > 0:
        try:
            with torch.no_grad():
                dummy_input = dataset[:2][0].to(device) # Need batch dim > 1 in case of batchnorm
                output = model(dummy_input)
                if isinstance(output, tuple) and len(output) == 2:
                    train_value_head = True
                    print("Model appears to have policy and value heads. Training both.")
                else:
                    print("Model appears to only have a policy head, or value head training is disabled.")
                    print("Only training the policy head.")
        except Exception as e:
            print(f"Could not determine model output structure: {e}. Training policy head only.")
            train_value_head = False
            ignore_value_head = True # Force ignore if check fails

    if ignore_value_head:
        train_value_head = False
        print("Explicitly ignoring value head training.")
    elif value_loss_weight <= 0:
         train_value_head = False
         print("Value loss weight is 0. Only training the policy head.")

    # --- 3. Training Loop ---
    start_time = time.time()
    for epoch in range(epochs):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0

        for batch_boards, batch_moves, batch_goals in dataloader:
            batch_boards = batch_boards.to(device)
            batch_moves = batch_moves.to(device) # Target for policy head [N, C] float
            batch_goals = batch_goals.to(device) # Base for value target [N] float

            optimizer.zero_grad()

            # --- Forward Pass ---
            model_output = model(batch_boards)

            policy_logits: torch.Tensor
            value_estimate: Optional[torch.Tensor] = None

            if train_value_head:
                policy_logits, value_estimate = model_output
            else:
                policy_logits = model_output # Assume single output

            # --- Calculate Policy Loss ---
            # Target is batch_moves (multi-hot float)
            policy_loss = policy_loss_fn(policy_logits, batch_moves)

            # --- Calculate Value Loss (Optional) ---
            value_loss = torch.tensor(0.0, device=device) # Default to 0
            if train_value_head and value_estimate is not None:
                # Target: Use goals directly (1.0 for win, 0.0 for block)
                # Squeeze value_estimate if it has shape [N, 1]
                value_targets = batch_goals
                if value_estimate.ndim == value_targets.ndim + 1 and value_estimate.shape[-1] == 1:
                   value_estimate_squeezed = value_estimate.squeeze(-1)
                else:
                   value_estimate_squeezed = value_estimate # Assume shapes match

                value_loss = value_loss_fn(value_estimate_squeezed, value_targets)

            # --- Combine Losses ---
            total_loss = policy_loss + value_loss_weight * value_loss

            # --- Backward Pass & Optimize ---
            total_loss.backward()
            optimizer.step()

            # --- Track Loss ---
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item() # Will be 0 if value head not trained
            epoch_total_loss += total_loss.item()
            num_batches += 1

        # --- End of Epoch Logging ---
        avg_policy_loss = epoch_policy_loss / num_batches
        avg_value_loss = epoch_value_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} | Time: {time.time()-start_time:.2f}s | "
              f"Avg Total Loss: {avg_total_loss:.4f} | "
              f"Avg Policy Loss: {avg_policy_loss:.4f} | "
              f"Avg Value Loss: {avg_value_loss:.4f}")
        start_time = time.time() # Reset timer for next epoch

    print("Pre-training finished.")
    model.eval() # Set model back to evaluation mode


def check_puzzle_stats(model: nn.Module, dataset: TensorDataset):
    from main import mask_invalid_moves_batch

    dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
    model.eval()
    
    all_policies = []
    all_values = []

    with torch.no_grad():
        for batch_boards, batch_moves, batch_goals in dataloader:
            output = model(batch_boards)
            if isinstance(output, tuple) and len(output) == 2:
                policy, values = output
            else:
                policy = output
                values = torch.zeros(policy.shape[0], device=policy.device) # Dummy values if no value head
            all_policies.append(policy)
            all_values.append(values)
    logits = torch.cat(all_policies, dim=0)
    values = torch.cat(all_values, dim=0)
    ns = logits.shape[0]
    
    masked_logits = mask_invalid_moves_batch(dataset[:][0], logits, mask_value=-1e9)
    
    probs = F.softmax(logits, dim=-1)
    
    entropy = -(probs * torch.log(probs)).sum(dim=1)
    print(f'Entropy: {entropy.mean().item():.4f}')
    
    stoch_moves = torch.multinomial(probs, 1)       # (B, 1)
    stoch_success = torch.gather(dataset[:][1], dim=1, index=stoch_moves).squeeze(1)      # (B, C) -> (B, 1) -> (B,)  (choose logprobs of actually taken actions)
    print(f'Stochastic success rate:    {stoch_success.sum().item() / ns:.3f}')
    
    det_moves = torch.argmax(probs, dim=1).unsqueeze(-1)
    det_success = torch.gather(dataset[:][1], dim=1, index=det_moves).squeeze(1)      # (B, C) -> (B, 1) -> (B,)  (choose logprobs of actually taken actions)
    print(f'Deterministic success rate: {det_success.sum().item() / ns:.3f}')
    print()
    
    win_samples, block_samples = torch.where(dataset[:][2] == 1)[0], torch.where(dataset[:][2] != 1)[0]
    nws, nbs = len(win_samples), len(block_samples)
    
    print(f'Stochastic success rate (winning moves):     {stoch_success[win_samples].sum().item() / nws:.3f}')
    print(f'Deterministic success rate (winning moves):  {det_success[win_samples].sum().item() / nws:.3f}')
    print(f'Stochastic success rate (blocking moves):    {stoch_success[block_samples].sum().item() / nbs:.3f}')
    print(f'Deterministic success rate (blocking moves): {det_success[block_samples].sum().item() / nbs:.3f}')
    print()
    
    win_value_error = torch.abs(1.0 - values[win_samples])
    print(f'Value error (winning moves) - mean:          {win_value_error.mean().item():.4f}')
    print(f'Value error (winning moves) - std:           {win_value_error.std().item():.4f}')
   



def main_checkstats():
    import sys
    from model import load_frozen_model
    model = load_frozen_model(sys.argv[1])
    dataset = load_puzzleset()
    check_puzzle_stats(model, dataset)



def main_pretrain():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    model = Connect4CNN_Mk4(value_head=True).to(DEVICE)
    #model.load_state_dict(torch.load('pretrained.pth')['model_state_dict'])

    learning_rate = 1e-4
    weight_decay = 0 #3e0
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 3. Run Pre-training
    pretrain_on_puzzles(
        model=model,
        optimizer=optimizer,
        puzzle_file='puzzles.dat',
        epochs=25,
        batch_size=1000,
        value_loss_weight=0.5, # Train both heads
        device=DEVICE,
        ignore_value_head=False
    )
    torch.save({
        'model_state_dict': model.state_dict()
    }, 'pretrained.pth')

if __name__ == '__main__':
    main_checkstats()
    #main_pretrain()
