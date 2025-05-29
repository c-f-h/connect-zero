import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

from board import make_move_and_check, make_move_and_check_batch
from treesearch import estimate_move_values_from_rollout, multiple_rollouts


# Define board dimensions
ROWS = 6
COLS = 7

INPUT_SIZE = ROWS * COLS # 6 * 7 = 42
OUTPUT_SIZE = COLS       # 7 columns to choose from


class SimpleMLPModel(nn.Module):
    """Create a simple MLP model for Connect4."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ROWS * COLS, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, COLS)
        )
    
    def forward(self, x):
        # Store original shape and determine batch size
        original_shape = x.shape
        if x.ndim == 2:
            x = x.unsqueeze(0)

        x = self.layers(x.float())

        # If the original input was a single instance, remove the batch dimension
        if len(original_shape) == 2:
            x = x.squeeze(0)
        return x



class Connect4MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for Connect-4.

    Takes a flattened board state as input and outputs logits
    representing the preference for each column move.
    """
    def __init__(self, input_size=INPUT_SIZE, hidden_size=64, num_layers=1, dropout_rate=0.05, output_size=OUTPUT_SIZE):
        """
        Initializes the layers of the MLP.

        Args:
            input_size (int): The size of the flattened input board state (rows * cols).
            hidden_size (int): The number of neurons in the hidden layer.
            output_size (int): The number of output neurons (number of columns).
        """
        super().__init__() # Initialize the parent nn.Module class
        self.flatten_size = input_size

        layers = [
            nn.Linear(input_size, hidden_size), # Input to hidden layer
            nn.ReLU(),                          # Activation function
        ]
        for k in range(num_layers - 1):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ]
        layers.append(nn.Linear(hidden_size, output_size)) # Hidden to output layer - raw logits

        self.layers = nn.ModuleList(layers) # Store layers in a ModuleList for easy iteration

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor representing the board state(s).
                              Expected shape: [batch_size, rows, cols] or [rows, cols].
                              Expected dtype: Can be int8, but will be cast to float32.

        Returns:
            torch.Tensor: Output logits. Shape: [batch_size, output_size] or [output_size].
        """
        # --- Input Handling ---
        # Store original shape and determine batch size
        original_shape = x.shape
        if x.ndim == 2: # Single board state [rows, cols]
            x = x.unsqueeze(0) # Add a batch dimension: [1, rows, cols]
        elif x.ndim != 3: # Expecting [batch_size, rows, cols]
             raise ValueError(f"Input tensor should have 2 or 3 dimensions (rows, cols) or (batch, rows, cols), got shape {original_shape}")

        batch_size = x.size(0)

        # Convert to float32 if it's not already (Linear layers expect float)
        if not x.is_floating_point():
            x = x.float()

        # Flatten the board state: [batch_size, rows, cols] -> [batch_size, rows * cols]
        x = x.view(batch_size, -1) # -1 infers the size (rows * cols)

        # Apply layers
        for layer in self.layers:
            x = layer(x)

        # If the original input was a single instance, remove the batch dimension
        if len(original_shape) == 2:
            x = x.squeeze(0) # [1, output_size] -> [output_size]
        return x


class RandomConnect4(nn.Module):
    def __init__(self, output_size=OUTPUT_SIZE):
        super().__init__() # Initialize the parent nn.Module class
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2: # Single board state [rows, cols]
            return torch.zeros((self.output_size,), dtype=torch.float32)
        elif x.ndim == 3: # A batch of board states: [batch_size, rows, cols]
            return torch.zeros((x.size(0), self.output_size), dtype=torch.float32)
        else:
             raise ValueError(f"Input tensor should have 2 or 3 dimensions (rows, cols) or (batch, rows, cols), got shape {x.shape}")


@torch.jit.script
def find_best_move(board):
    """
    Finds a winning move for the current player (represented as +1) in the given board state.
    If no winning move is found, moves randomly.

    Returns logits, shape (C,).
    """
    # first check if we have a winning move; then if we can block the opponent's winning move
    cols = board.shape[-1]
    for B in (board, -board):
        for c in range(cols):
            # Check if the move is valid
            if B[0, c] == 0:
                _, win = make_move_and_check(B, c)
                if win:
                    choice = torch.tensor(c, device=board.device)
                    logits = nn.functional.one_hot(choice, num_classes=cols).float()
                    return torch.log(logits + 1e-12) # Avoid log(0) by adding a small epsilon
    return torch.zeros((cols,), dtype=torch.float32, device=board.device)

def find_best_move2(board):
    """
    Finds a winning move for the current player (represented as +1) in the given board state.
    If no winning move is found, moves randomly.

    Returns logits, shape (C,).
    """
    # first check if we have a winning move; then if we can block the opponent's winning move
    cols = board.shape[-1]
    for B in (board, -board):
        valid_moves = torch.where(B[0, :] == 0)[0]
        B_rep = B.repeat(len(valid_moves), 1, 1)  # Repeat the board for each valid move
        B_new, wins, draws = make_move_and_check_batch(B_rep, valid_moves)
        if torch.any(wins):
            logits = torch.full((cols,), -1e12, dtype=torch.float32, device=board.device)
            logits[valid_moves[wins]] = 1.0
            return logits
    return torch.zeros((cols,), dtype=torch.float32, device=board.device)


class RandomPunisher(nn.Module):
    """Plays a winning or blocking move if available, otherwise plays a random move."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store original shape and determine batch size
        original_shape = x.shape
        if x.ndim == 2: # Single board state [rows, cols]
            x = x.unsqueeze(0) # Add a batch dimension: [1, rows, cols]
        elif x.ndim != 3: # Expecting [batch_size, rows, cols]
             raise ValueError(f"Input tensor should have 2 or 3 dimensions (rows, cols) or (batch, rows, cols), got shape {original_shape}")

        batch_size = x.size(0)
        logits = torch.stack([find_best_move(x[i]) for i in range(batch_size)])

        if len(original_shape) == 2:
            logits = logits.squeeze(0)
        return logits
    

def board_to_channels(board: torch.Tensor) -> torch.Tensor:
    """Converts a board tensor (0=empty, 1=p1, -1=p2) to channel representation."""
    device = board.device

    if board.ndim == 2: # Single board [rows, cols]
        board = board.unsqueeze(0) # Add batch dim: [1, rows, cols]

    # Create channels: [batch, channels, rows, cols]
    channels = torch.zeros((board.size(0), 3, ROWS, COLS),
                           dtype=torch.float32, device=device)

    # Channel 0: Player 1's pieces (value 1)
    channels[:, 0, :, :] = (board == 1).float()
    # Channel 1: Player 2's pieces (value -1)
    channels[:, 1, :, :] = (board == -1).float()
    # Channel 2: Empty cells (value 0)
    channels[:, 2, :, :] = (board == 0).float()

    return channels



class Connect4CNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for Connect-4.

    Takes a board state as input (represented with multiple channels)
    and outputs logits representing the preference for each column move.
    """
    def __init__(self, num_filters1=32, num_filters2=64, hidden_fc_size=128, output_size=OUTPUT_SIZE):
        """
        Initializes the layers of the CNN.

        Args:
            num_filters1 (int): Number of filters in the first convolutional layer.
            num_filters2 (int): Number of filters in the second convolutional layer.
            hidden_fc_size (int): Size of the hidden fully connected layer after conv layers.
            output_size (int): Number of output neurons (number of columns).
        """
        super().__init__()
        self.num_input_channels = 3     # my pieces, their pieces, empty cells

        # Convolutional layers
        # Conv2d parameters: in_channels, out_channels, kernel_size, stride, padding
        # Using padding='same' (or calculated padding=1 for kernel=3, stride=1)
        # preserves the height and width through the conv layers.
        self.conv1 = nn.Conv2d(self.num_input_channels, num_filters1, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filters1, num_filters2, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # Calculate the flattened size after conv layers
        # Output shape after conv2: [batch_size, num_filters2, ROWS, COLS]
        self.flattened_size = num_filters2 * ROWS * COLS

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, hidden_fc_size)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_fc_size, output_size)
        # Output raw logits


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor representing the board state(s).
                              Expected shape: [batch_size, rows, cols] or [rows, cols].
                              Values: 0=empty, 1=player 1, -1=player 2.
                              *Crucially, assumes the perspective is normalized*
                              *(i.e., the current player is always +1).*

        Returns:
            torch.Tensor: Output logits. Shape: [batch_size, output_size] or [output_size].
        """
        original_shape = x.shape
        if x.ndim == 2:
            batch_size = 1
            x_single = x.unsqueeze(0) # Add temporary batch dimension
        elif x.ndim == 3:
            batch_size = x.shape[0]
            x_single = x
        else:
             raise ValueError(f"Input tensor should have 2 or 3 dimensions (rows, cols) or (batch, rows, cols), got shape {original_shape}")

        # --- Convert board to channel representation ---
        # Input format expected by Conv2d: [batch_size, channels, height, width]
        x_ch = board_to_channels(x_single) # Shape: [batch_size, num_input_channels, ROWS, COLS]

        # --- Convolutional Layers ---
        x = self.conv1(x_ch)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x) # Shape: [batch_size, num_filters2, ROWS, COLS]

        # --- Fully Connected Layers ---
        x = self.fc1(x.view(batch_size, -1))
        x = self.relu3(x)
        logits = self.fc2(x) # Shape: [batch_size, output_size]

        # --- Output Handling ---
        # If the original input was a single instance, remove the batch dimension
        if len(original_shape) == 2:
            logits = logits.squeeze(0) # [1, output_size] -> [output_size]

        return logits


class Connect4CNN_MLP(nn.Module):
    """
    CNN model with a deep MLP classifier afterwards.
    """
    def __init__(self, num_filters1=32, num_filters2=64, hidden_fc_size=128, num_hidden_layers=1, output_size=OUTPUT_SIZE):
        """
        Initializes the layers of the CNN.
        """
        super().__init__()
        self.num_input_channels = 3     # my pieces, their pieces, empty cells

        # Convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(self.num_input_channels, num_filters1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters1),
            nn.ReLU(),
            nn.Conv2d(num_filters1, num_filters2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters2),
            nn.ReLU(),
        ])

        # Calculate the flattened size after conv layers
        # Output shape after conv2: [batch_size, num_filters2, ROWS, COLS]
        self.flattened_size = num_filters2 * ROWS * COLS

        fc_layers = [
            nn.Linear(self.flattened_size, hidden_fc_size),
            nn.ReLU()
        ]
        
        for _ in range(num_hidden_layers):
            fc_layers += [
                nn.Linear(hidden_fc_size, hidden_fc_size),
                nn.ReLU()
            ]

        fc_layers.append(nn.Linear(hidden_fc_size, output_size)) # Output raw logits
        self.fc_layers = nn.ModuleList(fc_layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        Returns:
            torch.Tensor: Output logits. Shape: [batch_size, output_size] or [output_size].
        """
        original_shape = x.shape
        if x.ndim == 2:
            batch_size = 1
            x = x.unsqueeze(0) # Add temporary batch dimension
        elif x.ndim == 3:
            batch_size = x.shape[0]
        else:
             raise ValueError(f"Input tensor should have 2 or 3 dimensions (rows, cols) or (batch, rows, cols), got shape {original_shape}")

        # --- Convert board to channel representation ---
        # Input format expected by Conv2d: [batch_size, channels, height, width]
        x = board_to_channels(x) # Shape: [batch_size, num_input_channels, ROWS, COLS]

        # --- Convolutional Layers ---
        for layer in self.conv_layers:
            x = layer(x)
        # Shape: [batch_size, num_filters2, ROWS, COLS]

        # reshape to one dimension per batch
        x = x.view(batch_size, -1)

        # --- Fully Connected Layers ---
        for layer in self.fc_layers:
            x = layer(x)

        # --- Output Handling ---
        # If the original input was a single instance, remove the batch dimension
        if len(original_shape) == 2:
            x = x.squeeze(0) # [1, output_size] -> [output_size]

        return x


class Connect4CNN_MLP_v2(nn.Module):
    """
    CNN model with a deep MLP classifier afterwards. With BatchNorm throughout.
    """
    def __init__(self, num_filters1=32, num_filters2=64, hidden_fc_size=128, num_hidden_layers=1, output_size=OUTPUT_SIZE):
        """
        Initializes the layers of the CNN.
        """
        super().__init__()
        self.num_input_channels = 3     # my pieces, their pieces, empty cells

        # Convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(self.num_input_channels, num_filters1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters1),
            nn.ReLU(),
            nn.Conv2d(num_filters1, num_filters2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters2),
            nn.ReLU(),
        ])

        # Calculate the flattened size after conv layers
        # Output shape after conv2: [batch_size, num_filters2, ROWS, COLS]
        self.flattened_size = num_filters2 * ROWS * COLS

        fc_layers = [
            nn.Linear(self.flattened_size, hidden_fc_size, bias=False),
            nn.BatchNorm1d(hidden_fc_size),
            nn.ReLU()
        ]
        
        for _ in range(num_hidden_layers):
            fc_layers += [
                nn.Linear(hidden_fc_size, hidden_fc_size),
                nn.BatchNorm1d(hidden_fc_size),
                nn.ReLU()
            ]

        fc_layers.append(nn.Linear(hidden_fc_size, output_size)) # Output raw logits
        self.fc_layers = nn.ModuleList(fc_layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        Returns:
            torch.Tensor: Output logits. Shape: [batch_size, output_size] or [output_size].
        """
        original_shape = x.shape
        if x.ndim == 2:
            batch_size = 1
            x = x.unsqueeze(0) # Add temporary batch dimension
        elif x.ndim == 3:
            batch_size = x.shape[0]
        else:
             raise ValueError(f"Input tensor should have 2 or 3 dimensions (rows, cols) or (batch, rows, cols), got shape {original_shape}")

        # --- Convert board to channel representation ---
        # Input format expected by Conv2d: [batch_size, channels, height, width]
        x = board_to_channels(x) # Shape: [batch_size, num_input_channels, ROWS, COLS]

        # --- Convolutional Layers ---
        for layer in self.conv_layers:
            x = layer(x)
        # Shape: [batch_size, num_filters2, ROWS, COLS]

        # reshape to one dimension per batch
        x = x.view(batch_size, -1)

        # --- Fully Connected Layers ---
        for layer in self.fc_layers:
            x = layer(x)

        # --- Output Handling ---
        # If the original input was a single instance, remove the batch dimension
        if len(original_shape) == 2:
            x = x.squeeze(0) # [1, output_size] -> [output_size]

        return x


class Connect4CNN_MLP_Value(nn.Module):
    """
    CNN model with a deep MLP classifier afterwards. With BatchNorm throughout. Fully connected second value head.
    """
    def __init__(self, num_filters1=32, num_filters2=64, hidden_fc_size=128, num_hidden_layers=1, value_fc_size=64, output_size=OUTPUT_SIZE):
        """
        Initializes the layers of the CNN.
        """
        super().__init__()
        self.num_input_channels = 3     # my pieces, their pieces, empty cells

        # Convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(self.num_input_channels, num_filters1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters1),
            nn.ReLU(),
            nn.Conv2d(num_filters1, num_filters2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters2),
            nn.ReLU(),
        ])

        # Calculate the flattened size after conv layers
        # Output shape after conv2: [batch_size, num_filters2, ROWS, COLS]
        self.flattened_size = num_filters2 * ROWS * COLS

        fc_layers = [
            nn.Linear(self.flattened_size, hidden_fc_size, bias=False),
            nn.BatchNorm1d(hidden_fc_size),
            nn.ReLU()
        ]
        
        for _ in range(num_hidden_layers):
            fc_layers += [
                nn.Linear(hidden_fc_size, hidden_fc_size),
                nn.BatchNorm1d(hidden_fc_size),
                nn.ReLU()
            ]

        fc_layers.append(nn.Linear(hidden_fc_size, output_size)) # Output raw logits
        self.fc_layers = nn.ModuleList(fc_layers)
        
        # second head for estimating the value of the current board state
        self.value_head = nn.ModuleList([
            nn.Linear(self.flattened_size, value_fc_size, bias=False),
            nn.BatchNorm1d(value_fc_size),
            nn.ReLU(),
            ##
            nn.Linear(value_fc_size, value_fc_size, bias=False),
            nn.BatchNorm1d(value_fc_size),
            nn.ReLU(),
            ##
            nn.Linear(value_fc_size, 1),
            nn.Tanh()
        ])


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Defines the forward pass of the model.
        Returns:
            torch.Tensor: Output logits. Shape: [batch_size, output_size] or [output_size].
        """
        original_shape = x.shape
        if x.ndim == 2:
            batch_size = 1
            x = x.unsqueeze(0) # Add temporary batch dimension
        elif x.ndim == 3:
            batch_size = x.shape[0]
        else:
             raise ValueError(f"Input tensor should have 2 or 3 dimensions (rows, cols) or (batch, rows, cols), got shape {original_shape}")

        # --- Convert board to channel representation ---
        # Input format expected by Conv2d: [batch_size, channels, height, width]
        x = board_to_channels(x)    # Shape: [batch_size, num_input_channels, ROWS, COLS]

        # --- Convolutional Layers ---
        for layer in self.conv_layers:
            x = layer(x)
        # Shape: [batch_size, num_filters2, ROWS, COLS]
        
        # reshape to one dimension per batch
        x = x.view(batch_size, -1)
        y = x               # second reference for value head branching off here

        # --- Fully Connected Layers ---
        for layer in self.fc_layers:
            x = layer(x)

        # --- Value head  ---
        for layer in self.value_head:
            y = layer(y)
        y = y.squeeze(-1)           # remove singleton dimension -> (batch_size,)

        # --- Output Handling ---
        # If the original input was a single instance, remove the batch dimension
        if len(original_shape) == 2:
            x = x.squeeze(0)    # [1, output_size] -> [output_size]
            y = y.squeeze(0)    # [1,] -> []

        return x, y


class Connect4CNN_MLP_Value_v2(nn.Module):
    """
    CNN model with a deep MLP classifier afterwards. Fully connected second value head.
    
    Apparently BatchNorm interacts badly with on-policy training due to eval/train differences. Trying GroupNorm/LayerNorm instead.
    """
    def __init__(self, num_filters1=32, num_filters2=64, hidden_fc_size=128, num_hidden_layers=1, value_fc_size=64, output_size=OUTPUT_SIZE):
        """
        Initializes the layers of the CNN.
        """
        super().__init__()
        self.num_input_channels = 3     # my pieces, their pieces, empty cells

        # Convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(self.num_input_channels, num_filters1, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(4, num_filters1),
            nn.ReLU(),
            nn.Conv2d(num_filters1, num_filters2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, num_filters2),
            nn.ReLU(),
        ])

        # Calculate the flattened size after conv layers
        # Output shape after conv2: [batch_size, num_filters2, ROWS, COLS]
        self.flattened_size = num_filters2 * ROWS * COLS

        fc_layers = [
            nn.Linear(self.flattened_size, hidden_fc_size),
            nn.LayerNorm((hidden_fc_size,)),
            nn.ReLU()
        ]
        
        for _ in range(num_hidden_layers):
            fc_layers += [
                nn.Linear(hidden_fc_size, hidden_fc_size),
                nn.LayerNorm((hidden_fc_size,)),
                nn.ReLU()
            ]

        fc_layers.append(nn.Linear(hidden_fc_size, output_size)) # Output raw logits
        self.fc_layers = nn.ModuleList(fc_layers)
        
        # second head for estimating the value of the current board state
        self.value_head = nn.ModuleList([
            nn.Linear(self.flattened_size, value_fc_size),
            nn.ReLU(),
            ##
            nn.Linear(value_fc_size, value_fc_size),
            nn.ReLU(),
            ##
            nn.Linear(value_fc_size, 1),
            nn.Tanh()
        ])


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Defines the forward pass of the model.
        Returns:
            torch.Tensor: Output logits. Shape: [batch_size, output_size] or [output_size].
        """
        original_shape = x.shape
        if x.ndim == 2:
            batch_size = 1
            x = x.unsqueeze(0) # Add temporary batch dimension
        elif x.ndim == 3:
            batch_size = x.shape[0]
        else:
             raise ValueError(f"Input tensor should have 2 or 3 dimensions (rows, cols) or (batch, rows, cols), got shape {original_shape}")

        # --- Convert board to channel representation ---
        # Input format expected by Conv2d: [batch_size, channels, height, width]
        x = board_to_channels(x)    # Shape: [batch_size, num_input_channels, ROWS, COLS]

        # --- Convolutional Layers ---
        for layer in self.conv_layers:
            x = layer(x)
        # Shape: [batch_size, num_filters2, ROWS, COLS]
        
        # reshape to one dimension per batch
        x = x.view(batch_size, -1)
        y = x               # second reference for value head branching off here

        # --- Fully Connected Layers ---
        for layer in self.fc_layers:
            x = layer(x)

        # --- Value head  ---
        for layer in self.value_head:
            y = layer(y)
        y = y.squeeze(-1)           # remove singleton dimension -> (batch_size,)

        # --- Output Handling ---
        # If the original input was a single instance, remove the batch dimension
        if len(original_shape) == 2:
            x = x.squeeze(0)    # [1, output_size] -> [output_size]
            y = y.squeeze(0)    # [1,] -> []

        return x, y


class Connect4CNN_MLP_Value_v3(nn.Module):
    """
    Increase CNN kernel size to 4, reduce layers, attach value head directly to main MLP network.
    """
    def __init__(self, num_filters1=32, num_filters2=64, hidden_fc_size=128, num_hidden_layers=3):
        """
        Initializes the layers of the CNN.
        """
        super().__init__()
        self.num_input_channels = 3     # my pieces, their pieces, empty cells

        # Convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(self.num_input_channels, num_filters2, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
        ])

        # Calculate the flattened size after conv layers
        # Output shape after conv2: [batch_size, num_filters2, ROWS, COLS]
        self.flattened_size = num_filters2 * (ROWS + 1) * (COLS + 1)

        fc_layers = [
            nn.Linear(self.flattened_size, hidden_fc_size),
            nn.LayerNorm((hidden_fc_size,)),
            nn.ReLU()
        ]
        
        for _ in range(num_hidden_layers):
            fc_layers += [
                nn.Linear(hidden_fc_size, hidden_fc_size),
                nn.LayerNorm((hidden_fc_size,)),
                nn.ReLU()
            ]

        output_size = 7 + 1         # columns + value
        fc_layers.append(nn.Linear(hidden_fc_size, output_size)) # Output raw logits
        self.fc_layers = nn.ModuleList(fc_layers)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Defines the forward pass of the model.
        Returns:
            torch.Tensor: Output logits. Shape: [batch_size, output_size] or [output_size].
        """
        original_shape = x.shape
        if x.ndim == 2:
            batch_size = 1
            x = x.unsqueeze(0) # Add temporary batch dimension
        elif x.ndim == 3:
            batch_size = x.shape[0]
        else:
             raise ValueError(f"Input tensor should have 2 or 3 dimensions (rows, cols) or (batch, rows, cols), got shape {original_shape}")

        # --- Convert board to channel representation ---
        # Input format expected by Conv2d: [batch_size, channels, height, width]
        x = board_to_channels(x)    # Shape: [batch_size, num_input_channels, ROWS, COLS]

        # --- Convolutional Layers ---
        for layer in self.conv_layers:
            x = layer(x)
        # Shape: [batch_size, num_filters2, ROWS+1, COLS+1]
        
        # reshape to one dimension per batch
        x = x.view(batch_size, -1)

        # --- Fully Connected Layers ---
        for layer in self.fc_layers:
            x = layer(x)

        # --- Output Handling ---
        # If the original input was a single instance, remove the batch dimension
        if len(original_shape) == 2:
            x = x.squeeze(0)    # [1, output_size] -> [output_size]

        return x[..., :-1], torch.tanh(x[..., -1])


class Connect4CNN_Mk4(nn.Module):
    """CNN/ResNet model with global average pooling and MLP classifier head."""
    def __init__(self, value_head=False):
        super().__init__()
        num_input_channels = 3     # my pieces, their pieces, empty cells

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_input_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.ReLU(),
        )

        def make_resblock():
            return nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(16, 64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(16, 64),
            )
        
        self.resblock1 = make_resblock()
        self.resblock2 = make_resblock()
        self.resblock3 = make_resblock()

        self.downsample = nn.Conv2d(64, 64, kernel_size=(6,1), stride=1, padding=0, bias=False)

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 7),      # output raw logits
        )
        if value_head:
            self.value_head = nn.Sequential(
                nn.Linear(64 * 7, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Tanh()
            )
        else:
            self.value_head = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if x.ndim == 2:
            x = x.unsqueeze(0)      # Add temporary batch dimension

        # Convert board to channel representation (3 channels)
        x = board_to_channels(x)       # [batch_size, 3, ROWS, COLS]
        x = self.feature_extractor(x)  # [batch_size, 64, ROWS, COLS]

        # Apply residual CNN blocks
        x = F.relu(self.resblock1(x) + x)
        x = F.relu(self.resblock2(x) + x)
        x = F.relu(self.resblock3(x) + x)

        #x = torch.mean(x, dim=(2,)).view(-1, 7 * 64)  # Columnwise average pooling: [batch_size, 7 * 64]
        x = self.downsample(x).view(-1, 7 * 64)  # Columnwise average pooling: [batch_size, 7 * 64]

        p = self.fc_layers(x)          # Fully connected layers (MLP head, outputs logits): [batch_size, 7]
        if len(original_shape) == 2:
            p = p.squeeze(0)

        if self.value_head is not None:
            v = self.value_head(x).squeeze(-1)  # Value head: [batch_size, 1] -> [batch_size]
            if len(original_shape) == 2:
                v = v.squeeze(0)

        if self.value_head is not None:
            return p, v
        else:
            return p


class Connect4CNN_Mk5(nn.Module):
    """CNN/ResNet model inspired by AlphaZero architecture."""
    def __init__(self, value_head=False):
        super().__init__()
        num_input_channels = 3     # my pieces, their pieces, empty cells
        FILTERS = 32

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_input_channels, FILTERS, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, FILTERS),
            nn.ReLU(),
        )

        def make_resblock():
            return nn.Sequential(
                nn.Conv2d(FILTERS, FILTERS, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(16, FILTERS),
                nn.ReLU(),
                nn.Conv2d(FILTERS, FILTERS, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(16, FILTERS),
            )
        
        self.resblock1 = make_resblock()
        self.resblock2 = make_resblock()
        self.resblock3 = make_resblock()
        
        self.scaledown = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FILTERS * 6 * 7, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Fully connected layers
        self.policy_head = nn.Sequential(
            nn.Linear(128, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 7),      # output raw logits
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.LayerNorm(64),
            nn.ReLU(),

            nn.Linear(64, 64, bias=False),
            nn.LayerNorm(64),
            nn.ReLU(),

            nn.Linear(64, 1),      # output raw logits
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        if x.ndim == 2:
            x = x.unsqueeze(0)      # Add temporary batch dimension

        # Convert board to channel representation (3 channels)
        x = board_to_channels(x)       # [batch_size, 3, ROWS, COLS]
        x = self.feature_extractor(x)  # [batch_size, 64, ROWS, COLS]

        # Apply residual CNN blocks
        x = F.relu(self.resblock1(x) + x)
        x = F.relu(self.resblock2(x) + x)
        x = F.relu(self.resblock3(x) + x)
        x = self.scaledown(x)          # [batch_size, 128]

        p = self.policy_head(x)              # Fully connected layers (MLP head, outputs logits): [batch_size, 7]
        v = self.value_head(x).squeeze(-1)   # Value head: [batch_size, 1] -> [batch_size]
        if len(original_shape) == 2:
            p = p.squeeze(0)
            v = v.squeeze(0)
        return p, v


def logits_and_value_from_rollout(model: nn.Module, board: torch.Tensor, width: int = 4, depth: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Use rollouts to find the best move. Return logits and move value."""
    values = estimate_move_values_from_rollout(board, model, width, depth)
    best_move = torch.argmax(values)

    logits = torch.full((COLS,), -1e12, device=board.device, dtype=torch.float32)
    logits[best_move] = 1.0
    return logits, values[best_move]


class RolloutModel(nn.Module):
    """
    A model wrapper that uses rollout value estimates to select moves.
    For each valid move, simulates rollouts using the given model, then picks the move with the highest estimated value.
    """
    def __init__(self, base_model: nn.Module, width: int = 4, depth: int = 5):
        super().__init__()
        self.base_model = base_model
        self.width = width
        self.depth = depth

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a board state, returns logits with 1.0 at the best move (by rollout value), -inf elsewhere.
        """
        if x.ndim == 2:
            return logits_and_value_from_rollout(self.base_model, x, self.width, self.depth)
        else:
            values = multiple_rollouts(x, self.base_model, self.width, self.depth)  # (batch_size, COLS)
            best_move = torch.argmax(values, dim=1)
            logits = torch.full((x.shape[0], x.shape[2]), -1e12, device=x.device, dtype=torch.float32)
            logits[torch.arange(x.shape[0]), best_move] = 1.0

            return logits, values[torch.arange(x.shape[0]), best_move]

# =============================================================================================== #

def load_frozen_model(name):
    if ':' in name:
        name, f_override = name.split(':')
    else:
        f_override = None
    
    def restore_model(model, fname):
        if f_override:
            fname = f_override
        if fname != "None":
            state = torch.load(fname)
            model.load_state_dict(state['model_state_dict'])
        return model

    if name == 'First':
        model = Connect4MLP(hidden_size=64, num_layers=3, dropout_rate=0.0)
        return restore_model(model, "ok_model.pth")
    elif name == "SimpleMLP":
        model = SimpleMLPModel()
        return restore_model(model, "last_cp.pth")
    elif name == 'CNN-MLP':
        model = Connect4CNN_MLP(num_filters1=32, num_filters2=64, hidden_fc_size=128, num_hidden_layers=3)
        return restore_model(model, 'model-cnnmlp.pth')
    elif name == 'CNN-MLP-v2':
        model = Connect4CNN_MLP(num_filters1=32, num_filters2=64, hidden_fc_size=128, num_hidden_layers=3)
        return restore_model(model, 'model-cnnmlp-v2.pth')
    elif name == 'CNN-MLP-v3':
        model = Connect4CNN_MLP_v2(num_filters1=32, num_filters2=64, hidden_fc_size=128, num_hidden_layers=3)  # batchnorm everywhere (bad idea!!)
        return restore_model(model, 'model-cnnmlp-v3.pth')
    elif name == 'CMV-v2':      # CNN-MLP with Value head - first really working version (no batchnorm!)
        model = Connect4CNN_MLP_Value_v2(num_filters1=32, num_filters2=64, hidden_fc_size=128, num_hidden_layers=3, value_fc_size=64)
        return restore_model(model, 'model-cmv-v2.pth')
    elif name == 'CNN-v2':
        model = Connect4CNN(num_filters1=32, num_filters2=64, hidden_fc_size=128)
        return restore_model(model, "model-cnn2.pth")
    elif name == 'A2C-v2':
        model = Connect4CNN_MLP_Value_v2(num_filters1=32, num_filters2=64, hidden_fc_size=128, num_hidden_layers=3, value_fc_size=64)
        return restore_model(model, 'model-a2c-v2.pth')
    elif name == 'CNN-Mk4':
        model = Connect4CNN_Mk4(value_head=True)
        return restore_model(model, 'model-cnn-mk4.pth')
    elif name == 'CNN-Mk5':
        model = Connect4CNN_Mk5()
        return restore_model(model, 'model-cnn-mk5.pth')
    elif name == 'cur':
        model = Connect4CNN_Mk4(value_head=True)
        return restore_model(model, 'best_model.pth')
    elif name == 'RandPunish':
        return RandomPunisher()
    elif name == 'Pretrained':
        model = Connect4CNN_MLP(num_filters1=32, num_filters2=64, hidden_fc_size=128, num_hidden_layers=3)
        return restore_model(model, 'pretrained.pth')
    else:
        raise ValueError('unknown model name: ' + name)
