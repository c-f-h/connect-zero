import torch
import torch.nn as nn
import unittest
import numpy as np

from train.treesearch import multiple_rollouts, estimate_move_values_from_rollout
from train.board import string_to_board, make_move_and_check_batch # For test setup
from train.globals import ROWS, COLS # Import ROWS, COLS from globals
from train.model import board_to_channels # Potentially for mock model if it uses channels

# Define a REWARD_DISCOUNT for testing, as it's imported from main in treesearch.py
# We need to make sure this is accessible or patched for the treesearch module.
# For simplicity in testing, we might need to mock the import if it's complex,
# or ensure treesearch.py can run using a globally defined one for tests.
# Let's try defining it here, and if treesearch.py picks it up, great.
# If not, we'll need to use unittest.mock.patch.
REWARD_DISCOUNT_FOR_TEST = 0.99

import sys # Import sys for sys.modules access

# Mock for the 'main' module that treesearch.py tries to import REWARD_DISCOUNT from
mock_main_module = unittest.mock.MagicMock()
mock_main_module.REWARD_DISCOUNT = REWARD_DISCOUNT_FOR_TEST


class MockConnect4Model(nn.Module):
    def __init__(self, default_logits=None, default_value=0.0, specific_responses=None):
        """
        Mock model for testing.
        specific_responses: dict mapping board_tuple -> (logits, value)
        """
        super().__init__()
        self.cols = COLS
        if default_logits is None:
            self.default_logits = torch.zeros(self.cols) # Uniform policy
        else:
            self.default_logits = default_logits
        self.default_value = default_value
        self.specific_responses = specific_responses if specific_responses is not None else {}
        self.call_history = [] # To inspect calls

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.call_history.append(boards.clone())
        is_batch = boards.ndim == 3
        if not is_batch: # Single board [R, C]
            boards = boards.unsqueeze(0)

        batch_size = boards.shape[0]
        batch_logits = []
        batch_values = []

        for i in range(batch_size):
            board = boards[i]
            board_tuple = tuple(board.view(-1).tolist()) # Convert tensor to hashable tuple for dict key

            if board_tuple in self.specific_responses:
                logits, value = self.specific_responses[board_tuple]
            else:
                logits, value = self.default_logits, self.default_value

            # Ensure logits are valid (not all -inf for available moves)
            # This mock needs to be simple; complex policy logic is not its job.
            # The `sample_moves` function in play.py handles illegal moves by adding -inf.
            # So, this mock should return raw logits for all columns.

            batch_logits.append(logits.clone() if isinstance(logits, torch.Tensor) else torch.tensor(logits, dtype=torch.float32))
            batch_values.append(torch.tensor(value, dtype=torch.float32).squeeze()) # Ensure value is scalar-like

        final_logits = torch.stack(batch_logits)
        final_values = torch.stack(batch_values)

        if not is_batch:
            return final_logits.squeeze(0), final_values.squeeze(0)
        return final_logits, final_values

class TestTreeSearch(unittest.TestCase):
    def setUp(self):
        self.rows = ROWS
        self.cols = COLS
        self.empty_board = torch.zeros((self.rows, self.cols), dtype=torch.int8)
        self.mock_main_patcher = unittest.mock.patch.dict(sys.modules, {'main': mock_main_module})

    def test_estimate_move_values_from_rollout_basic(self):
        # This is a simple wrapper, so just check it calls multiple_rollouts and returns correct shape
        mock_model = MockConnect4Model()
        board = self.empty_board.clone()

        self.mock_main_patcher.start() # Apply the mock for 'main' module
        values = estimate_move_values_from_rollout(board, mock_model, width=2, depth=1)
        self.mock_main_patcher.stop() # Stop the mock

        self.assertEqual(values.shape, (self.cols,))
        # self.assertEqual(len(mock_model.call_history), 1) # This was the incorrect assertion causing failure
                                                          # Actually, model is called by multiple_rollouts if depth is reached
                                                          # and by sample_moves within the rollout.
                                                          # For depth=1, sample_moves is called once.
                                                          # If any games are unfinished after depth=1, model is called for value head.
        # A simple board, depth 1, all moves are valid.
        # Initial moves (7) * width (2) = 14 rollouts. Each rollout is 1 ply.
        # sample_moves makes 1 call with batch_size = 14.
        # If all games end (win/draw), value head is not called.
        # If some are unfinished, value head is called for them.
        # For an empty board, no move leads to an immediate win/loss.
        # So, all 14 games will be unfinished. Mock model will be called once for policy, once for value.
        self.assertEqual(len(mock_model.call_history), 2) # This is the correct assertion

    def test_multiple_rollouts_immediate_win(self):
        # Board: P1 has X X X . . . . in the bottom row. Playing col 3 wins.
        board = self.empty_board.clone()
        board[self.rows - 1, 0] = 1 # X
        board[self.rows - 1, 1] = 1 # X
        board[self.rows - 1, 2] = 1 # X
        # Col 3 is the winning move. Other cols are just normal moves.

        mock_model = MockConnect4Model() # Default behavior for other moves

        self.mock_main_patcher.start()
        estimated_values = multiple_rollouts(board.unsqueeze(0), mock_model, width=2, depth=3)
        self.mock_main_patcher.stop()

        self.assertEqual(estimated_values.shape, (1, self.cols))
        self.assertAlmostEqual(estimated_values[0, 3].item(), 1.0, places=5)

        # Check that other valid moves (0,1,2 are taken by P1, 3 wins)
        # Cols 4, 5, 6 are open.
        # For these moves, rollouts occur. Since mock_model returns value 0 and 0 policy,
        # and depth is >0, the values will likely be 0 or negative if opponent wins.
        # For this test, primarily concerned with the winning move's value.
        # Example: if col 4 is played, board becomes P1 at (5,4). Then P2 (mock) plays.
        # If P2 wins, value is -REWARD_DISCOUNT. If game continues to depth, value head (0) is used.
        # So, other columns should not be 1.0.
        for c in [0,1,2,4,5,6]:
            if board[0,c] == 0: # If column is not full (already tested by board setup)
                                # Actually, (5,0),(5,1),(5,2) are full for the next piece.
                                # The check `valid_moves = (initial_boards[:, 0] == 0)` in multiple_rollouts
                                # refers to the *top* row. All top rows are 0 here.
                                # The `make_move_and_check_batch` handles placing pieces.
                                # Cols 0,1,2 will be full *after* P1's first piece if P1 chose them.
                                # But `multiple_rollouts` evaluates moves *from the initial_board*.
                                # For the initial_board, only (5,0),(5,1),(5,2) are occupied.
                                # So moves in col 0,1,2 will land in row 4.
                if c !=3: # If not the winning column
                     self.assertNotAlmostEqual(estimated_values[0, c].item(), 1.0, places=5, msg=f"Col {c} should not have value 1.0")
                if c in [0,1,2]: # These columns are already occupied in the winning line, P1 can't play there again at row 5.
                                 # A move in col 0 would go to row 4.
                    pass # Value can be something else, just not 1.0

    def test_multiple_rollouts_immediate_loss(self):
        # P1 makes a move (e.g. col 0). Then P2 (mock model) makes a move that wins for P2.
        # The value of P1's move in col 0 should be -REWARD_DISCOUNT.
        initial_board_p1 = self.empty_board.clone()

        # Define the board state P2 will see AFTER P1 plays in col 0 (at (5,0))
        # P1's piece at (5,0) will be -1 from P2's perspective.
        # P2 will have pieces +1.

        # initial_board_p1 is P1's view before P1 makes a move.
        # P1 will play in col 4 (idx 4), piece at (5,4).
        # P2 has pieces O at (5,0), (5,1), (5,2).
        initial_board_p1[self.rows - 1, 0] = -1 # O
        initial_board_p1[self.rows - 1, 1] = -1 # O
        initial_board_p1[self.rows - 1, 2] = -1 # O

        # Board P2 sees after P1 plays col 4 (at (5,4)):
        # P2 is +1. P2's O pieces are +1. P1's X piece is -1.
        # State: (5,0)=1, (5,1)=1, (5,2)=1, (5,4)=-1
        board_key_for_mock = self.empty_board.clone()
        board_key_for_mock[self.rows - 1, 0] = 1  # P2's O piece
        board_key_for_mock[self.rows - 1, 1] = 1  # P2's O piece
        board_key_for_mock[self.rows - 1, 2] = 1  # P2's O piece
        board_key_for_mock[self.rows - 1, 4] = -1 # P1's X move from P2's view
        # P2 can win by playing in col 3 (piece at (5,3)).

        p2_winning_logits = torch.full((self.cols,), -10.0)
        p2_winning_logits[3] = 10.0 # Make P2 choose col 3 to win

        mock_model = MockConnect4Model(specific_responses={
            tuple(board_key_for_mock.view(-1).tolist()): (p2_winning_logits, 0.0)
        })

        self.mock_main_patcher.start()
        # P1 plays col 4 from initial_board_p1
        estimated_values = multiple_rollouts(initial_board_p1.unsqueeze(0), mock_model, width=1, depth=1)
        self.mock_main_patcher.stop()

        self.assertEqual(estimated_values.shape, (1, self.cols))
        # Check value of P1's move in col 4
        self.assertAlmostEqual(estimated_values[0, 4].item(), -REWARD_DISCOUNT_FOR_TEST, places=5,
                               msg=f"Calls: {len(mock_model.call_history)}")

    def test_multiple_rollouts_depth_value_head(self):
        test_depth = 1
        value_from_head = 0.5
        initial_board_p1 = self.empty_board.clone() # P1 to play, e.g. col 0

        # Board P2 sees after P1 plays col 0 (at (5,0)): P1 piece is -1 at (5,0)
        board_for_p2_policy = self.empty_board.clone()
        board_for_p2_policy[self.rows-1, 0] = -1

        # P2 plays col 0 (at (4,0) from P2's view, which is (4,0) on main board with P2 piece)
        # Mock P2 to play col 0.
        p2_ply0_logits = torch.full((self.cols,), -10.0)
        p2_ply0_logits[0] = 10.0

        # Board state for value head evaluation:
        # P1 played (5,0). P2 played (4,0).
        # This is the board given to model's value head. Perspective is P1's.
        # So, P1 piece at (5,0) is 1. P2 piece at (4,0) is -1.
        board_for_value_eval = self.empty_board.clone()
        board_for_value_eval[self.rows-1, 0] = 1  # P1's first move
        board_for_value_eval[self.rows-2, 0] = -1 # P2's first move (ply 0)

        mock_model = MockConnect4Model(
            specific_responses={
                tuple(board_for_p2_policy.view(-1).tolist()): (p2_ply0_logits, 0.0), # P2's policy
                # Logits for value eval state should be [COLS] shape, not board shape
                tuple(board_for_value_eval.view(-1).tolist()): (torch.zeros(self.cols), value_from_head)
            },
            default_logits=torch.zeros(self.cols) # Default policy for other states if any
        )

        self.mock_main_patcher.start()
        estimated_values = multiple_rollouts(initial_board_p1.unsqueeze(0), mock_model, width=1, depth=test_depth)
        self.mock_main_patcher.stop()

        # Expected value calculation:
        # factor = (REWARD_DISCOUNT**(depth + 1)) * (1.0 if (depth % 2 == 1) else -1.0)
        # depth = 1 (P1 makes move, P2 makes move (ply 0), then value head)
        # factor = (REWARD_DISCOUNT_FOR_TEST**(1 + 1)) * (1.0 if (1 % 2 == 1) else -1.0)
        # factor = REWARD_DISCOUNT_FOR_TEST**2 * 1.0
        expected_val = (REWARD_DISCOUNT_FOR_TEST**(test_depth + 1)) * value_from_head

        self.assertEqual(estimated_values.shape, (1, self.cols))
        self.assertAlmostEqual(estimated_values[0, 0].item(), expected_val, places=5)

    def test_multiple_rollouts_full_column(self):
        initial_board = self.empty_board.clone()
        full_column_idx = 0
        # Fill the column
        for r in range(self.rows):
            initial_board[r, full_column_idx] = 1 # Fill with P1's pieces for simplicity

        mock_model = MockConnect4Model()
        self.mock_main_patcher.start()
        estimated_values = multiple_rollouts(initial_board.unsqueeze(0), mock_model, width=1, depth=1)
        self.mock_main_patcher.stop()

        self.assertEqual(estimated_values.shape, (1, self.cols))
        # Compare with the float32 representation of -1e12
        self.assertEqual(estimated_values[0, full_column_idx].item(), torch.tensor(-1e12, dtype=torch.float32).item())

        # Check that other columns have different values (e.g., 0 if no win/loss/value head)
        # These will be 0 if depth is 1, mock value is 0, and REWARD_DISCOUNT**(1+1)*0 = 0
        for c in range(1, self.cols):
            self.assertNotEqual(estimated_values[0, c].item(), torch.tensor(-1e12, dtype=torch.float32).item())


if __name__ == '__main__':
    unittest.main()
