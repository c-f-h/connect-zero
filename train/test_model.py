import torch
import unittest
import numpy as np

from train.model import (
    board_to_channels,
    find_best_move,
    ROWS,
    COLS
)
# Need to import from board.py for string_to_board helper and make_move_and_check (used by find_best_move)
import train.board

class TestModel(unittest.TestCase):
    def setUp(self):
        self.rows = ROWS
        self.cols = COLS
        self.empty_board = torch.zeros((self.rows, self.cols), dtype=torch.int8)

    def test_board_to_channels_empty_board(self):
        channels = board_to_channels(self.empty_board)
        # Expected shape: [1, 3, ROWS, COLS] for single board, or [3, ROWS, COLS] if not auto-unsqueezed by tested func
        # board_to_channels adds a batch dim if ndim == 2
        self.assertEqual(channels.shape, (1, 3, self.rows, self.cols))

        # Channel 0 (Player 1 pieces) should be all zeros
        self.assertTrue(torch.all(channels[0, 0, :, :] == 0))
        # Channel 1 (Player 2 pieces) should be all zeros
        self.assertTrue(torch.all(channels[0, 1, :, :] == 0))
        # Channel 2 (Empty cells) should be all ones
        self.assertTrue(torch.all(channels[0, 2, :, :] == 1))

    def test_board_to_channels_mixed_board(self):
        board = self.empty_board.clone()
        board[0, 0] = 1  # Player 1 piece
        board[1, 1] = -1 # Player 2 piece
        board[2, 2] = 1  # Player 1 piece

        channels = board_to_channels(board)
        self.assertEqual(channels.shape, (1, 3, self.rows, self.cols))

        # Channel 0 (P1 pieces)
        expected_p1_channel = torch.zeros((self.rows, self.cols), dtype=torch.float32)
        expected_p1_channel[0, 0] = 1
        expected_p1_channel[2, 2] = 1
        self.assertTrue(torch.equal(channels[0, 0, :, :], expected_p1_channel))

        # Channel 1 (P2 pieces)
        expected_p2_channel = torch.zeros((self.rows, self.cols), dtype=torch.float32)
        expected_p2_channel[1, 1] = 1
        self.assertTrue(torch.equal(channels[0, 1, :, :], expected_p2_channel))

        # Channel 2 (Empty cells)
        # All cells are empty except (0,0), (1,1), (2,2)
        expected_empty_channel = torch.ones((self.rows, self.cols), dtype=torch.float32)
        expected_empty_channel[0, 0] = 0
        expected_empty_channel[1, 1] = 0
        expected_empty_channel[2, 2] = 0
        self.assertTrue(torch.equal(channels[0, 2, :, :], expected_empty_channel))

    def test_board_to_channels_batch_input(self):
        board1 = self.empty_board.clone()
        board1[0,0] = 1

        board2 = self.empty_board.clone()
        board2[1,1] = -1

        batch = torch.stack([board1, board2]) # Shape: [2, ROWS, COLS]
        channels = board_to_channels(batch)
        self.assertEqual(channels.shape, (2, 3, self.rows, self.cols))

        # Check board1 channels (index 0 in batch)
        self.assertEqual(channels[0, 0, 0, 0], 1) # P1 piece
        self.assertEqual(channels[0, 1, 0, 0], 0) # Not P2 piece
        self.assertEqual(channels[0, 2, 0, 0], 0) # Not empty

        # Check board2 channels (index 1 in batch)
        self.assertEqual(channels[1, 0, 1, 1], 0) # Not P1 piece
        self.assertEqual(channels[1, 1, 1, 1], 1) # P2 piece
        self.assertEqual(channels[1, 2, 1, 1], 0) # Not empty
        self.assertEqual(channels[1, 2, 0, 0], 1) # An empty cell on board2

    def test_find_best_move_immediate_win(self):
        # Board where player 1 can win by playing in column 3
        # . . . . . . .
        # . . . . . . .
        # . . . . . . .
        # . . . . . . .
        # X X X . . . . (P1 has 3 in a row, col 3 is winning move)
        # O O O X . . .
        board = self.empty_board.clone()
        board[self.rows - 2, 0] = 1
        board[self.rows - 2, 1] = 1
        board[self.rows - 2, 2] = 1
        # Opponent pieces to make it realistic
        board[self.rows - 1, 0] = -1
        board[self.rows - 1, 1] = -1
        board[self.rows - 1, 2] = -1
        board[self.rows - 1, 3] = 1 # A P1 piece under the winning spot

        logits = find_best_move(board)
        self.assertEqual(logits.shape, (self.cols,))
        winning_move_col = 3

        # Expected: Logits should be high for col 3, very low for others.
        # find_best_move returns log(one_hot + eps)
        # log(0 + 1e-12) for non-chosen columns, log(1 + 1e-12) for chosen column
        expected_logits = torch.full((self.cols,), np.log(1e-12), dtype=torch.float32)
        expected_logits[winning_move_col] = np.log(1.0 + 1e-12)
        # Due to the +1e-12, direct comparison of -inf might be tricky.
        # Let's check that the winning move has a significantly higher logit.
        self.assertTrue(torch.allclose(logits, expected_logits, atol=1e-5)) # Check if close to one-hot log prob

    def test_find_best_move_block_opponent_win(self):
        # P1 has no win. P2 threatens to win at (5,3) (col 3)
        board_to_block = self.empty_board.clone()
        board_to_block[5,0] = -1 # O
        board_to_block[5,1] = -1 # O
        board_to_block[5,2] = -1 # O
        # P1 must play in col 3 (idx 3) to block. Piece lands at (5,3).

        # Add some P1 pieces that don't make an immediate win for P1
        board_to_block[4,0] = 1 # X
        board_to_block[3,0] = 1 # X
        # board_to_block[2,0] = 1 # X (This would be P1 win if [4,0] and [3,0] are clear for P1 to play)
        # To ensure P1 cannot win by playing col 0:
        board_to_block[4,1] = -1 # O (P2 piece in col 1)
        board_to_block[3,1] = 1  # X (P1 piece in col 1, to break P2's potential vertical win in col 1)

        logits = find_best_move(board_to_block)
        blocking_move_col = 3
        expected_logits = torch.full((self.cols,), np.log(1e-12), dtype=torch.float32)
        expected_logits[blocking_move_col] = np.log(1.0 + 1e-12)
        # print("DEBUG: test_find_best_move_block_opponent_win")
        # print("Board state:\n", board_to_block)
        # print("Returned logits:\n", logits)
        # print("Expected logits:\n", expected_logits)
        self.assertTrue(torch.allclose(logits, expected_logits, atol=1e-5))

    def test_find_best_move_no_immediate_action(self):
        # A board where no immediate win or block is obvious
        board = train.board.string_to_board_test_format( # Using the corrected string_to_board from board.py tests
            "       " # Row 0 (empty)
            "       " # Row 1
            "       " # Row 2
            "   X   " # Row 3, P1 in middle
            "   O   " # Row 4, P2 in middle
            "  XOX  " # Row 5, X O X
        )
        logits = find_best_move(board)
        # Expect zeros as per function spec when no win/block
        self.assertTrue(torch.all(logits == 0))

    def test_find_best_move_respects_full_column(self):
        # Board where col 0 is full. P1 has a winning move in col 1.
        # O . . . (col 0 full)
        # X . . .
        # O . . .
        # X . . .
        # O X X . (P1 to win in col 3, which is idx 3)
        # X O O .
        board = self.empty_board.clone()
        for r in range(self.rows): # Fill column 0
            board[r, 0] = 1 if r % 2 == 0 else -1 # Alternating X O

        # Player 1 can win in col 2 (idx 2)
        board[self.rows - 1, 1] = 1 # X
        board[self.rows - 1, 2] = 1 # X
        # Winning move for P1 is col 3 (idx=3)
        # board[self.rows - 1, 3] should be the winning move

        # Let's simplify: Col 0 is full. P1 can win at (5,3)
        # X . . .
        # O . . .
        # X . . .
        # O . . .
        # X X X _ <- P1 win here (5,3)
        # O O O X
        board_full_col = self.empty_board.clone()
        for r in range(self.rows): board_full_col[r,0] = 1 # Fill col 0 with P1 pieces

        board_full_col[self.rows-1, 1] = 1 # X
        board_full_col[self.rows-1, 2] = 1 # X
        # Col 3 is the winning move for P1. Col 0 is full and should not be chosen.

        # Make sure col 0 is not considered a winning move even if it hypothetically led to one.
        # (This test is more about `make_move_and_check` used by `find_best_move` not erroring)
        # `find_best_move` iterates `for c in range(cols)`, then `if B[0, c] == 0:`
        # This `if B[0,c] == 0` check correctly prevents trying moves in full columns.
        # So, if a win exists elsewhere, it should be found.

        logits = find_best_move(board_full_col)
        winning_move_col = 3
        expected_logits = torch.full((self.cols,), np.log(1e-12), dtype=torch.float32)
        expected_logits[winning_move_col] = np.log(1.0 + 1e-12)
        self.assertTrue(torch.allclose(logits, expected_logits, atol=1e-5))

    def test_find_best_move_prefers_win_over_block(self):
        # Board where P1 can win, AND P2 is threatening a win. P1 should take its own win.
        # . . . . . . .
        # . . . . . . .
        # X X X . . . . (P1 to win in col 3)
        # O O . . . . . (P2 threatening in col 2)
        # X X O . . . .
        # O O X . . . .
        board = self.empty_board.clone()
        # P1 (X) can win at (5,3) by playing col 3.
        board[5, 0] = 1 # X
        board[5, 1] = 1 # X
        board[5, 2] = 1 # X
        # p1_winning_col = 3, piece lands at (5,3)

        # P2 (O) threatens at (4,3) by playing col 3.
        board[4, 0] = -1 # O
        board[4, 1] = -1 # O
        board[4, 2] = -1 # O
        # p2_threatening_col = 3, piece would land at (4,3)

        # Add some other pieces to ensure correct lowest row placements
        board[3,0] = 1; board[3,1] = -1; board[3,2] = 1 # Fill spots below P2 threat

        logits = find_best_move(board)
        p1_winning_col = 3 # P1 win at (5,3)
        expected_logits = torch.full((self.cols,), np.log(1e-12), dtype=torch.float32)
        expected_logits[p1_winning_col] = np.log(1.0 + 1e-12)
        self.assertTrue(torch.allclose(logits, expected_logits, atol=1e-5))


if __name__ == '__main__':
    unittest.main()
