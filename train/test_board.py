import torch
import unittest

from train.board import (
    make_move,
    check_win_after_move,
    make_move_and_check,
    is_board_full_batch,
    check_win_batch_conv,
    make_move_and_check_batch,
    string_to_board, # Helper function for creating test boards
)

class TestBoard(unittest.TestCase):
    def setUp(self):
        # Standard 6x7 Connect Four board
        self.rows = 6
        self.cols = 7
        self.empty_board = torch.zeros((self.rows, self.cols), dtype=torch.int8)

    def test_make_move_valid(self):
        board = self.empty_board.clone()
        new_board, row = make_move(board, 3)
        self.assertEqual(row, self.rows - 1) # Piece should be in the last row
        self.assertEqual(new_board[row, 3], 1) # Piece of player 1
        self.assertNotEqual(id(board), id(new_board)) # Should be a new tensor

        # Make another move in the same column
        board = new_board
        new_board, row = make_move(board, 3)
        self.assertEqual(row, self.rows - 2) # Piece should be in the second to last row
        self.assertEqual(new_board[row, 3], 1)

    def test_make_move_invalid_column(self):
        with self.assertRaisesRegex(torch.jit.Error, "Invalid column index: 7"):
            make_move(self.empty_board, self.cols)
        with self.assertRaisesRegex(torch.jit.Error, "Invalid column index: -1"):
            make_move(self.empty_board, -1)

    def test_make_move_full_column(self):
        board = self.empty_board.clone()
        # Fill one column
        for i in range(self.rows):
            board, _ = make_move(board, 0)

        with self.assertRaisesRegex(torch.jit.Error, "Column 0 is full."):
            make_move(board, 0)

    def test_check_win_horizontal(self):
        # Test horizontal win
        board = self.empty_board.clone()
        # Player 1 places 4 in a row horizontally
        # ...X...
        # ...X...
        # ...X...
        # ...X...
        # .XXX...
        # XXXXXXX
        board[self.rows - 1, 0] = 1
        board[self.rows - 1, 1] = 1
        board[self.rows - 1, 2] = 1
        # Last move at (self.rows - 1, 3)
        self.assertFalse(check_win_after_move(board, self.rows - 1, 2)) # Not a win yet
        board[self.rows - 1, 3] = 1
        self.assertTrue(check_win_after_move(board, self.rows - 1, 3))
        # Test with opponent pieces interspersed (should not count)
        board[self.rows - 1, 1] = -1
        self.assertFalse(check_win_after_move(board, self.rows - 1, 3))

    def test_check_win_vertical(self):
        # Test vertical win
        board = self.empty_board.clone()
        # Player 1 places 4 in a row vertically in column 2
        # ...X...
        # ...X...
        # ..X....
        # ..X....
        # ..X....
        # ..X....
        board[self.rows - 1, 2] = 1
        board[self.rows - 2, 2] = 1
        board[self.rows - 3, 2] = 1
        # Last move at (self.rows - 4, 2)
        self.assertFalse(check_win_after_move(board, self.rows - 3, 2)) # Not a win yet
        board[self.rows - 4, 2] = 1
        self.assertTrue(check_win_after_move(board, self.rows - 4, 2))
        # Test with opponent pieces interspersed
        board[self.rows - 2, 2] = -1
        self.assertFalse(check_win_after_move(board, self.rows - 4, 2))

    def test_check_win_diagonal_positive(self):
        # Test positive slope diagonal win (\)
        # ......X (r-3, c+3)
        # .....X. (r-2, c+2)
        # ....X.. (r-1, c+1)
        # ...X... (r, c)
        board = self.empty_board.clone()
        # Bottom-left to top-right
        # Player 1 places 4 in a row diagonally
        # ...X...
        # ..X....
        # .X.....
        # X......
        r, c = self.rows - 1, 0 # Last piece (bottom-left)
        board[r, c] = 1
        board[r - 1, c + 1] = 1
        board[r - 2, c + 2] = 1
        self.assertFalse(check_win_after_move(board, r-2, c+2))
        board[r - 3, c + 3] = 1 # Winning move
        self.assertTrue(check_win_after_move(board, r - 3, c + 3))
        # Test with opponent piece
        board[r-1, c+1] = -1
        self.assertFalse(check_win_after_move(board, r-3, c+3))

        # Test another positive diagonal
        # Board:
        # .......
        # ......X (2,6)
        # .....X. (3,5)
        # ....X.. (4,4)
        # ...X... (5,3)
        board = self.empty_board.clone()
        r, c = self.rows -1, 3
        board[r,c] = 1 # (5,3)
        board[r-1, c+1] = 1 # (4,4)
        board[r-2, c+2] = 1 # (3,5)
        self.assertFalse(check_win_after_move(board, r-2, c+2))
        board[r-3, c+3] = 1 # (2,6) - Winning move
        self.assertTrue(check_win_after_move(board, r-3, c+3))


    def test_check_win_diagonal_negative(self):
        # Test negative slope diagonal win (/)
        # X...... (r-3, c-3)
        # .X..... (r-2, c-2)
        # ..X.... (r-1, c-1)
        # ...X... (r,c)
        board = self.empty_board.clone()
        # Top-left to bottom-right
        # Player 1 places 4 in a row diagonally
        # X......
        # .X.....
        # ..X....
        # ...X...
        r, c = self.rows - 1, 3 # Last piece (bottom-right most of the diagonal)
        board[r, c] = 1
        board[r - 1, c - 1] = 1
        board[r - 2, c - 2] = 1
        self.assertFalse(check_win_after_move(board, r-2, c-2))
        board[r - 3, c - 3] = 1 # Winning move
        self.assertTrue(check_win_after_move(board, r - 3, c - 3))
        # Test with opponent piece
        board[r-1, c-1] = -1
        self.assertFalse(check_win_after_move(board, r-3, c-3))

        # Test another negative diagonal
        # Board:
        # .......
        # X...... (2,0)
        # .X..... (3,1)
        # ..X.... (4,2)
        # ...X... (5,3)
        board = self.empty_board.clone()
        r,c = self.rows-1, 3
        board[r,c] = 1 # (5,3)
        board[r-1,c-1] = 1 # (4,2)
        board[r-2,c-2] = 1 # (3,1)
        self.assertFalse(check_win_after_move(board,r-2,c-2))
        board[r-3,c-3] = 1 # (2,0) - Winning move
        self.assertTrue(check_win_after_move(board,r-3,c-3))

    def test_check_win_no_win(self):
        board = self.empty_board.clone()
        board[self.rows - 1, 0] = 1
        board[self.rows - 1, 1] = 1
        board[self.rows - 1, 2] = 1
        # Not a win
        self.assertFalse(check_win_after_move(board, self.rows - 1, 2))
        # Almost vertical
        board[self.rows - 2, 0] = 1
        board[self.rows - 3, 0] = 1
        self.assertFalse(check_win_after_move(board, self.rows - 3, 0))

    def test_make_move_and_check_no_win(self):
        board = self.empty_board.clone()
        new_board, win = make_move_and_check(board, 0)
        self.assertFalse(win)
        self.assertEqual(new_board[self.rows - 1, 0], 1)
        self.assertNotEqual(id(board), id(new_board))

    def test_make_move_and_check_win(self):
        # Horizontal win
        board = self.empty_board.clone()
        board[self.rows - 1, 0] = 1
        board[self.rows - 1, 1] = 1
        board[self.rows - 1, 2] = 1
        # Winning move in column 3
        new_board, win = make_move_and_check(board, 3)
        self.assertTrue(win)
        self.assertEqual(new_board[self.rows - 1, 3], 1)

        # Vertical win
        board = self.empty_board.clone()
        board[self.rows - 1, 0] = 1
        board[self.rows - 2, 0] = 1
        board[self.rows - 3, 0] = 1
        # Winning move in column 0
        new_board, win = make_move_and_check(board, 0) # Piece at (rows-4, 0)
        self.assertTrue(win)
        self.assertEqual(new_board[self.rows - 4, 0], 1)


    def test_make_move_and_check_full_column(self):
        board = self.empty_board.clone()
        # Fill one column
        for i in range(self.rows):
            board, _ = make_move(board, 0)

        with self.assertRaisesRegex(torch.jit.Error, "Column 0 is full."):
            make_move_and_check(board, 0)

    def test_is_board_full_batch(self):
        # Test with empty boards
        empty_boards = torch.zeros((2, self.rows, self.cols), dtype=torch.int8)
        results = is_board_full_batch(empty_boards)
        self.assertFalse(results[0])
        self.assertFalse(results[1])
        self.assertEqual(results.dtype, torch.bool)

        # Test with partially filled boards
        partially_filled_board = self.empty_board.clone()
        partially_filled_board[self.rows -1, 0] = 1
        partially_filled_board[0, self.cols -1] = -1 # Fill a top cell

        boards = torch.stack([self.empty_board, partially_filled_board])
        results = is_board_full_batch(boards)
        self.assertFalse(results[0]) # empty_board
        self.assertFalse(results[1]) # partially_filled_board

        # Test with full boards
        # Full board with 1s
        full_board_1s = torch.ones((self.rows, self.cols), dtype=torch.int8)
        # Full board with mixed pieces
        full_board_mixed = torch.ones((self.rows, self.cols), dtype=torch.int8)
        full_board_mixed[0, ::2] = -1 # Player 2 pieces in first row, even columns
        full_board_mixed[1, 1::2] = -1 # Player 2 pieces in second row, odd columns

        # A board that is full except the top row (should not be 'full' by this function's logic)
        almost_full_board = torch.ones((self.rows, self.cols), dtype=torch.int8)
        almost_full_board[0,0] = 0

        full_boards_batch = torch.stack([full_board_1s, full_board_mixed, self.empty_board, almost_full_board])
        results = is_board_full_batch(full_boards_batch)
        self.assertTrue(results[0])
        self.assertTrue(results[1])
        self.assertFalse(results[2]) # The empty board
        self.assertFalse(results[3]) # The almost full board

        # Test with a single board in batch
        single_full_board = full_board_1s.unsqueeze(0)
        results = is_board_full_batch(single_full_board)
        self.assertTrue(results[0])
        self.assertEqual(results.shape, (1,))

        single_empty_board = self.empty_board.unsqueeze(0)
        results = is_board_full_batch(single_empty_board)
        self.assertFalse(results[0])
        self.assertEqual(results.shape, (1,))

        # Test with zero boards in batch
        zero_boards = torch.empty((0, self.rows, self.cols), dtype=torch.int8)
        results = is_board_full_batch(zero_boards)
        self.assertEqual(results.shape, (0,))

    def _reset_conv_kernel(self):
        # Access the global kernel from the board module and reset it
        import train.board
        train.board.g_win_conv_kernel = None

    def test_check_win_batch_conv(self):
        self._reset_conv_kernel()
        # Board 1: Horizontal win for player 1
        board1_win_h = self.empty_board.clone()
        board1_win_h[self.rows - 1, 0:4] = 1

        # Board 2: Vertical win for player 1
        board2_win_v = self.empty_board.clone()
        board2_win_v[self.rows - 4:, 2] = 1

        # Board 3: No win
        board3_no_win = self.empty_board.clone()
        board3_no_win[self.rows -1, 0] = 1
        board3_no_win[self.rows -1, 1] = 1
        board3_no_win[self.rows -1, 2] = -1 # Opponent piece
        board3_no_win[self.rows -1, 3] = 1

        # Board 4: Positive diagonal win for player 1
        board4_win_dpos = self.empty_board.clone()
        for i in range(4):
            board4_win_dpos[self.rows - 1 - i, i] = 1 # \ diagonal from bottom-left

        # Board 5: Negative diagonal win for player 1
        board5_win_dneg = self.empty_board.clone()
        for i in range(4):
            board5_win_dneg[self.rows - 1 - i, self.cols - 1 - i] = 1 # / diagonal from bottom-right

        # Board 6: Full board, no win (can happen in Connect Four, e.g. a tie)
        # For this test, just a board with no 4-in-a-row for player 1
        # string_to_board expects a 42-character string.
        # Each row below is 7 characters.
        board6_full_no_win_p1 = string_to_board(
            # This is a draw board. ' ' are empty (0), 'X' is 1, 'O' is -1.
            # The string_to_board charmap is {' ':0, 'X':1, 'O':-1}
            # Classic alternating draw pattern:
            # XOXOXOX
            # OXOXOXO
            # XOXOXOX
            # OXOXOXO
            # XOXOXOX
            # OXOXOXO
            # This XOXOXO pattern is actually a WIN for X.
            # Using the known draw pattern instead for board6_full_no_win_p1:
            # O O X X X O O
            # X X O O O X X
            # O O X X X O O
            # X X O O O X X
            # O O X X X O O
            # X X O O O X X
            "OOXXXOO" # Row 0
            "XXOOOXX" # Row 1
            "OOXXXOO" # Row 2
            "XXOOOXX" # Row 3
            "OOXXXOO" # Row 4
            "XXOOOXX" # Row 5
        )


        boards_batch = torch.stack([
            board1_win_h,
            board2_win_v,
            board3_no_win,
            board4_win_dpos,
            board5_win_dneg,
            self.empty_board,
            board6_full_no_win_p1
        ])

        expected_wins = torch.tensor([True, True, False, True, True, False, False])

        # Test on CPU
        self._reset_conv_kernel() # Ensure kernel is created on CPU
        actual_wins_cpu = check_win_batch_conv(boards_batch.cpu())
        self.assertEqual(actual_wins_cpu.dtype, torch.bool)
        self.assertTrue(torch.equal(actual_wins_cpu, expected_wins.cpu()))

        if torch.cuda.is_available():
            self._reset_conv_kernel() # Ensure kernel is created on GPU
            actual_wins_gpu = check_win_batch_conv(boards_batch.cuda())
            self.assertEqual(actual_wins_gpu.dtype, torch.bool)
            self.assertTrue(torch.equal(actual_wins_gpu, expected_wins.cuda()))

        # Test with player 2 pieces (should not detect win for player 1)
        self._reset_conv_kernel()
        board_p2_win = self.empty_board.clone()
        board_p2_win[self.rows-1, 0:4] = -1 # Player 2 horizontal line
        results_p2 = check_win_batch_conv(torch.stack([board_p2_win, self.empty_board]))
        self.assertFalse(results_p2[0]) # No win for player 1
        self.assertFalse(results_p2[1])

        # Test with empty batch
        self._reset_conv_kernel()
        empty_batch = torch.empty((0, self.rows, self.cols), dtype=torch.int8)
        results_empty = check_win_batch_conv(empty_batch)
        self.assertEqual(results_empty.shape, (0,))
        self.assertEqual(results_empty.dtype, torch.bool)

        # Test with a board that has lines of 3, but not 4
        self._reset_conv_kernel()
        board_almost_win = self.empty_board.clone()
        board_almost_win[self.rows-1, 0:3] = 1 # Horizontal 3: (5,0), (5,1), (5,2)
        # Vertical 3 in a different column to avoid accidental win from overlap
        board_almost_win[self.rows-3:, 4] = 1  # Vertical 3: (3,4), (4,4), (5,4)
        results_almost = check_win_batch_conv(board_almost_win.unsqueeze(0))
        self.assertFalse(results_almost[0])

    def test_make_move_and_check_batch(self):
        self._reset_conv_kernel() # Ensure conv kernel is reset for device consistency

        # Board 1: Will be a win after the move
        board1_pre_win = self.empty_board.clone()
        board1_pre_win[self.rows - 1, 0:3] = 1 # P1 has 3 in a row
        move1 = 3 # Winning move for P1

        # Board 2: Will not be a win, not a draw
        board2_continue = self.empty_board.clone()
        board2_continue[self.rows -1, 0] = 1
        move2 = 1 # Non-winning move

        # Board 3: Will be a draw (board becomes full, no win)
        # Create a board that is almost full, and the next move fills it without a win
        # XOXOXO. (O to move in last col)
        # OXOXOX.
        # XOXOXO.
        # OXOXOX.
        # XOXOXO.
        # OXOXOXX (X made last move, O is current player if we were alternating)
        # For this function, player 1 (X) is always the one moving.
        # So, board needs to be set up such that player 1's move causes a draw.
        # This board, when player 1 moves to (0,6), should be full and a draw.
        board3_pre_draw = torch.tensor([
            [ 1, -1,  1, -1,  1, -1,  0], # P1 (X) moves into (0,6)
            [-1,  1, -1,  1, -1,  1, -1], # Target: filled board, no win for X
            [ 1, -1,  1, -1,  1, -1,  1], # This configuration resulted in an X win previously.
            [-1,  1, -1,  1, -1,  1, -1], # Diagonal win: (0,6)X, (1,5)X, (2,4)X, (3,3)X
            [ 1, -1,  1, -1,  1, -1,  1], # Values: X, board[1,5]=X, board[2,4]=X, board[3,3]=X
            [-1,  1, -1,  1, -1,  1, -1]  # Need to change one of board[1,5], board[2,4], board[3,3] to O (-1)
        ], dtype=torch.int8) # Original problematic board from test run
        # To make it a draw (fill column 0, last available spot at (0,6) for player 1):
        # X O X O X O . (P1 to fill (0,6) with X)
        # O X O X O X O
        # X O X O X O X
        # O X O X O X O
        # X O X O X O X
        # O X O X O X O
        # This is a standard draw pattern. This was actually a WIN for X.
        # New board3_pre_draw for a real draw:
        # O O X X X O .  (P1 'X' to fill (0,6))
        # X X O O O X X
        # O O X X X O O
        # X X O O O X X
        # O O X X X O O
        # X X O O O X X
        board3_pre_draw = torch.tensor([
            [-1, -1,  1,  1,  1, -1,  0], # P1 (X) moves to (0,6)
            [ 1,  1, -1, -1, -1,  1,  1],
            [-1, -1,  1,  1,  1, -1, -1],
            [ 1,  1, -1, -1, -1,  1,  1],
            [-1, -1,  1,  1,  1, -1, -1],
            [ 1,  1, -1, -1, -1,  1,  1]
        ], dtype=torch.int8)
        move3 = 6 # Move to fill the last spot (0,6)

        # Board 4: Empty board, simple move
        board4_empty = self.empty_board.clone()
        move4 = 0

        boards_batch = torch.stack([board1_pre_win, board2_continue, board3_pre_draw, board4_empty])
        moves_batch = torch.tensor([move1, move2, move3, move4], dtype=torch.long)

        new_boards, wins, draws = make_move_and_check_batch(boards_batch, moves_batch)

        # Check board 1 (win)
        self.assertTrue(wins[0])
        self.assertFalse(draws[0])
        self.assertEqual(new_boards[0, self.rows - 1, move1], 1) # Piece placed

        # Check board 2 (continue)
        self.assertFalse(wins[1])
        self.assertFalse(draws[1])
        self.assertEqual(new_boards[1, self.rows - 1, move2], 1)

        # Check board 3 (draw)
        self.assertFalse(wins[2], f"Board 3 should be a draw, not a win. Board:\n{new_boards[2]}")
        self.assertTrue(draws[2], f"Board 3 should be a draw. Board:\n{new_boards[2]}")
        self.assertEqual(new_boards[2, 0, move3], 1) # Piece placed in the top row
        self.assertTrue(torch.all(new_boards[2] != 0)) # Board is full

        # Check board 4 (simple move on empty)
        self.assertFalse(wins[3])
        self.assertFalse(draws[3])
        self.assertEqual(new_boards[3, self.rows - 1, move4], 1)

        # Test with an invalid move (full column) in one of the batch items
        board_fill_col0 = self.empty_board.clone()
        for r in range(self.rows): # Fill column 0
            board_fill_col0[r, 0] = 1

        batch_with_invalid = torch.stack([self.empty_board, board_fill_col0])
        moves_invalid = torch.tensor([0,0], dtype=torch.long) # Try to move in col 0 for both

        with self.assertRaisesRegex(ValueError, "Attempted move on a full column in make_moves_batch!"):
            make_move_and_check_batch(batch_with_invalid, moves_invalid)

        # Test empty batch
        self._reset_conv_kernel()
        empty_boards_input = torch.empty((0, self.rows, self.cols), dtype=torch.int8)
        empty_moves_input = torch.empty((0,), dtype=torch.long)
        res_boards, res_wins, res_draws = make_move_and_check_batch(empty_boards_input, empty_moves_input)
        self.assertEqual(res_boards.shape, (0, self.rows, self.cols))
        self.assertEqual(res_wins.shape, (0,))
        self.assertEqual(res_draws.shape, (0,))


if __name__ == '__main__':
    unittest.main()
