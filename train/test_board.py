import torch
import unittest

import board
import globals

class TestBoard(unittest.TestCase):
    def setUp(self):
        # Standard 6x7 Connect Four board
        self.rows = globals.ROWS
        self.cols = globals.COLS
        self.empty_board = torch.zeros((self.rows, self.cols), dtype=torch.int8)

    def test_make_move_valid(self):
        b = self.empty_board.clone()
        new_board, row = board.make_move(b, 3)
        self.assertEqual(row, self.rows - 1) # Piece should be in the last row
        self.assertEqual(new_board[row, 3], 1) # Piece of player 1
        self.assertNotEqual(id(b), id(new_board)) # Should be a new tensor

        # Make another move in the same column
        b2 = new_board
        new_board, row = board.make_move(b2, 3)
        self.assertEqual(row, self.rows - 2) # Piece should be in the second to last row
        self.assertEqual(new_board[row, 3], 1)

    def test_make_move_invalid_column(self):
        with self.assertRaisesRegex(torch.jit.Error, "Invalid column index: 7"):
            board.make_move(self.empty_board, self.cols)
        with self.assertRaisesRegex(torch.jit.Error, "Invalid column index: -1"):
            board.make_move(self.empty_board, -1)

    def test_make_move_full_column(self):
        b = self.empty_board.clone()
        # Fill one column
        for i in range(self.rows):
            b, _ = board.make_move(b, 0)

        with self.assertRaisesRegex(torch.jit.Error, "Column 0 is full."):
            board.make_move(b, 0)

    def test_check_win_horizontal(self):
        # Test horizontal win
        b = self.empty_board.clone()
        b[self.rows - 1, 0] = 1
        b[self.rows - 1, 1] = 1
        b[self.rows - 1, 2] = 1
        self.assertFalse(board.check_win_after_move(b, self.rows - 1, 2)) # Not a win yet
        b[self.rows - 1, 3] = 1
        self.assertTrue(board.check_win_after_move(b, self.rows - 1, 3))
        b[self.rows - 1, 1] = -1
        self.assertFalse(board.check_win_after_move(b, self.rows - 1, 3))

    def test_check_win_vertical(self):
        b = self.empty_board.clone()
        b[self.rows - 1, 2] = 1
        b[self.rows - 2, 2] = 1
        b[self.rows - 3, 2] = 1
        self.assertFalse(board.check_win_after_move(b, self.rows - 3, 2)) # Not a win yet
        b[self.rows - 4, 2] = 1
        self.assertTrue(board.check_win_after_move(b, self.rows - 4, 2))
        b[self.rows - 2, 2] = -1
        self.assertFalse(board.check_win_after_move(b, self.rows - 4, 2))

    def test_check_win_diagonal_positive(self):
        b = self.empty_board.clone()
        r, c = self.rows - 1, 0
        b[r, c] = 1
        b[r - 1, c + 1] = 1
        b[r - 2, c + 2] = 1
        self.assertFalse(board.check_win_after_move(b, r-2, c+2))
        b[r - 3, c + 3] = 1
        self.assertTrue(board.check_win_after_move(b, r - 3, c + 3))
        b[r-1, c+1] = -1
        self.assertFalse(board.check_win_after_move(b, r-3, c+3))

        b = self.empty_board.clone()
        r, c = self.rows -1, 3
        b[r,c] = 1
        b[r-1, c+1] = 1
        b[r-2, c+2] = 1
        self.assertFalse(board.check_win_after_move(b, r-2, c+2))
        b[r-3, c+3] = 1
        self.assertTrue(board.check_win_after_move(b, r-3, c+3))

    def test_check_win_diagonal_negative(self):
        b = self.empty_board.clone()
        r, c = self.rows - 1, 3
        b[r, c] = 1
        b[r - 1, c - 1] = 1
        b[r - 2, c - 2] = 1
        self.assertFalse(board.check_win_after_move(b, r-2, c-2))
        b[r - 3, c - 3] = 1
        self.assertTrue(board.check_win_after_move(b, r - 3, c - 3))
        b[r-1, c-1] = -1
        self.assertFalse(board.check_win_after_move(b, r-3, c-3))

        b = self.empty_board.clone()
        r,c = self.rows-1, 3
        b[r,c] = 1
        b[r-1,c-1] = 1
        b[r-2,c-2] = 1
        self.assertFalse(board.check_win_after_move(b, r-2, c-2))
        b[r-3,c-3] = 1
        self.assertTrue(board.check_win_after_move(b, r-3, c-3))

    def test_check_win_no_win(self):
        b = self.empty_board.clone()
        b[self.rows - 1, 0] = 1
        b[self.rows - 1, 1] = 1
        b[self.rows - 1, 2] = 1
        self.assertFalse(board.check_win_after_move(b, self.rows - 1, 2))
        b[self.rows - 2, 0] = 1
        b[self.rows - 3, 0] = 1
        self.assertFalse(board.check_win_after_move(b, self.rows - 3, 0))

    def test_make_move_and_check_no_win(self):
        b = self.empty_board.clone()
        new_board, win = board.make_move_and_check(b, 0)
        self.assertFalse(win)
        self.assertEqual(new_board[self.rows - 1, 0], 1)
        self.assertNotEqual(id(b), id(new_board))

    def test_make_move_and_check_win(self):
        b = self.empty_board.clone()
        b[self.rows - 1, 0] = 1
        b[self.rows - 1, 1] = 1
        b[self.rows - 1, 2] = 1
        new_board, win = board.make_move_and_check(b, 3)
        self.assertTrue(win)
        self.assertEqual(new_board[self.rows - 1, 3], 1)

        b = self.empty_board.clone()
        b[self.rows - 1, 0] = 1
        b[self.rows - 2, 0] = 1
        b[self.rows - 3, 0] = 1
        new_board, win = board.make_move_and_check(b, 0)
        self.assertTrue(win)
        self.assertEqual(new_board[self.rows - 4, 0], 1)

    def test_make_move_and_check_full_column(self):
        b = self.empty_board.clone()
        for i in range(self.rows):
            b, _ = board.make_move(b, 0)

        with self.assertRaisesRegex(torch.jit.Error, "Column 0 is full."):
            board.make_move_and_check(b, 0)

    def test_is_board_full_batch(self):
        empty_boards = torch.zeros((2, self.rows, self.cols), dtype=torch.int8)
        results = board.is_board_full_batch(empty_boards)
        self.assertFalse(results[0])
        self.assertFalse(results[1])
        self.assertEqual(results.dtype, torch.bool)

        partially_filled_board = self.empty_board.clone()
        partially_filled_board[self.rows -1, 0] = 1
        partially_filled_board[0, self.cols -1] = -1

        boards = torch.stack([self.empty_board, partially_filled_board])
        results = board.is_board_full_batch(boards)
        self.assertFalse(results[0])
        self.assertFalse(results[1])

        full_board_1s = torch.ones((self.rows, self.cols), dtype=torch.int8)
        full_board_mixed = torch.ones((self.rows, self.cols), dtype=torch.int8)
        full_board_mixed[0, ::2] = -1
        full_board_mixed[1, 1::2] = -1

        almost_full_board = torch.ones((self.rows, self.cols), dtype=torch.int8)
        almost_full_board[0,0] = 0

        full_boards_batch = torch.stack([full_board_1s, full_board_mixed, self.empty_board, almost_full_board])
        results = board.is_board_full_batch(full_boards_batch)
        self.assertTrue(results[0])
        self.assertTrue(results[1])
        self.assertFalse(results[2])
        self.assertFalse(results[3])

        single_full_board = full_board_1s.unsqueeze(0)
        results = board.is_board_full_batch(single_full_board)
        self.assertTrue(results[0])
        self.assertEqual(results.shape, (1,))

        single_empty_board = self.empty_board.unsqueeze(0)
        results = board.is_board_full_batch(single_empty_board)
        self.assertFalse(results[0])
        self.assertEqual(results.shape, (1,))

        zero_boards = torch.empty((0, self.rows, self.cols), dtype=torch.int8)
        results = board.is_board_full_batch(zero_boards)
        self.assertEqual(results.shape, (0,))

    def _reset_conv_kernel(self):
        import board as board_module # explicit import for module attribute access
        board_module.g_win_conv_kernel = None

    def test_check_win_batch_conv(self):
        self._reset_conv_kernel()
        board1_win_h = self.empty_board.clone()
        board1_win_h[self.rows - 1, 0:4] = 1

        board2_win_v = self.empty_board.clone()
        board2_win_v[self.rows - 4:, 2] = 1

        board3_no_win = self.empty_board.clone()
        board3_no_win[self.rows -1, 0] = 1
        board3_no_win[self.rows -1, 1] = 1
        board3_no_win[self.rows -1, 2] = -1
        board3_no_win[self.rows -1, 3] = 1

        board4_win_dpos = self.empty_board.clone()
        for i in range(4):
            board4_win_dpos[self.rows - 1 - i, i] = 1

        board5_win_dneg = self.empty_board.clone()
        for i in range(4):
            board5_win_dneg[self.rows - 1 - i, self.cols - 1 - i] = 1

        board6_full_no_win_p1 = board.string_to_board_test_format(
            "OOXXXOO"
            "XXOOOXX"
            "OOXXXOO"
            "XXOOOXX"
            "OOXXXOO"
            "XXOOOXX"
        )

        boards_batch = torch.stack([
            board1_win_h, board2_win_v, board3_no_win,
            board4_win_dpos, board5_win_dneg, self.empty_board, board6_full_no_win_p1
        ])
        expected_wins = torch.tensor([True, True, False, True, True, False, False])

        self._reset_conv_kernel()
        actual_wins_cpu = board.check_win_batch_conv(boards_batch.cpu())
        self.assertEqual(actual_wins_cpu.dtype, torch.bool)
        self.assertTrue(torch.equal(actual_wins_cpu, expected_wins.cpu()))

        if torch.cuda.is_available():
            self._reset_conv_kernel()
            actual_wins_gpu = board.check_win_batch_conv(boards_batch.cuda())
            self.assertEqual(actual_wins_gpu.dtype, torch.bool)
            self.assertTrue(torch.equal(actual_wins_gpu, expected_wins.cuda()))

        self._reset_conv_kernel()
        board_p2_win = self.empty_board.clone()
        board_p2_win[self.rows-1, 0:4] = -1
        results_p2 = board.check_win_batch_conv(torch.stack([board_p2_win, self.empty_board]))
        self.assertFalse(results_p2[0])
        self.assertFalse(results_p2[1])

        self._reset_conv_kernel()
        empty_batch = torch.empty((0, self.rows, self.cols), dtype=torch.int8)
        results_empty = board.check_win_batch_conv(empty_batch)
        self.assertEqual(results_empty.shape, (0,))
        self.assertEqual(results_empty.dtype, torch.bool)

        self._reset_conv_kernel()
        board_almost_win = self.empty_board.clone()
        board_almost_win[self.rows-1, 0:3] = 1
        board_almost_win[self.rows-3:, 4] = 1
        results_almost = board.check_win_batch_conv(board_almost_win.unsqueeze(0))
        self.assertFalse(results_almost[0])

    def test_make_move_and_check_batch(self):
        self._reset_conv_kernel()

        board1_pre_win = self.empty_board.clone()
        board1_pre_win[self.rows - 1, 0:3] = 1
        move1 = 3

        board2_continue = self.empty_board.clone()
        board2_continue[self.rows -1, 0] = 1
        move2 = 1

        board3_pre_draw = torch.tensor([
            [-1, -1,  1,  1,  1, -1,  0],
            [ 1,  1, -1, -1, -1,  1,  1],
            [-1, -1,  1,  1,  1, -1, -1],
            [ 1,  1, -1, -1, -1,  1,  1],
            [-1, -1,  1,  1,  1, -1, -1],
            [ 1,  1, -1, -1, -1,  1,  1]
        ], dtype=torch.int8)
        move3 = 6

        board4_empty = self.empty_board.clone()
        move4 = 0

        boards_batch = torch.stack([board1_pre_win, board2_continue, board3_pre_draw, board4_empty])
        moves_batch = torch.tensor([move1, move2, move3, move4], dtype=torch.long)

        new_boards, wins, draws = board.make_move_and_check_batch(boards_batch, moves_batch)

        self.assertTrue(wins[0])
        self.assertFalse(draws[0])
        self.assertEqual(new_boards[0, self.rows - 1, move1], 1)

        self.assertFalse(wins[1])
        self.assertFalse(draws[1])
        self.assertEqual(new_boards[1, self.rows - 1, move2], 1)

        self.assertFalse(wins[2], f"Board 3 should be a draw, not a win. Board:\n{new_boards[2]}")
        self.assertTrue(draws[2], f"Board 3 should be a draw. Board:\n{new_boards[2]}")
        self.assertEqual(new_boards[2, 0, move3], 1)
        self.assertTrue(torch.all(new_boards[2] != 0))

        self.assertFalse(wins[3])
        self.assertFalse(draws[3])
        self.assertEqual(new_boards[3, self.rows - 1, move4], 1)

        board_fill_col0 = self.empty_board.clone()
        for r in range(self.rows):
            board_fill_col0[r, 0] = 1

        batch_with_invalid = torch.stack([self.empty_board, board_fill_col0])
        moves_invalid = torch.tensor([0,0], dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "Attempted move on a full column in make_moves_batch!"):
            board.make_move_and_check_batch(batch_with_invalid, moves_invalid)

        self._reset_conv_kernel()
        empty_boards_input = torch.empty((0, self.rows, self.cols), dtype=torch.int8)
        empty_moves_input = torch.empty((0,), dtype=torch.long)
        res_boards, res_wins, res_draws = board.make_move_and_check_batch(empty_boards_input, empty_moves_input)
        self.assertEqual(res_boards.shape, (0, self.rows, self.cols))
        self.assertEqual(res_wins.shape, (0,))
        self.assertEqual(res_draws.shape, (0,))


if __name__ == '__main__':
    unittest.main()
