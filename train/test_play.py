from globals import init_device
from main import play_parallel_with_results, play_parallel_with_results
from model import RandomConnect4, load_frozen_model
from board import pretty_print_board
import torch

from play import play_parallel

def test_play_parallel():
    init_device(False)
    model = RandomConnect4()
    NUM_GAMES = 100

    torch.manual_seed(1)
    results = play_parallel(model, model, NUM_GAMES)
    #assert results == (57, 42, 1), f"Results do not match" # results seem to differ by PyTorch version
    assert results[0] + results[1] + results[2] == NUM_GAMES
    assert results[0] > 50
    assert results[2] < 5


def test_play_parallel_with_results():
    init_device(False)
    model = RandomConnect4()
    NUM_GAMES = 1000

    torch.manual_seed(0)
    b, m, r, done, wr = play_parallel_with_results(model, model, 0, NUM_GAMES)
    ns = len(b)         # number of board states
    assert len(m) == ns
    assert r.shape == (ns,)

    torch.manual_seed(0)
    b2, m2, r2, done2, wr2 = play_parallel_with_results(model, model, 0, NUM_GAMES)
    assert len(b2) == ns
    for bs1, bs2 in zip(b, b2):
        assert torch.equal(bs1, bs2)
    assert torch.equal(m, m2)
    assert torch.equal(r, r2)
