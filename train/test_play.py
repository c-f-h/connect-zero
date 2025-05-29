from globals import init_device
from main import play_parallel_with_results, play_parallel_with_results
from model import RandomConnect4, load_frozen_model
from board import pretty_print_board
import torch

from play import play_parallel

def test_play_parallel():
    model = RandomConnect4()
    NUM_GAMES = 100

    torch.manual_seed(0)
    results = play_parallel(model, model, NUM_GAMES)
    assert results == (53, 47, 0), f"Results do not match"


def test_play_parallel_with_results():
    model = RandomConnect4()
    NUM_GAMES = 1000

    from pyinstrument import profile

    torch.manual_seed(0)
    with profile():
        b, m, r = play_parallel_with_results(model, model, 0, NUM_GAMES)
    ns = len(b)         # number of board states
    assert len(m) == ns
    assert r.shape == (ns,)

    torch.manual_seed(0)
    with profile():
        b2, m2, r2 = play_parallel_with_results(model, model, 0, NUM_GAMES)
    assert len(b2) == ns
    for bs1, bs2 in zip(b, b2):
        assert torch.equal(bs1, bs2)
    assert m2 == m          # list
    assert torch.equal(r, r2)
