########## Example 3: Self-play using REINFORCE with baseline ##########

from globals import init_device
from main import set_params, self_play_loop
from model import Connect4CNN_Mk4

if __name__ == "__main__":
    init_device(False)
    set_params(
        entropy_bonus=0.05,
        reward_discount=0.90,
        bootstrap_value=False,      # disable A2C - defaults to REINFORCE with baseline
    )
    constr = lambda: Connect4CNN_Mk4(value_head=True)   # model constructor
    self_play_loop(constr, games_per_batch=50, batches_per_epoch=100, learning_rate=1e-4, fname_prefix="ex3")
