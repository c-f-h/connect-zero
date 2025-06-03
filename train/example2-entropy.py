########## Example 2: Entropy bonus to avoid policy collapse ##########

from main import set_params, self_play_loop
from model import SimpleMLPModel

if __name__ == "__main__":
    set_params(
        entropy_bonus=0.05,     # entropy bonus to avoid policy collapse
        reward_discount=0.90,
        bootstrap_value=False,
    )
    constr = SimpleMLPModel   # model constructor
    self_play_loop(constr, games_per_batch=50, batches_per_epoch=100, learning_rate=1e-3, fname_prefix="ex2",
                   games_per_batch=50, batches_per_epoch=100)
