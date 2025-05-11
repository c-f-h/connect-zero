########## Example 4: Initial training against RandomPunisher with A2C ##########

from main import set_params, train_against_opponents, init_device
from model import Connect4CNN_Mk4, RandomPunisher

if __name__ == "__main__":
    init_device(False)
    set_params(
        learning_rate=1e-4,
        entropy_bonus=0.05,
        value_loss_weight=1.5,
        reward_discount=0.90,
        bootstrap_value=True,      # enable A2C
    )

    model = Connect4CNN_Mk4(value_head=True)
    opponents = [
        RandomPunisher()
    ]
    train_against_opponents(model, opponents, debug=False)
