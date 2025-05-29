########## Example 5: Initial training against RandomPunisher with PPO ##########

from globals import init_device
from main import set_params, train_against_opponents
from model import Connect4CNN_Mk4, RandomPunisher, load_frozen_model

if __name__ == "__main__":
    device = init_device(True)
    set_params(
        algorithm="PPO",
        bootstrap_value=True,
        learning_rate=1e-4,
        entropy_bonus=0.05,
        value_loss_weight=1.5,
        reward_discount=0.90,
        # PPO-specific parameters
        ppo_clip_epsilon=0.2,
        ppo_epochs=4,
        ppo_target_kl=0.01,
    )

    model = Connect4CNN_Mk4(value_head=True).to(device)
    opponents = [
        RandomPunisher()
    ]
    train_against_opponents(model, opponents, checkpoint_file="ex5_checkpoint.pth", debug=False)
