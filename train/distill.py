import torch
import torch.nn.functional as F

from main import play_multiple_against_model
from stats import BatchStats, UpdatablePlot

def distill(teacher, student, optimizer, num_batches: int, batch_size: int = 50):
    """
    Distill knowledge from a teacher model to a student model.

    Args:
        teacher_model: The pre-trained teacher model.
        student_model: The student model to be trained.
        optimizer: Optimizer for the student model.
        num_batches: Number of batches to train on.
        batch_size: Number of games per batch.
    """
    teacher.eval()
    student.train()

    device = next(student.parameters()).device

    stats = BatchStats(["policy_loss", "value_loss", "total_loss"])
    plot = UpdatablePlot(labels=[['Policy loss', 'Value loss'],
                                 ['Total loss', None]], show_last_n=200)

    for i in range(num_batches):
        # Generate games using the teacher model
        states, actions, rewards, done, wr = play_multiple_against_model(teacher, teacher, batch_size, augment_sym=True)

        # Teacher model outputs
        with torch.no_grad():
            teacher_logits, teacher_values = teacher(states)
            teacher_logprobs = F.log_softmax(teacher_logits, dim=-1)

        # Student model outputs
        student_logits, student_values = student(states)
        student_logprobs = F.log_softmax(student_logits, dim=-1)

        # Policy loss (KL divergence)
        policy_loss = F.kl_div(student_logprobs, teacher_logprobs, reduction='batchmean', log_target=True)

        # Value loss (MSE)
        value_loss = F.mse_loss(student_values.squeeze(-1), teacher_values.squeeze(-1), reduction='mean')

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss

        losses = (policy_loss.item(), value_loss.item(), total_loss.item())
        stats.add("policy_loss", losses[0])
        stats.add("value_loss", losses[1])
        stats.add("total_loss", losses[2])
        stats.aggregate()
        plot.update_from(stats, ("policy_loss", "value_loss", "total_loss"))

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"Batch {i}/{num_batches} done. Policy Loss: {losses[0]:.4f}, Value Loss: {losses[1]:.4f}")

import click

@click.command()
@click.argument('teacher', type=str)
@click.argument('student', type=str)
@click.option('--num-batches', '-n', default=250, help='Number of batches to train on.')
@click.option('--batch-size', default=50, help='Number of games per batch.')
@click.option('--lr', default=1e-3, help='Learning rate for the optimizer.')
def main_run(teacher, student, num_batches=250, batch_size=50, lr=1e-3):
    from model import load_frozen_model
    from globals import init_device
    device = init_device(True)

    teacher = load_frozen_model(teacher).to(device)
    #student = load_frozen_model("CNN-Mk4:best_cp.pth")
    student = load_frozen_model(student).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)

    distill(teacher, student, optimizer, num_batches=num_batches, batch_size=batch_size)
    # Save the distilled student model
    torch.save({
        'model_state_dict': student.state_dict(),
    }, "student.pth")



if __name__ == "__main__":
    main_run()