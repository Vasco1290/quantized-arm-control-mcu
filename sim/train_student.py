"""Train the student network to copy the teacher (behaviour cloning).

Plain supervised learning: show the network an observation, compare its answer with the
teacher's answer, nudge the weights to shrink the difference. Repeat many times.
Saves the weights as NumPy arrays in models/student_float.npz.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from collect import OUT_PATH as DATA_PATH
from student import MODEL_PATH, Student

EPISODE_LEN = 50
VAL_FRACTION = 0.1    # last 10% of episodes are held back to check for memorising
EPOCHS = 60           # passes over the whole dataset
BATCH_SIZE = 256      # pairs per weight update
LEARNING_RATE = 1e-3  # how big each nudge is
HIDDEN = 64


def main():
    torch.manual_seed(0)  # same starting weights every run -> same result every run
    d = np.load(DATA_PATH)
    obs, act = d["obs"], d["act"]

    # Split by whole episodes, so validation targets are never seen in training.
    n_val = int(len(obs) // EPISODE_LEN * VAL_FRACTION) * EPISODE_LEN
    train_obs, val_obs = obs[:-n_val], obs[-n_val:]
    train_act, val_act = act[:-n_val], act[-n_val:]

    # Input scaling: joint speeds can be ~10 while positions are ~0.5. Shifting and
    # dividing each input by its typical size puts them all around -1..+1.
    in_mean = train_obs.mean(axis=0)
    in_std = train_obs.std(axis=0) + 1e-8

    def to_tensor(o):
        return torch.tensor((o - in_mean) / in_std)

    x_train, y_train = to_tensor(train_obs), torch.tensor(train_act)
    x_val, y_val = to_tensor(val_obs), torch.tensor(val_act)

    net = nn.Sequential(
        nn.Linear(10, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, 2),
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    # Smaller nudges near the end, to settle into a good spot instead of bouncing around it.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.MSELoss()  # mean of (student answer - teacher answer) squared

    for epoch in range(EPOCHS):
        order = torch.randperm(len(x_train))  # shuffle so batches are mixed
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            loss = loss_fn(net(x_train[idx]), y_train[idx])
            optimizer.zero_grad()
            loss.backward()   # work out which way to nudge every weight
            optimizer.step()  # nudge them
        scheduler.step()
        if epoch % 10 == 9 or epoch == 0:
            with torch.no_grad():
                train_loss = loss_fn(net(x_train), y_train).item()
                val_loss = loss_fn(net(x_val), y_val).item()
            print(f"epoch {epoch + 1:3d}  train error {train_loss:.5f}  validation error {val_loss:.5f}")

    # Save as plain NumPy. PyTorch stores weights as (out, in); our NumPy code uses (in, out).
    layers = [m for m in net if isinstance(m, nn.Linear)]
    MODEL_PATH.parent.mkdir(exist_ok=True)
    np.savez(MODEL_PATH, in_mean=in_mean, in_std=in_std,
             **{f"W{k + 1}": l.weight.detach().numpy().T for k, l in enumerate(layers)},
             **{f"b{k + 1}": l.bias.detach().numpy() for k, l in enumerate(layers)})

    # Check: the NumPy version must give the same answers as the PyTorch version.
    with torch.no_grad():
        torch_out = net(x_val).clamp(-1, 1).numpy()
    numpy_out = Student(MODEL_PATH)(val_obs)
    print(f"saved {MODEL_PATH}; NumPy vs PyTorch max difference: {np.abs(torch_out - numpy_out).max():.2e}")


if __name__ == "__main__":
    main()
