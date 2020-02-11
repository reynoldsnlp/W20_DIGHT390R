import numpy as np
import torch

batch_size = 10_000
input_size = 2
hidden_size = 42
output_size = 1
lr = 0.01

n_epochs = 10
n_batches = 10


def get_toy_data(batch_size):
    LABELS = [0, 0, 1, 1]
    CENTERS = [(-3, -3), (3, 3), (3, -3), (-3, 3)]
    assert len(CENTERS) == len(LABELS), 'centers and labels have diff len'
    x_data = []
    y_targets = np.zeros(batch_size)
    n_centers = len(CENTERS)
    for batch_i in range(batch_size):
        center_idx = np.random.randint(0, n_centers)
        x_data.append(np.random.normal(loc=CENTERS[center_idx]))
        y_targets[batch_i] = LABELS[center_idx]
    return (torch.tensor(x_data, dtype=torch.float32),
            torch.tensor(y_targets, dtype=torch.float32))


# 1) Write a new Pytorch module with two fully connected layers

# 2) Write a training loop using `get_toy_data()`. On each iteration, print
# your loss.
