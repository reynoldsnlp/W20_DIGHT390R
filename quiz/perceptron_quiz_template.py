"""Quiz to test whether students can build Perceptron from scratch."""

import numpy as np
import torch

LEFT_CENTER = (3, 3)
RIGHT_CENTER = (3, -2)

lr = 0.01
input_dim = 2

batch_size = 1000
n_epochs = 12
n_batches = 5


def get_toy_data(batch_size,
                 left_center=LEFT_CENTER,
                 right_center=RIGHT_CENTER):
    x_data = []
    y_targets = np.zeros(batch_size)
    for batch_i in range(batch_size):
        if np.random.random() > 0.5:
            x_data.append(np.random.normal(loc=left_center))
        else:
            x_data.append(np.random.normal(loc=right_center))
            y_targets[batch_i] = 1
    return (torch.tensor(x_data, dtype=torch.float32),
            torch.tensor(y_targets, dtype=torch.float32))
