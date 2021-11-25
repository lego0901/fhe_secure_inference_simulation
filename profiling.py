import torch
import numpy as np


class Profiling:
    def __init__(self, division=None, device=None):
        if division is None:
            division = Profiling._default_division()
        self.division = division
        self.counts = np.zeros(len(self.division) + 1, dtype=np.long)
        self.device = device or "cpu"

    def init_counts(self):
        self.counts = np.zeros(len(self.division) + 1, dtype=np.long)

    def update(self, x):
        x = x.flatten()
        classes = torch.zeros(*x.shape, dtype=torch.uint8).to(self.device)
        for border in self.division:
            classes += (x >= border).to(self.device)
        for i in range(len(self.counts)):
            self.counts[i] += (classes == i).sum().item()

    @staticmethod
    def _default_division():
        return [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
