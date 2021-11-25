import torch
import torch.nn as nn

from components import MaxPool2d2x2, FHEIncompatible, ReLU, LeakyReLU


class FHEModule(nn.Module):
    def get_profiling_results(self):
        profiling_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, FHEIncompatible):
                profiling_dict[name] = module.profiling_layer.counts
        return profiling_dict

    def set_use_profiling_layers(self, use_profiling_layers):
        for module in self.modules():
            if isinstance(module, FHEIncompatible):
                module.use_profiling_layer = use_profiling_layers

    def set_approx_degree(self, n):
        for module in self.modules():
            if isinstance(module, FHEIncompatible):
                module.n = n

    def set_block_type(self, block_type):
        for module in self.modules():
            if isinstance(module, FHEIncompatible):
                module.block_type = block_type

    def set_clip_before(self, clip_before):
        for module in self.modules():
            if isinstance(module, FHEIncompatible):
                module.clip_before = clip_before

    def sum_activation_lk_norm(self, p=1):
        activation_sum = 0
        for module in self.modules():
            if isinstance(module, ReLU) or isinstance(module, LeakyReLU):
                activation_sum += torch.norm(module.activation, p)
        return activation_sum

    @staticmethod
    def make_relu(n, block_type, use_leaky_relu, negative_slope):
        if use_leaky_relu:
            return LeakyReLU(negative_slope, n, block_type)
        else:
            return ReLU(n, block_type)


class SimpleMNISTNet(FHEModule):
    def __init__(
        self, n=-1, block_type="naive", use_leaky_relu=False, negative_slope=0.1
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.mp1 = MaxPool2d2x2(n, block_type)
        self.relu1 = self.make_relu(n, block_type, use_leaky_relu, negative_slope)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.mp2 = MaxPool2d2x2(n, block_type)
        self.relu2 = self.make_relu(n, block_type, use_leaky_relu, negative_slope)
        self.fc3 = nn.Linear(32 * 6 * 6, 120)
        self.relu3 = self.make_relu(n, block_type, use_leaky_relu, negative_slope)
        self.fc4 = nn.Linear(120, 84)
        self.relu4 = self.make_relu(n, block_type, use_leaky_relu, negative_slope)
        self.fc5 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.mp1(self.relu1(self.conv1(x)))
        x = self.mp2(self.relu2(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.fc5(x)
        return x
