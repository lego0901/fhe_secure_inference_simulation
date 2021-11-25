import torch.nn as nn

from approximations import maxpool2d, relu, leaky_relu
from profiling import Profiling


class FHEIncompatible(nn.Module):
    def __init__(
        self,
        n=-1,
        block_type="naive",
        clip_before=False,
        attach_profiling_layer=True,
        use_profiling_layer=False,
        profiling_layer_division=None,
    ):
        super().__init__()
        self.n = n
        self.block_type = block_type
        self.clip_before = clip_before
        if attach_profiling_layer:
            self.use_profiling_layer = use_profiling_layer
            self.profiling_layer = Profiling(profiling_layer_division)
        else:
            self.use_profiling_layer = False
        self.activation = 0

    def operator(self, _):
        raise NotImplementedError

    def forward(self, x):
        self.activation = x
        if self.use_profiling_layer:
            self.profiling_layer.update(x)
        return self.operator(x)


class MaxPool2d2x2(FHEIncompatible):
    def operator(self, x):
        return maxpool2d(x, self.n, self.block_type, self.clip_before)


class ReLU(FHEIncompatible):
    def operator(self, x):
        return relu(x, self.n, self.block_type, self.clip_before)


class LeakyReLU(FHEIncompatible):
    def __init__(
        self,
        negative_slope=0.1,
        n=-1,
        block_type="naive",
        clip_before=False,
        attach_profiling_layer=True,
        use_profiling_layer=False,
        profiling_layer_division=None,
    ):
        super().__init__(
            n,
            block_type,
            clip_before,
            attach_profiling_layer,
            use_profiling_layer,
            profiling_layer_division,
        )
        self.negative_slope = negative_slope

    def operator(self, x):
        return leaky_relu(
            x, self.n, self.negative_slope, self.block_type, self.clip_before
        )
