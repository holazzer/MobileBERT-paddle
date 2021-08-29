import paddle
from paddle import nn

from util import NORM2FN


class BottleneckLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intra_bottleneck_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.intra_bottleneck_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        layer_input = self.dense(hidden_states)
        layer_input = self.LayerNorm(layer_input)
        return layer_input




