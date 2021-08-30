import paddle
from paddle import nn

from typing import Tuple, Union


class NoNorm(nn.Layer):
    def __init__(self, feat_size: Union[int, Tuple], eps=None):
        super().__init__()

        if isinstance(feat_size, int): feat_size = (feat_size, )

        bias = paddle.zeros(feat_size)
        weight = paddle.ones(feat_size)

        self.bias = paddle.create_parameter(
            shape=bias.shape, dtype=bias.dtype,
            is_bias=True,
            default_initializer=paddle.nn.initializer.Assign(bias)
        )

        self.weight = paddle.create_parameter(
            shape=weight.shape, dtype=weight.dtype,
            default_initializer=paddle.nn.initializer.Assign(weight)
        )

    def forward(self, input_tensor):
        return input_tensor * self.weight + self.bias  # Hadamard product

