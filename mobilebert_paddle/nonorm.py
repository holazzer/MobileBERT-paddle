import paddle
from paddle import nn


class NoNorm(nn.Layer):
    def __init__(self, feat_size, eps=None):
        super().__init__()
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

