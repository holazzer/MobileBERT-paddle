import paddle
from paddle import nn
from paddle.autograd import PyLayer, PyLayerContext
import numpy as np


class NumpyMatmul(PyLayer):
    @staticmethod
    def forward(ctx: 'PyLayerContext', a, b):
        an = a.numpy()
        bn = b.numpy()
        mm = np.matmul(an, bn)
        mp = paddle.to_tensor(mm)
        ctx.an = an
        ctx.bn = bn
        return mp

    @staticmethod
    def backward(ctx: 'PyLayerContext', dy):
        an = ctx.an       # (1,2)
        bn = ctx.bn       # (2,1)
        yn = dy.numpy()   # (1,1)
        print(an.shape, bn.shape, yn.shape)
        ga = np.matmul(yn, bn.T)
        gb = np.matmul(an.T, yn)
        ga = paddle.to_tensor(ga)
        gb = paddle.to_tensor(gb)
        return ga, gb


class Linear(nn.Layer):
    def __init__(self, in_features, out_features,
                 weight_attr=None, bias_attr=None,
                 name=None):
        super().__init__()
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self.weight = self.create_parameter(
            shape=[in_features, out_features],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False)
        self.bias = self.create_parameter(
            shape=[out_features],
            attr=self._bias_attr,
            dtype=self._dtype,
            is_bias=True)
        self.name = name

    def forward(self, input):
        out = NumpyMatmul.apply(input, self.weight) + self.bias
        return out

    def extra_repr(self):
        name_str = ', name={}'.format(self.name) if self.name else ''
        return 'in_features={}, out_features={}, dtype={}{}'.format(
            self.weight.shape[0], self.weight.shape[1], self._dtype, name_str)


