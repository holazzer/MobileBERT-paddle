import paddle
import numpy as np

from collections import OrderedDict


data: OrderedDict = paddle.load('d:/mb.bin')

# 1. remove the 'mobilebert.' prefix
data_ = OrderedDict()
for k, v in data.items():
    data_[k[11:]] = v

data = data_

# 2. some weights should be transposed.
wrong_shape = []

from z_debug_script import MobileBertModel, MobileBertConfig
target = MobileBertModel(MobileBertConfig()).state_dict()
src = data

for k in target:
    if src[k].shape != target[k].shape:
        wrong_shape.append(k)

for k in wrong_shape:
    data[k] = data[k].t()

# 3. save
paddle.save(data, 'd:/fixed-mb.bin')


# find what types of layers need having their weight transposed

sd = target
msd = src

























