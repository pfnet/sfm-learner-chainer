import numpy
import six

import chainer
from chainer import configuration
from chainer import cuda
try:
    from chainer import function_node
    ParentClass = function_node.FunctionNode
except:
    from chainer import function
    ParentClass = function.Function
from chainer.utils import argument
from chainer.utils import type_check

def create_timer():
    start = chainer.cuda.Event()
    stop = chainer.cuda.Event()
    start.synchronize()
    start.record()
    return start, stop

def print_timer(start, stop, sentence=None):
    stop.record()
    stop.synchronize()
    elapsed_time = chainer.cuda.cupy.cuda.get_elapsed_time(
                           start, stop) / 1000
    if sentence is not None:
        print(sentence, elapsed_time)
    return elapsed_time


class CPU2GPU(ParentClass):

    """CPU2GPU"""

    def __init__(self):
        pass

    def check_type_forward(self, in_types):
        pass

    def forward(self, inputs):
        x = inputs[0]
        x = chainer.cuda.to_gpu(x)
        return x,

    def backward(self, x, gy):
        out = gy[0]
        out.to_cpu()
        return out,
        # return gy[0][self.batch_indexes, :, self.d_indexes, self.h_indexes, self.w_indexes],


def cpu2gpu(x):
    """CPU to GPU function."""
    return CPU2GPU().apply((x,))[0]


class GPU2CPU(ParentClass):

    """GPU2CPU"""

    def __init__(self):
        pass

    def check_type_forward(self, in_types):
        pass

    def forward(self, inputs):
        x = inputs[0]
        x = chainer.cuda.to_cpu(x)
        return x,

    def backward(self, x, gy):
        out = gy[0]
        out.to_gpu()
        return out,
        # return gy[0][self.batch_indexes, :, self.d_indexes, self.h_indexes, self.w_indexes],


def gpu2cpu(x):
    """GPU to CPU function."""
    return GPU2CPU().apply((x,))[0]
