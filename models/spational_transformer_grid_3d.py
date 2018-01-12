# TODO 未実装 3D spatial transfomer grid. cudnn高速化したかったらかいたほうがいいかも？
import numpy

import chainer
from chainer import cuda
from chainer import function
from chainer.utils import argument
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cuda.cudnn
    _sampler_type = libcudnn.CUDNN_SAMPLER_BILINEAR


class SpatialTransformerGrid3D(function.Function):

    def __init__(self, output_shape):
        self.output_shape = output_shape

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(n_in == 1)

        theta_type = in_types[0]
        type_check.expect(
            theta_type.dtype.char == 'f',
            theta_type.ndim == 3,
            theta_type.shape[1] == 2,
            theta_type.shape[2] == 3,
        )

    def forward_cpu(self, inputs):
        return self._forward(inputs)

    def forward_gpu(self, inputs):
        if not chainer.should_use_cudnn('>=auto', 5000):
            return self._forward(inputs)
        theta, = inputs
        B, _, _ = theta.shape
        H, W = self.output_shape
        grid_t = cuda.cupy.empty((B, H, W, 2), dtype=theta.dtype)

        # Unlike spatial_transformer_sampler,
        # channel size can be anything in this case.
        shape = numpy.array((B, 1, H, W), dtype=numpy.int32)
        theta = cuda.cupy.ascontiguousarray(theta)
        handle = cudnn.get_handle()
        self.st_desc =\
            cuda.cupy.cudnn.create_spatial_transformer_descriptor(
                _sampler_type, grid_t.dtype, len(shape), shape.ctypes.data)

        libcudnn.spatialTfGridGeneratorForward(
            handle, self.st_desc.value, theta.data.ptr, grid_t.data.ptr)
        grid = cuda.cupy.transpose(grid_t, (0, 3, 1, 2))

        return grid,

    def _forward(self, inputs):
        theta, = inputs
        H, W = self.output_shape
        B, _, _ = theta.shape
        xp = cuda.get_array_module(theta)

        ys, xs = xp.meshgrid(
            xp.linspace(-1, 1, H, dtype=numpy.float32),
            xp.linspace(-1, 1, W, dtype=numpy.float32), indexing='ij',
            copy=False
        )

        coords = xp.concatenate(
            [xs[None], ys[None], xp.ones((1, H, W), dtype=numpy.float32)],
            axis=0)
        grid = theta.dot(coords.reshape(3, H * W)).reshape(B, 2, H, W)
        return grid,

    def backward_cpu(self, inputs, grad_outputs):
        return self._backward(inputs, grad_outputs)

    def backward_gpu(self, inputs, grad_outputs):
        if not chainer.should_use_cudnn('>=auto', 5000):
            return self._backward(inputs, grad_outputs)
        theta, = inputs
        ggrid, = grad_outputs
        ggrid_t = cuda.cupy.transpose(ggrid, (0, 2, 3, 1))

        gtheta = cuda.cupy.empty_like(theta)
        handle = cudnn.get_handle()
        ggrid_t = cuda.cupy.ascontiguousarray(ggrid_t)
        libcudnn.spatialTfGridGeneratorBackward(
            handle, self.st_desc.value, ggrid_t.data.ptr, gtheta.data.ptr)
        return gtheta,

    def _backward(self, inputs, grad_outputs):
        theta, = inputs
        ggrid, = grad_outputs
        H, W = self.output_shape
        B, _, _ = theta.shape
        xp = cuda.get_array_module(theta)

        ys, xs = xp.meshgrid(
            xp.linspace(-1, 1, H, dtype=numpy.float32),
            xp.linspace(-1, 1, W, dtype=numpy.float32), indexing='ij',
            copy=False
        )

        coords = xp.concatenate(
            [xs[None], ys[None], xp.ones((1, H, W), dtype=numpy.float32)],
            axis=0)
        coords_T = coords.reshape(3, H * W).transpose(1, 0)
        ggrid = ggrid.reshape(B, 2, H * W)
        gtheta = ggrid.dot(coords_T).reshape(B, 2, 3)
        return gtheta,


def spatial_transformer_grid_3d(theta, output_shape, **kwargs):
    """3D Spatial Transformer grid.
    See 2D version : https://docs.chainer.org/en/stable/reference/generated/chainer.functions.spatial_transformer_grid.html
    """
    argument.check_unexpected_kwargs(
        kwargs, use_cudnn="The argument \"use_cudnn\" is not "
        "supported anymore. "
        "Use chainer.using_config('use_cudnn', value) "
        "context where value can be `always`, `never`, or `auto`.")
    argument.assert_kwargs_empty(kwargs)
    return SpatialTransformerGrid3D(output_shape)(theta)