import numpy

from chainer import cuda
from chainer import initializer


# Original code forked from MIT licensed keras project
# https://github.com/fchollet/keras/blob/master/keras/initializations.py

class Uniform(initializer.Initializer):

    """Initializes array with a scaled uniform distribution.
    Each element of the array is initialized by the value drawn
    independently from uniform distribution :math:`[0, scale]`.
    Attributes:
        scale (float): A constant that determines the
            scale of the uniform distribution.
        dtype: Data type specifier.
    """

    def __init__(self, scale=0.05, dtype=None):
        self.scale = scale
        super(Uniform, self).__init__(dtype)

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype
        xp = cuda.get_array_module(array)
        array[...] = xp.random.uniform(
            low=0.0, high=self.scale, size=array.shape)
