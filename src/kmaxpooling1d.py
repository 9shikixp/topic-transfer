import numpy as np
import chainer
import chainer.functions as F
from chainer import function, initializers
import chainer.links as L
from chainer import cuda

class KMaxPooling1D(function.Function):
    
    def __init__(self, ndim, k):
        if ndim <= 0:
            raise ValueError(
                'pooling operation requires at least one spatial dimension.')

        self.ndim = ndim
        self.k = k

        self._used_cudnn = False

    def forward_gpu(self, x):
        if chainer.should_use_cudnn('>=auto') and 2 <= self.ndim <= 3:
            # With cuDNN v3 or greater, use cuDNN implementation for inputs
            # with spatial dimensions of two or more.
            return super(KMaxPooling1D, self).forward_gpu(x)

        self.retain_inputs(())
        self._in_shape = x[0].shape
        self._in_dtype = x[0].dtype

        n, c = x[0].shape[:2]
        dims = x[0].shape[2:]
        ys = (self.k,)

        y_shape = (n, c) + ys
        y = cuda.cupy.empty(y_shape, dtype=x[0].dtype)
        self.indexes = cuda.cupy.empty(y_shape, dtype=np.int32)
        
        cuda.elementwise('raw T in, int32 d_0, int32 out_0',
                         'T out, S indexes', 
                         '''int c0 = i / (out_0);
                            int out_x_0 = i % out_0;
                            int in_x0_0 = 0;
                            int in_x1_0 = d_0;
                            int argmax_0[''' + str(self.k) + '''];
                            T maxval[''' + str(self.k) + '''];
                            for (int a = 0; a < out_0; ++a) {
                              maxval[a] = (T)-(1.0/0.0);
                              argmax_0[a] = -1;
                            }
                            for (int a = 0; a < out_0; ++a) {
                              for (int x_0 = in_x0_0; x_0 < in_x1_0; ++x_0) {
                                int offset_0 = 1 * (x_0 + d_0 * c0);
                                int found = 0;
                                for (int b = 0; b < a; ++b) {
                                  if (argmax_0[b] == x_0) {
                                    found = 1;
                                    break;
                                  }
                                }
                                if (found) {
                                  continue;
                                }
                                T v = in[offset_0];
                                if (maxval[a] < v) {
                                  maxval[a] = v;
                                  argmax_0[a] = x_0;
                                }
                              }
                            }
                            out = maxval[i % out_0];
                            int argmax_k_0 = argmax_0[i % out_0];
                            indexes = argmax_k_0;
                         ''',
                         'k_max_pool_1d_fwd')(
            x[0].reduced_view(),
            *(dims + ys +
              (y, self.indexes)))
                
        return y,


    def backward_gpu(self, x, gy):
        if self._used_cudnn:
            return super(KMaxPooling1D, self).backward_gpu(x, gy)

        n, c = self._in_shape[:2]
        dims = self._in_shape[2:]
        ys = gy[0].shape[2:]
        gx = cuda.cupy.empty(self._in_shape, self._in_dtype)

        ndim = self.ndim
        cuda.elementwise('raw T gy, raw S indexes, int32 d_0, int32 out_0',
                         'T gx',
                         '''operation:
                            int c0  = i / (d_0);
                            int x_0 = i % d_0;
                            int out_x0_0 = 0;
                            int out_x1_0 = out_0;
                            T val = 0;
                            for (int out_x_0 = out_x0_0; out_x_0 < out_x1_0; ++out_x_0) {
                              int offset_0 = 1 * (out_x_0 + out_0 * c0);
                              int kx = x_0;
                              if (indexes[offset_0] == kx) {
                                val = val + gy[offset_0];
                              }
                            }
                            gx = val;
                         ''',
                         'k_max_pool_1d_bwd')(
            gy[0].reduced_view(), self.indexes.reduced_view(),
            *(dims + ys + (gx,)))

        return gx,

def k_max_pooling_1d(x, k):
    ndim = len(x.shape[2:])
    return KMaxPooling1D(ndim, k)(x)
