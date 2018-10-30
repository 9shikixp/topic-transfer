import chainer
import chainer.functions as F
from chainer import function, initializers
import chainer.links as L
from chainer import cuda
import numpy as np
from kmaxpooling1d import k_max_pooling_1d
from init import Uniform


class ConvBlock(chainer.Chain):

    def __init__(self, ch_size, out_size):
        initialW = initializers.HeUniform()
        super(ConvBlock, self).__init__()
        with self.init_scope():
            self.conv1 = L.ConvolutionND(
                1, ch_size, ch_size, 3, 1, 1,
                initialW=initialW,
                initial_bias=initializers.Uniform(np.sqrt(6/(ch_size*3))))
            self.bn1 = L.BatchNormalization(ch_size, initial_gamma=Uniform(1.0))
            self.conv2 = L.ConvolutionND(
                1, ch_size, out_size, 3, 1, 1,
                initialW=initialW,
                initial_bias=initializers.Uniform(np.sqrt(6/(ch_size*3))))
            self.bn2 = L.BatchNormalization(out_size, initial_gamma=Uniform(1.0))

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h)

    
class BuildBlock(chainer.ChainList):
    
    def __init__(self, layer, ch_size, out_size):
        super(BuildBlock, self).__init__()
        for i in range(layer - 1):
            self.add_link(ConvBlock(ch_size, ch_size))
        self.add_link(ConvBlock(ch_size, out_size))
    
    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x

    
class MultiVDCNN(chainer.Chain):
    
    def __init__(self, emb_dim, n_out_1, n_out_2, n_out_3, depth=[2,2,2,2]):
        super(MultiVDCNN, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(
                emb_dim, 16, initializers.Normal(), ignore_label=-1)
            self.conv1 = L.ConvolutionND(
                1, 16, 64, 3, 1, 1,
                initialW=initializers.HeUniform(),
                initial_bias=initializers.Uniform(np.sqrt(6/(16*3))))
            self.cb2 = BuildBlock(depth[0], 64, 128)
            self.cb3 = BuildBlock(depth[1], 128, 256)
            self.cb4 = BuildBlock(depth[2], 256, 512)
            self.cb5 = BuildBlock(depth[3], 512, 512)
            self.fc6 = L.Linear(
                4096, 2048,
                initialW=initializers.Uniform(1 / np.sqrt(4096)),
                initial_bias=initializers.Uniform(1 / np.sqrt(4096)))
            self.fc7 = L.Linear(
                2048, 2048,
                initialW=initializers.Uniform(1 / np.sqrt(2048)),
                initial_bias=initializers.Uniform(1 / np.sqrt(2048)))
            self.fc8_1 = L.Linear(
                2048, n_out_1,
                initialW=initializers.Uniform(1 / np.sqrt(2048)),
                initial_bias=initializers.Uniform(1 / np.sqrt(2048)))
            self.fc8_2 = L.Linear(
                2048, n_out_2,
                initialW=initializers.Uniform(1 / np.sqrt(2048)),
                initial_bias=initializers.Uniform(1 / np.sqrt(2048)))
            self.fc8_3 = L.Linear(
                2048, n_out_3,
                initialW=initializers.Uniform(1 / np.sqrt(2048)),
                initial_bias=initializers.Uniform(1 / np.sqrt(2048)))
    
    def __call__(self, x):
        h = self.embed(x)
        h = h.transpose(0,2,1)
        h = self.conv1(h)
        h = self.cb2(h)
        h = F.max_pooling_nd(h, 3, 2, 1, cover_all=False)
        h = self.cb3(h)
        h = F.max_pooling_nd(h, 3, 2, 1, cover_all=False)
        h = self.cb4(h)
        h = F.max_pooling_nd(h, 3, 2, 1, cover_all=False)
        h = self.cb5(h)
        h = k_max_pooling_1d(h, 8)
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        o1 = self.fc8_1(h)
        o2 = self.fc8_2(h)
        o3 = self.fc8_3(h)
        out = F.concat([o1, o2, o3], axis=1)
        return out

    
class VDCNN(chainer.Chain):
    
    def __init__(self, emb_dim, n_out, depth=[2,2,2,2]):
        super(VDCNN, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(
                emb_dim, 16, initializers.Normal(), ignore_label=-1)
            self.conv1 = L.ConvolutionND(
                1, 16, 64, 3, 1, 1,
                initialW=initializers.HeUniform(),
                initial_bias=initializers.Uniform(np.sqrt(6/(16*3))))
            self.cb2 = BuildBlock(depth[0], 64, 128)
            self.cb3 = BuildBlock(depth[1], 128, 256)
            self.cb4 = BuildBlock(depth[2], 256, 512)
            self.cb5 = BuildBlock(depth[3], 512, 512)
            self.fc6 = L.Linear(
                4096, 2048,
                initialW=initializers.Uniform(1 / np.sqrt(4096)),
                initial_bias=initializers.Uniform(1 / np.sqrt(4096)))
            self.fc7 = L.Linear(
                2048, 2048,
                initialW=initializers.Uniform(1 / np.sqrt(2048)),
                initial_bias=initializers.Uniform(1 / np.sqrt(2048)))
            self.fc8 = L.Linear(
                2048, n_out,
                initialW=initializers.Uniform(1 / np.sqrt(2048)),
                initial_bias=initializers.Uniform(1 / np.sqrt(2048)))
    
    def __call__(self, x):
        h = self.embed(x)
        h = h.transpose(0,2,1)
        h = self.conv1(h)
        h = self.cb2(h)
        h = F.max_pooling_nd(h, 3, 2, 1, cover_all=False)
        h = self.cb3(h)
        h = F.max_pooling_nd(h, 3, 2, 1, cover_all=False)
        h = self.cb4(h)
        h = F.max_pooling_nd(h, 3, 2, 1, cover_all=False)
        h = self.cb5(h)
        h = k_max_pooling_1d(h, 8)
        h = F.relu(self.fc6(h))
        h = F.relu(self.fc7(h))
        h = self.fc8(h)
        return h