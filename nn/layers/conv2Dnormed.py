import mxnet as mx
from mxnet import gluon
from mxnet.gluon import HybridBlock
from ceecnet.utils.get_norm import * 


class Conv2DNormed(HybridBlock):
    """
        Convenience wrapper layer for 2D convolution followed by a normalization layer 
        All other keywords are the same as gluon.nn.Conv2D 
    """

    def __init__(self,  channels, kernel_size, strides=(1, 1), 
                 padding=(0, 0), dilation=(1, 1),   activation=None, 
                 weight_initializer=None,  in_channels=0, _norm_type = 'BatchNorm', norm_groups=None, axis =1 , groups=1, **kwards):
        super().__init__(**kwards)

        with self.name_scope():
            self.conv2d = gluon.nn.Conv2D(channels, kernel_size = kernel_size, 
                                          strides= strides, 
                                          padding=padding,
                                          dilation= dilation, 
                                          activation=activation, 
                                          use_bias=False, 
                                          weight_initializer = weight_initializer, 
                                          groups=groups,
                                          in_channels=0)

            self.norm_layer = get_norm(_norm_type, axis=axis, norm_groups= norm_groups)

    def hybrid_forward(self,F,_x):

        x = self.conv2d(_x)
        x = self.norm_layer(x)

        return x

