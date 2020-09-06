from mxnet import gluon
from mxnet.gluon import HybridBlock

from ceecnet.nn.layers.scale import *
from ceecnet.nn.layers.conv2Dnormed import *


"""
For combining layers with Fusion (i.e. relative attention), see ../units/ceecnet.py
"""


class combine_layers(HybridBlock):
    def __init__(self,_nfilters,  _norm_type = 'BatchNorm', norm_groups=None, **kwards):
        HybridBlock.__init__(self,**kwards)
        
        with self.name_scope():

            # This performs convolution, no BatchNormalization. No need for bias. 
            self.up = UpSample(_nfilters, _norm_type = _norm_type, norm_groups=norm_groups) 

            self.conv_normed = Conv2DNormed(channels = _nfilters, 
                                            kernel_size=(1,1),
                                            padding=(0,0), 
                                            _norm_type=_norm_type,
                                            norm_groups=norm_groups)

        
            
        
    def hybrid_forward(self,F,_layer_lo, _layer_hi):
        
        up = self.up(_layer_lo)
        up = F.relu(up)
        x = F.concat(up,_layer_hi, dim=1)
        x = self.conv_normed(x)
        
        return x



