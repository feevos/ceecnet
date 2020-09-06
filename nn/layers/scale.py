from mxnet  import gluon
from mxnet.gluon import HybridBlock

from ceecnet.nn.layers.conv2Dnormed import * 
from ceecnet.utils.get_norm import * 

class DownSample(HybridBlock):
    def __init__(self, nfilters, factor=2,  _norm_type='BatchNorm', norm_groups=None, **kwargs): 
        super().__init__(**kwargs)
        
        
        # Double the size of filters, since you downscale by 2. 
        self.factor = factor 
        self.nfilters = nfilters * self.factor

        self.kernel_size = (3,3) 
        self.strides = (factor,factor)
        self.pad = (1,1)

        with self.name_scope():
            self.convdn = Conv2DNormed(self.nfilters,
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    padding=self.pad,
                    _norm_type = _norm_type, 
                    norm_groups=norm_groups)
 
    
    def hybrid_forward(self,F,_xl):
        
        x = self.convdn(_xl)

        return x 


class UpSample(HybridBlock):
    def __init__(self,nfilters, factor = 2,  _norm_type='BatchNorm', norm_groups=None, **kwards):
        HybridBlock.__init__(self,**kwards)
        
        
        self.factor = factor
        self.nfilters = nfilters // self.factor
        
        with self.name_scope():
            self.convup_normed = Conv2DNormed(self.nfilters,
                                              kernel_size = (1,1),
                                              _norm_type = _norm_type, 
                                              norm_groups=norm_groups)
    
    def hybrid_forward(self,F,_xl):
        x = F.UpSampling(_xl, scale=self.factor, sample_type='nearest')
        x = self.convup_normed(x)
        
        return x

