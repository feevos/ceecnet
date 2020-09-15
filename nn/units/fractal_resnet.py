from mxnet import gluon
from mxnet.gluon import HybridBlock
from ceecnet.nn.layers.conv2Dnormed import *
from ceecnet.utils.get_norm import *
from ceecnet.nn.layers.attention import *

class ResNet_v2_block(HybridBlock):
    """
    ResNet v2 building block. It is built upon the assumption of ODD kernel 
    """
    def __init__(self, _nfilters,_kernel_size=(3,3),_dilation_rate=(1,1), 
                 _norm_type='BatchNorm', norm_groups=None, ngroups=1, **kwards):
        super().__init__(**kwards)

        self.nfilters = _nfilters
        self.kernel_size = _kernel_size
        self.dilation_rate = _dilation_rate


        with self.name_scope():

            # Ensures padding = 'SAME' for ODD kernel selection 
            p0 = self.dilation_rate[0] * (self.kernel_size[0] - 1)/2 
            p1 = self.dilation_rate[1] * (self.kernel_size[1] - 1)/2 
            p = (int(p0),int(p1))


            self.BN1 = get_norm(_norm_type, norm_groups=norm_groups )
            self.conv1 = gluon.nn.Conv2D(self.nfilters,kernel_size = self.kernel_size,padding=p,dilation=self.dilation_rate,use_bias=False,groups=ngroups)
            self.BN2 = get_norm(_norm_type, norm_groups= norm_groups)
            self.conv2 = gluon.nn.Conv2D(self.nfilters,kernel_size = self.kernel_size,padding=p,dilation=self.dilation_rate,use_bias=True, groups=ngroups)


    def hybrid_forward(self,F,_input_layer):
 
        x = self.BN1(_input_layer)
        x = F.relu(x)
        x = self.conv1(x)

        x = self.BN2(x)
        x = F.relu(x)
        x = self.conv2(x)

        return x 

class FracTALResNet_unit(HybridBlock):
    def __init__(self, nfilters, ngroups=1, nheads=1, kernel_size=(3,3), dilation_rate=(1,1), norm_type = 'BatchNorm', norm_groups=None, ftdepth=5,**kwards):
        super().__init__()

        with self.name_scope():
            self.block1 = ResNet_v2_block(nfilters,kernel_size,dilation_rate,_norm_type = norm_type, norm_groups=norm_groups, ngroups=ngroups)
            self.attn1 = FTAttention2D(nfilters=nfilters, nheads=nheads, kernel_size=kernel_size, norm = norm_type, norm_groups = norm_groups,ftdepth=ftdepth)
            self.attn2 = FTAttention2D(nfilters=nfilters, nheads=nheads, kernel_size=kernel_size, norm = norm_type, norm_groups = norm_groups,ftdepth=ftdepth)

            self.gamma1  = self.params.get('gamma1', shape=(1,), init=mx.init.Zero())
            self.gamma2  = self.params.get('gamma2', shape=(1,), init=mx.init.Zero())

    def hybrid_forward(self, F, input, gamma1,gamma2):
        out1 = self.block1(input)
        att1 = self.attn1(out1)
        att1 = F.broadcast_mul(gamma1,att1)
        out1  = F.broadcast_mul(out1 , F.ones_like(out1) + att1) 


        att2 = self.attn2(input)
        att2 = F.broadcast_mul(gamma2,att2)
        
        out  = F.broadcast_mul((input + out1) , F.ones_like(out1) + att2) 
        return out 
