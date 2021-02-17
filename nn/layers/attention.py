from mxnet import gluon
from mxnet.gluon import HybridBlock
from ceecnet.nn.layers.conv2Dnormed import *
from ceecnet.nn.layers.ftnmt import * 


        
class RelFTAttention2D(HybridBlock):
    def __init__(self, nkeys, kernel_size=3, padding=1,nheads=1, norm = 'BatchNorm', norm_groups=None,ftdepth=5,**kwards):
        super().__init__(**kwards)

        with self.name_scope():

            self.query  = Conv2DNormed(channels=nkeys,kernel_size= kernel_size, padding = padding, _norm_type= norm, norm_groups=norm_groups, groups=nheads)
            self.key    = Conv2DNormed(channels=nkeys,kernel_size= kernel_size, padding = padding, _norm_type= norm, norm_groups=norm_groups, groups=nheads)
            self.value  = Conv2DNormed(channels=nkeys,kernel_size= kernel_size, padding = padding, _norm_type= norm, norm_groups=norm_groups, groups=nheads)

            
            self.metric_channel = FTanimoto(depth=ftdepth, axis=[2,3])
            self.metric_space = FTanimoto(depth=ftdepth, axis=1)
             
            self.norm = get_norm(name=norm, axis=1, norm_groups= norm_groups)
            
    def hybrid_forward(self, F, input1, input2, input3):

        # These should work with ReLU as well 
        q = F.sigmoid(self.query(input1))
        k = F.sigmoid(self.key(input2))# B,C,H,W 
        v = F.sigmoid(self.value(input3)) # B,C,H,W

        att_spat =  self.metric_space(q,k) # B,1,H,W 
        v_spat  =  F.broadcast_mul(att_spat, v) # emphasize spatial features

        att_chan =  self.metric_channel(q,k) # B,C,1,1
        v_chan   =  F.broadcast_mul(att_chan, v) # emphasize spatial features


        v_cspat =   0.5*F.broadcast_add(v_chan, v_spat) # emphasize spatial features
        v_cspat = self.norm(v_cspat)

        return v_cspat



class FTAttention2D(HybridBlock):
    def __init__(self, nkeys, kernel_size=3, padding=1, nheads=1, norm = 'BatchNorm', norm_groups=None,ftdepth=5,**kwards):
        super().__init__(**kwards)
        
        with self.name_scope():
            self. att = RelFTAttention2D(nkeys=nkeys,kernel_size=kernel_size, padding=padding, nheads=nheads, norm = norm, norm_groups=norm_groups, ftdepth=ftdepth,**kwards)


    def hybrid_forward(self, F, input):
        return self.att(input,input,input)










