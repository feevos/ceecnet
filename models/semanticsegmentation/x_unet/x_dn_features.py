from mxnet import gluon
from mxnet.gluon import HybridBlock

from ceecnet.nn.layers.conv2Dnormed import *
from ceecnet.nn.layers.attention import * 
from ceecnet.nn.pooling.psp_pooling import *


from ceecnet.nn.layers.scale import *
from ceecnet.nn.layers.combine import *

# CEEC units 
from ceecnet.nn.units.ceecnet import *

# FracTALResUnit
from ceecnet.nn.units.fractal_resnet import *  

"""
if upFuse == True, then instead of concatenation of the encoder features with the decoder features, the algorithm performs Fusion with 
relative attention.  
"""


class X_dn_features(HybridBlock):
    def __init__(self, nfilters_init, depth, widths=[1], psp_depth=4, verbose=True, norm_type='BatchNorm', norm_groups=None, nheads_start=8,  model='CEECNetV1', upFuse=False, ftdepth=5, **kwards):
        super().__init__(**kwards)
        

        self.depth = depth


        if len(widths) == 1 and depth != 1:
            widths = widths * depth
        else:
            assert depth == len(widths), ValueError("depth and length of widths must match, aborting ...")

        with self.name_scope():
            
            self.conv_first = Conv2DNormed(nfilters_init,kernel_size=(1,1), _norm_type = norm_type, norm_groups=norm_groups)
            
            # List of convolutions and pooling operators 
            self.convs_dn = gluon.nn.HybridSequential()
            self.pools = gluon.nn.HybridSequential()


            for idx in range(depth):
                nheads = nheads_start * 2**idx #
                nfilters = nfilters_init * 2 **idx
                if verbose:
                    print ("depth:= {0}, nfilters: {1}, nheads::{2}, widths::{3}".format(idx,nfilters,nheads,widths[idx]))
                tnet = gluon.nn.HybridSequential()
                for _ in range(widths[idx]):
                    if model == 'CEECNetV1':
                        tnet.add(CEEC_unit_v1(nfilters=nfilters, nheads = nheads, ngroups = nheads , norm_type = norm_type, norm_groups=norm_groups,ftdepth=ftdepth))
                    elif model == 'CEECNetV2':
                        tnet.add(CEEC_unit_v2(nfilters=nfilters, nheads = nheads, ngroups = nheads , norm_type = norm_type, norm_groups=norm_groups,ftdepth=ftdepth))
                    elif model == 'FracTALResNet':
                        tnet.add(FracTALResNet_unit(nfilters=nfilters, nheads = nheads, ngroups = nheads , norm_type = norm_type, norm_groups=norm_groups,ftdepth=ftdepth))
                    else:
                        raise ValueError("I don't know requested model, available options: CEECNetV1, CEECNetV2, FracTALResNet - Given model::{}, aborting ...".format(model))
                self.convs_dn.add(tnet)

                if idx < depth-1:
                    self.pools.add(DownSample(nfilters, _norm_type=norm_type, norm_groups=norm_groups)) 
            # Middle pooling operator 
            self.middle = PSP_Pooling(nfilters,depth=psp_depth, _norm_type=norm_type,norm_groups=norm_groups)
                               
            
            self.convs_up = gluon.nn.HybridSequential() # 1 argument
            self.UpCombs = gluon.nn.HybridSequential() # 2 arguments
            for idx in range(depth-1,0,-1):
                nheads = nheads_start * 2**idx 
                nfilters = nfilters_init * 2 **(idx-1)
                if verbose:
                    print ("depth:= {0}, nfilters: {1}, nheads::{2}, widths::{3}".format(2*depth-idx-1,nfilters,nheads,widths[idx]))
                
                tnet = gluon.nn.HybridSequential()
                for _ in range(widths[idx]):
                    if model == 'CEECNetV1':
                        tnet.add(CEEC_unit_v1(nfilters=nfilters, nheads = nheads, ngroups = nheads , norm_type = norm_type, norm_groups=norm_groups,ftdepth=ftdepth))
                    elif model == 'CEECNetV2':
                        tnet.add(CEEC_unit_v2(nfilters=nfilters, nheads = nheads, ngroups = nheads , norm_type = norm_type, norm_groups=norm_groups,ftdepth=ftdepth))
                    elif model == 'FracTALResNet':
                        tnet.add(FracTALResNet_unit(nfilters=nfilters, nheads = nheads, ngroups = nheads , norm_type = norm_type, norm_groups=norm_groups,ftdepth=ftdepth))
                    else:
                        raise ValueError("I don't know requested model, available options: CEECNetV1, CEECNetV2, FracTALResNet - Given model::{}, aborting ...".format(model))
                self.convs_up.add(tnet)
                
                if upFuse==True:
                    self.UpCombs.add(combine_layers_wthFusion(nfilters=nfilters, nheads=nheads, _norm_type=norm_type,norm_groups=norm_groups,ftdepth=ftdepth))
                else:
                    self.UpCombs.add(combine_layers(nfilters, _norm_type=norm_type,norm_groups=norm_groups))
                
    def hybrid_forward(self, F, input):

        conv1_first = self.conv_first(input)
 
        
        # ******** Going down ***************
        fusions   = []

        # Workaround of a mxnet bug 
        # https://github.com/apache/incubator-mxnet/issues/16736
        pools = F.identity(conv1_first) 

        for idx in range(self.depth):
            conv1 = self.convs_dn[idx](pools)
            if idx < self.depth-1:
                # Evaluate fusions 
                conv1 = F.identity(conv1)
                fusions = fusions + [conv1]
                # Evaluate pools 
                pools =  self.pools[idx](conv1)

        # Middle psppooling
        middle =  self.middle(conv1)
        # Activation of middle layer
        middle = F.relu(middle)
        fusions   = fusions + [middle] 

        # ******* Coming up ****************
        convs_up = middle
        for idx in range(self.depth-1):
            convs_up = self.UpCombs[idx](convs_up, fusions[-idx-2])
            convs_up = self.convs_up[idx](convs_up)
            
        return convs_up, conv1_first
    

