from mxnet import gluon
from mxnet.gluon import HybridBlock
from ceecnet.nn.layers.conv2Dnormed import *
from ceecnet.utils.get_norm import *
from ceecnet.nn.layers.attention import *


class ResizeLayer(HybridBlock):
    """
    Applies bilinear up/down sampling in spatial dims and changes number of filters as well 
    """
    def __init__(self, nfilters, height, width,   _norm_type = 'BatchNorm', norm_groups=None, **kwards):
        super().__init__(**kwards)

        self.height=height
        self.width = width

        with self.name_scope():

            self.conv2d = Conv2DNormed(channels=nfilters,kernel_size=3,padding=1, _norm_type=_norm_type, norm_groups = norm_groups, **kwards)


    def hybrid_forward(self, F, input):
        out = F.contrib.BilinearResize2D(input,height=self.height,width=self.width)
        out = self.conv2d(out)

        return out

class ExpandLayer(HybridBlock):
    def __init__(self,nfilters, _norm_type = 'BatchNorm', norm_groups=None, ngroups=1,**kwards):
        super().__init__(**kwards)


        with self.name_scope():
            self.conv1 = Conv2DNormed(channels=nfilters,kernel_size=3,padding=1,groups=ngroups, _norm_type=_norm_type, norm_groups = norm_groups, **kwards)
            self.conv2 = Conv2DNormed(channels=nfilters,kernel_size=3,padding=1,groups=ngroups,_norm_type=_norm_type, norm_groups = norm_groups,**kwards) 

    def hybrid_forward(self, F, input):

        out = F.contrib.BilinearResize2D(input,scale_height=2.,scale_width=2.)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)

        return out

class ExpandNCombine(HybridBlock):
    def __init__(self,nfilters, _norm_type = 'BatchNorm', norm_groups=None,ngroups=1,**kwards):
        super().__init__(**kwards)

        with self.name_scope():
            self.conv1 = Conv2DNormed(channels=nfilters,kernel_size=3,padding=1,groups=ngroups,_norm_type=_norm_type, norm_groups = norm_groups,**kwards)
            self.conv2 = Conv2DNormed(channels=nfilters,kernel_size=3,padding=1,groups=ngroups,_norm_type=_norm_type, norm_groups = norm_groups,**kwards)  

    def hybrid_forward(self, F, input1, input2):

        out = F.contrib.BilinearResize2D(input1,scale_height=2.,scale_width=2.)
        out = self.conv1(out)
        out = F.relu(out)
        out2 = self.conv2(F.concat(out,input2,dim=1))
        out2 = F.relu(out2)

        return out2



class CEEC_unit_v1(HybridBlock):
    def __init__(self, nfilters, nheads= 1, ngroups=1, norm_type='BatchNorm', norm_groups=None, ftdepth=5, **kwards):
        super().__init__(**kwards)


        with self.name_scope():
            nfilters_init = nfilters//2
            self.conv_init_1 = Conv2DNormed(channels=nfilters_init, kernel_size=3,padding=1,strides=1, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups, **kwards)
            self.compr11 = Conv2DNormed(channels=nfilters_init*2, kernel_size=3,padding=1,strides=2, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups, **kwards)
            self.compr12 = Conv2DNormed(channels=nfilters_init*2, kernel_size=3,padding=1,strides=1, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups,**kwards)
            self.expand1 = ExpandNCombine(nfilters_init,_norm_type = norm_type, norm_groups=norm_groups,ngroups=ngroups) 

            # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

            self.conv_init_2 = Conv2DNormed(channels=nfilters_init, kernel_size=3,padding=1,strides=1, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups, **kwards)#half size 

            self.expand2 = ExpandLayer(nfilters_init//2 ,_norm_type = norm_type, norm_groups=norm_groups,ngroups=ngroups )
            self.compr21 = Conv2DNormed(channels=nfilters_init, kernel_size=3,padding=1,strides=2, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups,**kwards)
            self.compr22 = Conv2DNormed(channels=nfilters_init, kernel_size=3,padding=1,strides=1, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups,**kwards)

            # Will join with master input with concatenation  -- IMPORTANT: ngroups = 1 !!!!
            self.collect = Conv2DNormed(channels=nfilters, kernel_size=3,padding=1,strides=1, groups=1, _norm_type=norm_type, norm_groups=norm_groups,**kwards)


            self.att  = FTAttention2D(nkeys=nfilters,nheads=nheads,norm=norm_type, norm_groups = norm_groups,ftdepth=ftdepth)
            self.ratt122 = RelFTAttention2D(nkeys=nfilters_init, nheads=nheads,norm=norm_type, norm_groups = norm_groups,ftdepth=ftdepth)
            self.ratt211 = RelFTAttention2D(nkeys=nfilters_init, nheads=nheads,norm=norm_type, norm_groups = norm_groups,ftdepth=ftdepth)


            self.gamma1  = self.params.get('gamma1', shape=(1,), init=mx.init.Zero())
            self.gamma2  = self.params.get('gamma2', shape=(1,), init=mx.init.Zero())
            self.gamma3  = self.params.get('gamma3', shape=(1,), init=mx.init.Zero())


    def hybrid_forward(self, F, input, gamma1, gamma2, gamma3):
        
        # =========== UNet branch ===========
        out10 = self.conv_init_1(input)
        out1 = self.compr11(out10)
        out1 = F.relu(out1)
        out1 = self.compr12(out1)
        out1 = F.relu(out1)
        out1 = self.expand1(out1,out10)
        out1 = F.relu(out1)


        # =========== \capNet branch ===========
        input = F.identity(input) # Solves a mxnet bug

        out20 = self.conv_init_2(input)
        out2 = self.expand2(out20)
        out2 = F.relu(out2)
        out2 = self.compr21(out2)
        out2 = F.relu(out2)
        out2 = self.compr22(F.concat(out2,out20,dim=1))
        out2 = F.relu(out2)

        att  = F.broadcast_mul(gamma1,self.att(input))
        ratt122 = F.broadcast_mul(gamma2,self.ratt122(out1,out2,out2))
        ratt211 = F.broadcast_mul(gamma3,self.ratt211(out2,out1,out1))

        ones1 = F.ones_like(out10)
        ones2 = F.ones_like(input)

        # Enhanced output of 1, based on memory of 2
        out122 = F.broadcast_mul(out1,ones1 + ratt122)
        # Enhanced output of 2, based on memory of 1        
        out211 = F.broadcast_mul(out2,ones1 + ratt211)

        out12 = F.relu(self.collect(F.concat(out122,out211,dim=1)))

        # Emphasize residual output from memory on input 
        out_res = F.broadcast_mul(input + out12, ones2 + att)
        return out_res






# ======= Definitions for CEEC unit v2  (replace concatenations with Fusion =========================
# -------------------------------------- helper functions -------------------------------------------

class Fusion(HybridBlock):
    def __init__(self,nfilters, kernel_size=3, padding=1,nheads=1, norm = 'BatchNorm', norm_groups=None, ftdepth=5,**kwards):
        super().__init__(**kwards)


        with self.name_scope():
            self.fuse = Conv2DNormed(nfilters,kernel_size= kernel_size, padding = padding, _norm_type= norm, norm_groups=norm_groups, groups=nheads,**kwards)
            # Or shall I use the same? 
            self.relatt12 = RelFTAttention2D(nkeys=nfilters, kernel_size=kernel_size, padding=padding, nheads=nheads, norm =norm, norm_groups=norm_groups,ftdepth=ftdepth,**kwards)
            self.relatt21 = RelFTAttention2D(nkeys=nfilters, kernel_size=kernel_size, padding=padding, nheads=nheads, norm =norm, norm_groups=norm_groups,ftdepth=ftdepth,**kwards)


            self.gamma1  = self.params.get('gamma1', shape=(1,), init=mx.init.Zero())
            self.gamma2  = self.params.get('gamma2', shape=(1,), init=mx.init.Zero())


    def hybrid_forward(self, F, input_t1, input_t2, gamma1, gamma2):
        # These inputs must have the same dimensionality , t1, t2 
        relatt12 = F.broadcast_mul(gamma1,self.relatt12(input_t1,input_t2,input_t2))
        relatt21 = F.broadcast_mul(gamma2,self.relatt21(input_t2,input_t1,input_t1))

        ones = F.ones_like(input_t1)

        # Enhanced output of 1, based on memory of 2
        out12 = F.broadcast_mul(input_t1,ones + relatt12)
        # Enhanced output of 2, based on memory of 1        
        out21 = F.broadcast_mul(input_t2,ones + relatt21)


        fuse = self.fuse(F.concat(out12, out21,dim=1))
        fuse = F.relu(fuse)

        return fuse



class CATFusion(HybridBlock):
    """
    Alternative to concatenation followed by normed convolution: improves performance. 
    """
    def __init__(self,nfilters_out, nfilters_in, kernel_size=3, padding=1,nheads=1, norm = 'BatchNorm', norm_groups=None, ftdepth=5,**kwards):
        super().__init__(**kwards)


        with self.name_scope():
            self.fuse = Conv2DNormed(nfilters_out,kernel_size= kernel_size, padding = padding, _norm_type= norm, norm_groups=norm_groups, groups=nheads,**kwards)
            # Or shall I use the same? 
            self.relatt12 = RelFTAttention2D(nkeys=nfilters_in, kernel_size=kernel_size, padding=padding, nheads=nheads, norm =norm, norm_groups=norm_groups,ftdepth=ftdepth,**kwards)
            self.relatt21 = RelFTAttention2D(nkeys=nfilters_in, kernel_size=kernel_size, padding=padding, nheads=nheads, norm =norm, norm_groups=norm_groups,ftdepth=ftdepth,**kwards)


            self.gamma1  = self.params.get('gamma1', shape=(1,), init=mx.init.Zero())
            self.gamma2  = self.params.get('gamma2', shape=(1,), init=mx.init.Zero())



    def hybrid_forward(self, F, input_t1, input_t2, gamma1, gamma2):
        # These inputs must have the same dimensionality , t1, t2 
        relatt12 = F.broadcast_mul(gamma1,self.relatt12(input_t1,input_t2,input_t2))
        relatt21 = F.broadcast_mul(gamma2,self.relatt21(input_t2,input_t1,input_t1))

        ones = F.ones_like(input_t1)

        # Enhanced output of 1, based on memory of 2
        out12 = F.broadcast_mul(input_t1,ones + relatt12)
        # Enhanced output of 2, based on memory of 1        
        out21 = F.broadcast_mul(input_t2,ones + relatt21)


        fuse = self.fuse(F.concat(out12, out21,dim=1))
        fuse = F.relu(fuse)

        return fuse




class combine_layers_wthFusion(HybridBlock):
    def __init__(self,nfilters, nheads=1,  _norm_type = 'BatchNorm', norm_groups=None,ftdepth=5, **kwards):
        HybridBlock.__init__(self,**kwards)

        with self.name_scope():

            self.conv1 = Conv2DNormed(channels=nfilters,kernel_size=3,padding=1, groups=nheads, _norm_type=_norm_type, norm_groups = norm_groups, **kwards)# restore help 
            self.conv3 = Fusion(nfilters=nfilters, kernel_size=3, padding=1, nheads=nheads, norm=_norm_type, norm_groups = norm_groups, ftdepth=ftdepth,**kwards) # process 

    def hybrid_forward(self,F,_layer_lo, _layer_hi):

        up = F.contrib.BilinearResize2D(_layer_lo,scale_height=2.,scale_width=2.)
        up = self.conv1(up)
        up = F.relu(up)
        x = self.conv3(up,_layer_hi)

        return x


class ExpandNCombine_V3(HybridBlock):
    def __init__(self,nfilters, _norm_type = 'BatchNorm', norm_groups=None,ngroups=1,ftdepth=5,**kwards):
        super().__init__(**kwards)


        with self.name_scope():
            self.conv1 = Conv2DNormed(channels=nfilters,kernel_size=3,padding=1,groups=ngroups,_norm_type=_norm_type, norm_groups = norm_groups,**kwards)# restore help 
            self.conv2 = Conv2DNormed(channels=nfilters,kernel_size=3,padding=1,groups=ngroups,_norm_type=_norm_type, norm_groups = norm_groups,**kwards)# restore help 
            self.conv3 = Fusion(nfilters=nfilters,kernel_size=3,padding=1,nheads=ngroups,norm=_norm_type, norm_groups = norm_groups,ftdepth=ftdepth,**kwards) # process 

    def hybrid_forward(self, F, input1, input2):

        out = F.contrib.BilinearResize2D(input1,scale_height=2.,scale_width=2.)
        out = self.conv1(out)
        out1 = F.relu(out)

        out2 = self.conv2(input2)
        out2 = F.relu(out2)

        outf = self.conv3(out1,out2)
        outf = F.relu(outf)

        return outf




# -------------------------------------------------------------------------------------------------------------------

class CEEC_unit_v2(HybridBlock):
    def __init__(self, nfilters, nheads= 1, ngroups=1, norm_type='BatchNorm', norm_groups=None, ftdepth=5, **kwards):
        super().__init__(**kwards)


        with self.name_scope():
            nfilters_init = nfilters//2
            self.conv_init_1 = Conv2DNormed(channels=nfilters_init, kernel_size=3,padding=1,strides=1, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups, **kwards)#half size
            self.compr11 = Conv2DNormed(channels=nfilters_init*2, kernel_size=3,padding=1,strides=2, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups, **kwards)#half size 
            self.compr12 = Conv2DNormed(channels=nfilters_init*2, kernel_size=3,padding=1,strides=1, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups,**kwards)# process
            self.expand1 = ExpandNCombine_V3(nfilters_init,_norm_type = norm_type, norm_groups=norm_groups,ngroups=ngroups,ftdepth=ftdepth) # restore original size + process


            self.conv_init_2 = Conv2DNormed(channels=nfilters_init, kernel_size=3,padding=1,strides=1, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups, **kwards)#half size
            self.expand2 = ExpandLayer(nfilters_init//2 ,_norm_type = norm_type, norm_groups=norm_groups,ngroups=ngroups )
            self.compr21 = Conv2DNormed(channels=nfilters_init, kernel_size=3,padding=1,strides=2, groups=ngroups, _norm_type=norm_type, norm_groups=norm_groups,**kwards)
            self.compr22 = Fusion(nfilters=nfilters_init, kernel_size=3,padding=1, nheads=ngroups, norm=norm_type, norm_groups=norm_groups,ftdepth=ftdepth,**kwards)

            self.collect = CATFusion(nfilters_out=nfilters, nfilters_in=nfilters_init, kernel_size=3,padding=1,nheads=1, norm=norm_type, norm_groups=norm_groups,ftdepth=ftdepth,**kwards)

            self.att  = FTAttention2D(nkeys=nfilters,nheads=nheads,norm=norm_type, norm_groups = norm_groups, ftdepth=ftdepth)
            self.ratt122 = RelFTAttention2D(nkeys=nfilters_init, nheads=nheads,norm=norm_type, norm_groups = norm_groups, ftdepth=ftdepth)
            self.ratt211 = RelFTAttention2D(nkeys=nfilters_init, nheads=nheads,norm=norm_type, norm_groups = norm_groups, ftdepth=ftdepth)


            self.gamma1  = self.params.get('gamma1', shape=(1,), init=mx.init.Zero())
            self.gamma2  = self.params.get('gamma2', shape=(1,), init=mx.init.Zero())
            self.gamma3  = self.params.get('gamma3', shape=(1,), init=mx.init.Zero())

    def hybrid_forward(self, F, input, gamma1, gamma2, gamma3):

        # =========== UNet branch ===========
        out10 = self.conv_init_1(input)
        out1 = self.compr11(out10)
        out1 = F.relu(out1)
        #print (out1.shape)
        out1 = self.compr12(out1)
        out1 = F.relu(out1)
        #print (out1.shape)
        out1 = self.expand1(out1,out10)
        out1 = F.relu(out1)


        # =========== \capNet branch ===========
        input = F.identity(input) # Solves a mxnet bug

        out20 = self.conv_init_2(input)
        out2 = self.expand2(out20)
        out2 = F.relu(out2)
        out2 = self.compr21(out2)
        out2 = F.relu(out2)
        out2 = self.compr22(out2,out20) 



        input = F.identity(input) # Solves a mxnet bug

        att  = F.broadcast_mul(gamma1,self.att(input))
        ratt122 = F.broadcast_mul(gamma2,self.ratt122(out1,out2,out2))
        ratt211 = F.broadcast_mul(gamma3,self.ratt211(out2,out1,out1))

        ones1 = F.ones_like(out10)
        ones2 = F.ones_like(input)

        # Enhanced output of 1, based on memory of 2
        out122 = F.broadcast_mul(out1,ones1 + ratt122)
        # Enhanced output of 2, based on memory of 1        
        out211 = F.broadcast_mul(out2,ones1 + ratt211)


        out12 = self.collect(out122,out211) # includes relu, it's for fusion

        out_res = F.broadcast_mul(input + out12, ones2 + att)
        return out_res


