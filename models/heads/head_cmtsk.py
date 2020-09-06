from mxnet import gluon
from mxnet.gluon import HybridBlock 


from ceecnet.nn.activations.sigmoid_crisp import *
from ceecnet.nn.pooling.psp_pooling import *
from ceecnet.nn.layers.conv2Dnormed import *

# Helper classification head, for a single layer output 
class HeadSingle(HybridBlock):
    def __init__(self, nfilters,  NClasses, depth=2, norm_type='BatchNorm',norm_groups=None, **kwargs):
        super().__init__(**kwargs)


        with self.name_scope():
            self.logits = gluon.nn.HybridSequential()
            for _ in range(depth):
                self.logits.add( Conv2DNormed(channels = nfilters,kernel_size = (3,3),padding=(1,1), _norm_type=norm_type, norm_groups=norm_groups))
                self.logits.add( gluon.nn.Activation('relu'))
            self.logits.add( gluon.nn.Conv2D(NClasses,kernel_size=1,padding=0))

    def hybrid_forward(self,F,input):
        return self.logits(input)



class Head_CMTSK_BC(HybridBlock):
    # BC: Balanced (features) Crisp (boundaries) 
    def __init__(self, _nfilters_init,  _NClasses,  norm_type = 'BatchNorm', norm_groups=None, **kwards):
        super().__init__()
        
        self.model_name = "Head_CMTSK_BC" 

        self.nfilters = _nfilters_init # Initial number of filters 
        self.NClasses = _NClasses
        
            
        with self.name_scope():
 

            self.psp_2ndlast = PSP_Pooling(self.nfilters, _norm_type = norm_type, norm_groups=norm_groups)
            
            # bound logits 
            self.bound_logits = HeadSingle(self.nfilters, self.NClasses, norm_type = norm_type, norm_groups=norm_groups)
            self.bound_Equalizer = Conv2DNormed(channels = self.nfilters,kernel_size =1, _norm_type=norm_type, norm_groups=norm_groups)
            
            # distance logits -- deeper for better reconstruction 
            self.distance_logits = HeadSingle(self.nfilters, self.NClasses, norm_type = norm_type, norm_groups=norm_groups)
            self.dist_Equalizer = Conv2DNormed(channels = self.nfilters,kernel_size =1, _norm_type=norm_type, norm_groups=norm_groups)


            self.Comb_bound_dist =  Conv2DNormed(channels = self.nfilters,kernel_size =1, _norm_type=norm_type, norm_groups=norm_groups)


            # Segmenetation logits -- deeper for better reconstruction 
            self.final_segm_logits = HeadSingle(self.nfilters, self.NClasses, norm_type = norm_type, norm_groups=norm_groups)
         


            self.CrispSigm = SigmoidCrisp()

            # Last activation, customization for binary results
            if ( self.NClasses == 1):
                self.ChannelAct = gluon.nn.HybridLambda(lambda F,x: F.sigmoid(x))
            else:
                self.ChannelAct = gluon.nn.HybridLambda(lambda F,x: F.softmax(x,axis=1))

    def hybrid_forward(self,F, UpConv4, conv1):
    
        
         # second last layer 
        convl = F.concat(conv1,UpConv4)
        conv = self.psp_2ndlast(convl)
        conv = F.relu(conv)
    

        # logits 

        # 1st find distance map, skeleton like, topology info
        dist = self.distance_logits(convl) # do not use max pooling for distance
        dist = self.ChannelAct(dist)
        distEq = F.relu(self.dist_Equalizer(dist)) # makes nfilters equals to conv and convl  


        # Then find boundaries 
        bound = F.concat(conv, distEq)
        bound = self.bound_logits(bound)
        bound   = self.CrispSigm(bound) # Boundaries are not mutually exclusive 
        boundEq = F.relu(self.bound_Equalizer(bound))


        # Now combine all predictions in a final segmentation mask 
        # Balance first boundary and distance transform, with the features
        comb_bd = self.Comb_bound_dist(F.concat(boundEq, distEq,dim=1))
        comb_bd = F.relu(comb_bd)

        all_layers = F.concat(comb_bd, conv)
        final_segm = self.final_segm_logits(all_layers)
        final_segm = self.ChannelAct(final_segm)


        return  final_segm, bound, dist







