import mxnet as mx
from mxnet  import gluon 

def get_norm(name, axis=1, norm_groups=None):
    if (name == 'BatchNorm'):                                
        return gluon.nn.BatchNorm(axis=axis)
    elif (name == 'InstanceNorm'):                           
        return gluon.nn.InstanceNorm(axis=axis)
    elif (name == 'LayerNorm'):  
        return gluon.nn.LayerNorm(axis=axis)
    elif (name == 'GroupNorm' and norm_groups is not None):
        return gluon.nn.GroupNorm(num_groups = norm_groups) # applied to channel axis 
    else:                             
        raise NotImplementedError     
