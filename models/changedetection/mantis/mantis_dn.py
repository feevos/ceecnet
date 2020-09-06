from ceecnet.models.heads.head_cmtsk import *
from ceecnet.models.changedetection.mantis.mantis_dn_features import * 


# Mantis conditioned multitasking. 
class mantis_dn_cmtsk(HybridBlock):
    def __init__(self, nfilters_init, depth, NClasses,widths=[1], psp_depth=4,verbose=True, norm_type='BatchNorm', norm_groups=None,nheads_start=8,  model='CEECNetV1', upFuse=False, ftdepth=5,**kwards):
        super().__init__(**kwards)
        
        with self.name_scope():
            
            self.features = mantis_dn_features(nfilters_init=nfilters_init, depth=depth, widths=widths, psp_depth=psp_depth, verbose=verbose, norm_type=norm_type, norm_groups=norm_groups,  nheads_start=nheads_start, model=model,  upFuse=upFuse,  ftdepth=ftdepth, **kwards)
            self.head = Head_CMTSK_BC(nfilters_init,NClasses, norm_type=norm_type, norm_groups=norm_groups, **kwards)
            
    def hybrid_forward(self,F,input_t1, input_t2):
        out1, out2= self.features(input_t1,input_t2)

        return self.head(out1,out2)

