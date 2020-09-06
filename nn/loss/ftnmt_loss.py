"""
Fractal Tanimoto (with dual) loss 
"""

from mxnet.gluon.loss import Loss                                                                                                                                       
class ftnmt_loss(Loss):
    """
    This function calculates the average fractal tanimoto similarity for d = 0...depth
    """                                                                                          
    def __init__(self, depth=5, axis= [1,2,3], smooth = 1.0e-5, batch_axis=0, weight=None, **kwargs):
        super().__init__(batch_axis, weight, **kwargs)
        
        assert depth>= 0, ValueError("depth must be >= 0, aborting...")
        
        self.smooth = smooth
        self.axis=axis
        self.depth = depth

        if depth == 0:
            self.depth = 1
            self.scale = 1.
        else:
            self.depth = depth
            self.scale = 1./depth

    def inner_prod(self, F, prob, label):
        prod = F.broadcast_mul(prob,label)
        prod = F.sum(prod,axis=self.axis)

        return prod

    def tnmt_base(self, F, preds, labels):

        tpl  = self.inner_prod(F,preds,labels)
        tpp  = self.inner_prod(F,preds,preds)
        tll  = self.inner_prod(F,labels,labels)
        
       
        num = tpl + self.smooth
        scale = 1./self.depth
        denum = 0.0
        for d in range(self.depth):
            a = 2.**d
            b = -(2.*a-1.)

            denum = denum + F.reciprocal(F.broadcast_add(a*(tpp+tll), b *tpl) + self.smooth)

        result =  F.broadcast_mul(num,denum)*scale       
        return  F.mean(result, axis=0,exclude=True)

                                                                                                                           
    def hybrid_forward(self,F, preds, labels):

        l1 = self.tnmt_base(F,preds,labels)
        l2 = self.tnmt_base(F,1.-preds, 1.-labels)
         
        result = 0.5*(l1+l2) 
         
        return  1. - result
    
