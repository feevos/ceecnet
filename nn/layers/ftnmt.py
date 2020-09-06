from mxnet.gluon import HybridBlock


class FTanimoto(HybridBlock):
    """
    This is the average fractal Tanimoto set similarity with complement. 
    """
    def __init__(self, depth=5, smooth=1.0e-5, axis=[2,3],**kwards):
        super().__init__(**kwards)
        
        assert depth >= 0, "Expecting depth >= 0, aborting ..."

        if depth == 0:
            self.depth = 1
            self.scale = 1.
        else:
            self.depth = depth
            self.scale = 1./depth
        
        self.smooth = smooth
        self.axis=axis
        
    def inner_prod(self, F, prob, label):
        prod = F.broadcast_mul(prob,label)
        prod = F.sum(prod,axis=self.axis,keepdims=True)

        return prod

        

    def tnmt_base(self, F, preds, labels):

        tpl  = self.inner_prod(F,preds,labels)
        tpp  = self.inner_prod(F,preds,preds)
        tll  = self.inner_prod(F,labels,labels)
        
       
        num = tpl + self.smooth
        denum = 0.0


        for d in range(self.depth):
            a = 2.**d
            b = -(2.*a-1.)

            denum = denum + F.reciprocal(F.broadcast_add(a*(tpp+tll), b *tpl) + self.smooth)

        return F.broadcast_mul(num,denum)*self.scale

    def hybrid_forward(self, F, preds, labels):
            l12 = self.tnmt_base(F,preds,labels)
            l12 = l12 + self.tnmt_base(F,1.-preds, 1.-labels)

            return 0.5*l12
