from mxnet.gluon import HybridBlock
import mxnet as mx


class SigmoidCrisp(HybridBlock):
    def __init__(self, smooth=1.e-2,**kwards):
        super().__init__(**kwards)


        self.smooth = smooth
        with self.name_scope():
            self.gamma  = self.params.get('gamma', shape=(1,), init=mx.init.One())


    def hybrid_forward(self, F, input, gamma):
            out = self.smooth + F.sigmoid(gamma)
            out = F.reciprocal(out)

            out = F.broadcast_mul(input,out)
            out = F.sigmoid(out)
            return out 



