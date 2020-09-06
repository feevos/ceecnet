from ceecnet.nn.loss.ftnmt_loss import *

class mtsk_loss(object):
    """
    Here NClasses = 2 by default, for a binary segmentation problem in 1hot representation 
    """

    def __init__(self,depth=0, NClasses=2):

        self.ftnmt = ftnmt_loss(depth=depth)
        self.ftnmt.hybridize() 

        self.skip = NClasses

    def loss(self,_prediction,_label):

        pred_segm  = _prediction[0]
        pred_bound = _prediction[1]
        pred_dists = _prediction[2]
        
        # In our implementation of the labels, we stack together the [segmentation, boundary, distance] labels, 
        # along the channel axis. 
        label_segm  = _label[:,:self.skip,:,:]
        label_bound = _label[:,self.skip:2*self.skip,:,:]
        label_dists = _label[:,2*self.skip:,:,:]


        loss_segm  = self.ftnmt(pred_segm,   label_segm)
        loss_bound = self.ftnmt(pred_bound, label_bound)
        loss_dists = self.ftnmt(pred_dists, label_dists)

        return (loss_segm+loss_bound+loss_dists)/3.0

