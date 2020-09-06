"""
DataSet reader for the LEVIRCD dataset. 
"""

import numpy as np
import glob 
from mxnet.gluon.data import dataset
import cv2
import mxnet as mx 
import pickle 

class LVRCDDataset(dataset.Dataset):
    def __init__(self, root=r'/Location/Of/Your/LEVIRCD/Files/', mode='train', mtsk = True, transform=None, norm=None, pMessUp=0.0, Filter=256, prob_swap=0.5, prob_zero_change=0.5):
        
        self.NClasses=2
        self.NChannels=3 # RGB 
        # Transformation of augmented data
        self._mode = mode
        self.mtsk = mtsk

        self.prob_swap = prob_swap 
        self.prob_zero_change = prob_zero_change


        self._transform = transform
        self._norm = norm # Normalization of img
 
        if (root[-1]!='/'):
            root = root + r'/'

        if mode is 'train':
            flname_idx = glob.glob(root + r'training_LVRCD_F{}.idx'.format(Filter))[0]
            flname_rec = glob.glob(root + r'training_LVRCD_F{}.rec'.format(Filter))[0]
        elif mode is 'val':
            flname_idx = glob.glob(root + r'validation_LVRCD_F{}.idx'.format(Filter))[0]
            flname_rec = glob.glob(root + r'validation_LVRCD_F{}.rec'.format(Filter))[0]
        else:
            raise Exception ('I was given inconcistent mode, available choices: {train, val}, aborting ...')


        self.record = mx.recordio.MXIndexedRecordIO(idx_path=flname_idx, uri=flname_rec , flag='r')
         
    def get_boundary(self, labels, _kernel_size = (3,3)):
        
        label = labels.copy().astype(np.uint8)
        for channel in range(label.shape[0]):
            temp = cv2.Canny(label[channel],0,1)
            label[channel] = cv2.dilate(temp, cv2.getStructuringElement(cv2.MORPH_CROSS,_kernel_size) ,iterations = 1)
    
        label = label.astype(np.float32)
        label /= 255.
        return label

    def get_distance(self,labels):
        label = labels.copy().astype(np.uint8)
        dists = np.empty_like(label,dtype=np.float32)
        for channel in range(label.shape[0]):
            dist = cv2.distanceTransform(label[channel], cv2.DIST_L2, 0)
            dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
            dists[channel] = dist
        
        return dists
    

    def __getitem__(self, idx):
 
        key = self.record.keys[idx]
        imgall = pickle.loads(self.record.read_idx(key))

        base = imgall[:self.NChannels*2].astype(np.float32)
        mask = imgall[self.NChannels*2:].astype(np.float32)
        mask[self.NClasses*2:] = mask[self.NClasses*2:]/100. # Bring the distance transform to 0,1 scale 

         
        if self.mtsk == False:
            mask = mask[:self.NClasses,:,:] 

        if self._transform is not None:
            base, mask = self._transform(base, mask)
            # RGB images, 3 bands
            t1 = base[:self.NChannels]
            t2 = base[self.NChannels:]
            
            
            # @@@@@@@@@@@@@@@@@@@@@ TWO ESSENTIAL transformations @@@@@@@@@@@@@@@
            # Select randomly  NOCHANGE or Great Scott
            if np.random.rand() >= 0.5:
                # Great Scott: random time ordering
                if np.random.rand() >= self.prob_swap:
                    temp = t2.copy()
                    t2 = t1
                    t1 = temp
            else:
                # NOCHANGE to help avoid learning buildings as a mask 
                if np.random.rand() >= self.prob_zero_change:
                    if np.random.rand() >= 0.5:
                        t2 = t1
                    else: 
                        t1 = t2
                    # Segmentation is all NOCHANGE now  fix mask 
                    mask = mask[:self.NClasses,:,:]
                    # No CHANGE 
                    mask[0] = 1
                    mask[1] = 0
                    boundaries = self.get_boundary(mask)
                    dists = self.get_distance(mask)
                    mask = np.concatenate([mask,boundaries,dists],axis=0)
                    mask = mask.astype(np.float32)
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



            if self._norm is not None:
                t1 = self._norm(t1.astype(np.float32))
                t2 = self._norm(t2.astype(np.float32))

            return t1.astype(np.float32), t2.astype(np.float32), mask.astype(np.float32)
            
        else:
            # RGB images, 3 bands
            t1 = base[:self.NChannels]
            t2 = base[self.NChannels:]
            if self._norm is not None:
                t1 = self._norm(t1.astype(np.float32))
                t2 = self._norm(t2.astype(np.float32))


            return t1.astype(np.float32), t2.astype(np.float32),  mask.astype(np.float32)

    def __len__(self):
        return len(self.record.keys)


