# ============================== Helper Functions ==================================
# Helper functions to create boundary and distance transform
# ground trouth label in 1hot format 
import cv2
import  glob
import numpy as np


def get_boundary(labels, _kernel_size = (3,3)):

    label = labels.copy()
    for channel in range(label.shape[0]):
        temp = cv2.Canny(label[channel],0,1)
        label[channel] = cv2.dilate(temp, cv2.getStructuringElement(cv2.MORPH_CROSS,_kernel_size) ,iterations = 1)

    label = label.astype(np.float32)
    label /= 255.
    label = label.astype(np.uint8)
    return label

def get_distance(labels):
    label = labels.copy()
    dists = np.empty_like(label,dtype=np.float32)
    for channel in range(label.shape[0]):
        dist = cv2.distanceTransform(label[channel], cv2.DIST_L2, 0)
        dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        dists[channel] = dist

    dists = dists * 100.
    dists = dists.astype(np.uint8)
    return dists
# ====================================================================================



import mxnet as mx
import pickle # this is necessary for translating from/to string for writing/reading in mxnet recordio
from multiprocessing import Lock
from pathos.pools import ThreadPool as pp
import rasterio


class WriteDataRecordIO(object):
    def __init__(self,
                 ListOfFlnames123,
                 flname_prefix_write= r'/Location/Of/Your/LEVIRCD/Files/',
                 NClasses=2, # 1hot encoding
                 Filter=256,
                 stride_div=2,
                 length_scale = 0.317):


        self.listOfFlnames123 = ListOfFlnames123
        self.lock = Lock()

        self.Filter=Filter
        self.stride = Filter//stride_div

        self.teye_label = np.eye(NClasses,dtype=np.uint8)
        self.global_train_idx = 0
        self.global_valid_idx = 0

        flname_train_idx = flname_prefix_write + r'training_LVRCD_F{}.idx'.format(Filter)
        flname_train_rec = flname_prefix_write + r'training_LVRCD_F{}.rec'.format(Filter)

        flname_valid_idx = flname_prefix_write + r'validation_LVRCD_F{}.idx'.format(Filter)
        flname_valid_rec = flname_prefix_write + r'validation_LVRCD_F{}.rec'.format(Filter)

        self.record_train = mx.recordio.MXIndexedRecordIO(idx_path=flname_train_idx, uri=flname_train_rec , flag='w')
        self.record_valid = mx.recordio.MXIndexedRecordIO(idx_path=flname_valid_idx, uri=flname_valid_rec , flag='w')

        self.length_scale=length_scale

        self.Filter = Filter
        self.stride = Filter // stride_div

    def update_imgs_mask(self, flnames123):
        name_img1, name_img2, name_mask = flnames123


        with rasterio.open(name_img1,mode='r',driver='png') as src1:
            img1 = src1.read()
            
        with rasterio.open(name_img2,mode='r',driver='png') as src2:
            img2 = src2.read()

        with rasterio.open(name_mask,mode='r',driver='png') as srcm:
            self.label = srcm.read(1)
            self.label[ self.label > 0 ]  = 1

        self.img = np.concatenate((img1,img2),axis=0)

        # Constants that relate to rows, columns 
        self.nTimesRows = int((self.img.shape[1] - self.Filter)//self.stride + 1)
        self.nTimesCols = int((self.img.shape[2] - self.Filter)//self.stride + 1)


        self.nTimesRows_val = int((1.0-self.length_scale)*self.nTimesRows)
        self.nTimesCols_val = int((1.0-self.length_scale)*self.nTimesCols)


    def _2D21H(self,tmask_label):

        tmask_label_1h = self.teye_label[tmask_label]
        tmask_label_1h = tmask_label_1h.transpose([2,0,1])
        distance_map = get_distance(tmask_label_1h)
        bounds_map = get_boundary(tmask_label_1h)
        tlabels_all = np.concatenate([tmask_label_1h, bounds_map, distance_map],axis=0)

        return tlabels_all

    def chop_all(self):
        # For all triples in list of filenames 
        for idx,name123 in enumerate(self.listOfFlnames123):
            print ("============================")
            print ("----------------------------")
            print ("Processing:: {}/{} triplets".format(idx, len(self.listOfFlnames123)))
            print ("----------------------------")
            for name in name123:
                print("Processing File:{}".format(name))
            print ("****************************")

            # read image and mask 
            self.update_imgs_mask(name123)

            # Do the chop on specific images 
            self.thread_chop()


        self.record_train.close()
        self.record_valid.close()

    # Change here nthread to maximum available threads you have (or less)
    def thread_chop(self,nthread=24):
        """
        Extracts patches in parallel from a single tuple of (raster, label, group_label) 
        """
        RowsCols = [(row, col) for row in range(self.nTimesRows-1) for col in range(self.nTimesCols-1)]
        Rows = [row for row in range(self.nTimesRows-1)]
        Cols = [col for col in range(self.nTimesCols-1)]

        pool = pp(nodes=nthread)
        result1 = pool.map(self.extract_patch,RowsCols)
        result2 = pool.map(self.extract_last_Col,Rows)
        result3 = pool.map(self.extract_last_Row,Cols)
    
    
    def extract_patch(self,  RowCol):
        """
        Single chip extraction.
        """
        row, col = RowCol
        # Extract temporary
        tmask_label  = self.label[row*self.stride:row*self.stride+self.Filter, col*self.stride:col*self.stride+self.Filter].copy().astype(np.uint8)
        timg = self.img[ :, row*self.stride:row*self.stride+self.Filter, col*self.stride:col*self.stride+self.Filter].copy()

        tlabels_all = self._2D21H(tmask_label)

        timg = np.concatenate((timg,tlabels_all),axis=0).astype(np.uint8)
        timg = pickle.dumps(timg)

        self.lock.acquire()

        if row >= self.nTimesRows_val and col >= self.nTimesCols_val :
            self.record_valid.write_idx(self.global_valid_idx,timg)
            self.global_valid_idx += 1
        else:
            self.record_train.write_idx(self.global_train_idx,timg)
            self.global_train_idx += 1

        self.lock.release()




    def extract_last_Col(self,row):
        # Keep the overlapping non integer final row/column images as validation images as well 
        rev_col = self.img.shape[2] - self.Filter
        timg = self.img[:, row*self.stride:row*self.stride+self.Filter, rev_col:].copy()

        tmask_label  = self.label[row*self.stride:row*self.stride+self.Filter, rev_col:].copy().astype(np.uint8)

        tlabels_all = self._2D21H(tmask_label)

        timg = np.concatenate((timg,tlabels_all),axis=0).astype(np.uint8)
        timg = pickle.dumps(timg)

        self.lock.acquire()
        self.record_valid.write_idx(self.global_valid_idx,timg)
        self.global_valid_idx += 1
        self.lock.release()


    def extract_last_Row(self,col):
        # Keep the overlapping non integer final row/column images as validation images as well 
        rev_row = self.img.shape[1] - self.Filter

        timg = self.img[ :, rev_row:, col*self.stride:col*self.stride+self.Filter].copy()
        tmask_label  = self.label[rev_row:, col*self.stride:col*self.stride+self.Filter].copy().astype(np.uint8)

        tlabels_all = self._2D21H(tmask_label)

        timg = np.concatenate((timg,tlabels_all),axis=0).astype(np.uint8)
        timg = pickle.dumps(timg)

        self.lock.acquire()
        self.record_valid.write_idx(self.global_valid_idx,timg)
        self.global_valid_idx += 1
        self.lock.release()



