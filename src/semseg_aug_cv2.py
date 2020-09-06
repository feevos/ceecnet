import cv2 
import itertools
import numpy as np


class ParamsRange(dict):
    def __init__(self):
        
        
        # Good default values for 256x256 images 
        self['center_range']   =[0,256]
        self['rot_range']      =[-85.0,85.0]
        self['zoom_range']     = [0.75,1.25]
        self['noise_mean']     = [0]*5
        self['noise_var']      = [10]*5

        
class SemSegAugmentor_CV(object):
    """
    INPUTS: 
        parameters range for all transformations 
        probability of transformation to take place - default to 1. 
        Nrot: number of rotations in comparison with reflections x,y,xy. Default to equal the number of reflections. 
    """
    def __init__(self, params_range, prob = 1.0, Nrot=5, norm = None, one_hot = True):
        
        self.norm = norm # This is a necessary hack to apply brightness normalization 
        self.one_hot = one_hot 
        self.range = params_range
        self.prob = prob
        assert self.prob <= 1 , "prob must be in range [0,1], you gave prob::{}".format(prob)
    

        # define a proportion of operations? 
        self.operations = [self.reflect_x, self.reflect_y, self.reflect_xy,self.random_brightness, self.random_shadow]
        self.operations += [self.rand_shit_rot_zoom]*Nrot
        self.iterator = itertools.cycle(self.operations)
         
    
    def _shift_rot_zoom(self,_img, _mask, _center, _angle, _scale):
        """
        OpenCV random scale+rotation 
        """
        imgT = _img.transpose([1,2,0])
        if (self.one_hot):
            maskT = _mask.transpose([1,2,0])
        else:
            maskT = _mask
        
        cols, rows = imgT.shape[:-1]
        
        # Produces affine rotation matrix, with center, for angle, and optional zoom in/out scale
        tRotMat = cv2.getRotationMatrix2D(_center, _angle, _scale)
    
        img_trans = cv2.warpAffine(imgT,tRotMat,(cols,rows),flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT_101) #  """,flags=cv2.INTER_CUBIC,""" 
        mask_trans= cv2.warpAffine(maskT,tRotMat,(cols,rows),flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT_101)
    
        img_trans = img_trans.transpose([2,0,1])
        if (self.one_hot):
            mask_trans = mask_trans.transpose([2,0,1])

        return img_trans, mask_trans
    
    
    def reflect_x(self,_img,_mask):
        
        img_z  = _img[:,::-1,:]
        if self.one_hot:
            mask_z = _mask[:,::-1,:] # 1hot representation
        else:
            mask_z = _mask[::-1,:] # standard (int's representation)
        
        return img_z, mask_z 
        
    def reflect_y(self,_img,_mask):
        img_z  = _img[:,:,::-1]
        if self.one_hot:
            mask_z = _mask[:,:,::-1] # 1hot representation
        else:
            mask_z = _mask[:,::-1] # standard (int's representation)
        
        return img_z, mask_z 
        
    def reflect_xy(self,_img,_mask):
        img_z  = _img[:,::-1,::-1]
        if self.one_hot:
            mask_z = _mask[:,::-1,::-1] # 1hot representation
        else:
            mask_z = _mask[::-1,::-1] # standard (int's representation)
        
        return img_z, mask_z 
    
        
        
    def rand_shit_rot_zoom(self,_img,_mask):
        
        center = np.random.randint(low=self.range['center_range'][0],
                                  high=self.range['center_range'][1],
                                  size=2)
        # This is in radians
        angle = np.random.uniform(low=self.range['rot_range'][0],
                                  high=self.range['rot_range'][1])
        
        scale = np.random.uniform(low=self.range['zoom_range'][0],
                                  high=self.range['zoom_range'][1])
 

        return self._shift_rot_zoom(_img,_mask,tuple(center),angle,scale) #, tuple(center),angle,scale



    # ============================================ New additions below =======================================================
    # **************** Random brightness (light/dark) and random shadow polygons *************
    # ******** Taken from: https://medium.freecodecamp.org/image-augmentation-make-it-rain-make-it-snow-how-to-modify-a-photo-with-machine-learning-163c0cb3843f 
    # ******** See https://github.com/UjjwalSaxena/Automold--Road-Augmentation-Library for library 

    def random_brightness(self,_img, _mask):
        """
        This function only applies only on the first 3 channels (RGB) of an image.  
	Input: RGB image, transforms to np.uint8
        Output: RGB image + extra channels. 
        """

        if self.norm is not None:
            image = self.norm.restore(_img).transpose([1,2,0])[:,:,:3].copy() # use only three bands
            imgcp = self.norm.restore(_img.copy()) # use only three bands
    
        else : 
            
            image = _img.transpose([1,2,0])[:,:,:3].copy().astype(np.uint8) # use only three bands
            imgcp = _img.copy() .astype(np.uint8)# use only three bands
        
        image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
        image_HLS = np.array(image_HLS, dtype = np.float64) 
        random_brightness_coefficient = np.random.uniform()+0.5 ## generates value between 0.5 and 1.5
        image_HLS[:,:,1] = image_HLS[:,:,1]*random_brightness_coefficient ## scale pixel values up or down for channel 1(Lightness)
        image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
        image_HLS = np.array(image_HLS, dtype = np.uint8)
        image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion back to RGB
        

        imgcp[:3,:,:] = image_RGB.transpose([2,0,1])

        if self.norm is not None:
            imgcp = self.norm(imgcp)

        return imgcp.astype(_img.dtype), _mask

   

    def _generate_shadow_coordinates(self,imshape, no_of_shadows=1):
        vertices_list=[]
        for index in range(no_of_shadows):
            vertex=[]
            for dimensions in range(np.random.randint(3,15)): ## Dimensionality of the shadow polygon
                vertex.append(( imshape[1]*np.random.uniform(), imshape[0]*np.random.uniform()))
            vertices = np.array([vertex], dtype=np.int32) ## single shadow vertices 
            vertices = cv2.convexHull(vertices[0])
            vertices = vertices.transpose([1,0,2])
            vertices_list.append(vertices)
        return vertices_list ## List of shadow vertices

    def _add_shadow(self, image, no_of_shadows=1):
        image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
        tmask = np.zeros_like(image[:,:,0]) 
        imshape = image.shape
        vertices_list= self._generate_shadow_coordinates(imshape, no_of_shadows) #3 getting list of shadow vertices
        for vertices in vertices_list: 
            cv2.fillPoly(tmask, vertices, 255) 
        image_HLS[:,:,1][tmask[:,:]==255] = image_HLS[:,:,1][tmask[:,:]==255]*0.5   
        image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
        return image_RGB


    def random_shadow(self,_img, _mask):

        
        if self.norm is not None:
            image = self.norm.restore(_img).transpose([1,2,0])[:,:,:3].copy() # use only three bands
            imgcp = self.norm.restore(_img.copy()) # use only three bands
    
        else : 

            image = _img.transpose([1,2,0])[:,:,:3].copy().astype(np.uint8)# use only three bands
            imgcp = _img.copy() .astype(np.uint8)# use only three bands

        shadow_image = self._add_shadow(image)
        
        imgcp.transpose([1,2,0])[:,:,:3] = shadow_image

        if self.norm is not None:
            imgcp = self.norm(imgcp)

        return imgcp.astype(_img.dtype), _mask

    # =====================================================================================


    def __call__(self,_img, _mask):
        
        rand = np.random.rand()
        if (rand <= self.prob):
            return next(self.iterator)(_img,_mask)
        else :
            return _img, _mask
