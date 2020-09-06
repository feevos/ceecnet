"""
Code: slicing of large raster images in image patches of window size F (= 256). In this code, the ~10% of the area of each image
is kept as validation data. To achieve this we keep the lowest (bottom right) 10% of each tile as validation data. This is done by 
using all the indices corresponding to the lowest 10% of area (i.e. after the ~70% of the length of each area). 

Area_test = (0.3 * Height) * (0.3 * Width) ~= 0.1 * Height*Width
"""


from chopchop2rec import * 

# Reads triplet of names  
import glob 
flname_prefix_data = r'/Location/Of/LEVIRCD/Data/'
flnames_images_A = sorted(glob.glob(flname_prefix_data + r'train/A/*.png'))
flnames_images_A += sorted(glob.glob(flname_prefix_data + r'val/A/*.png'))
flnames_images_A = sorted(flnames_images_A)


flnames_images_B = sorted(glob.glob(flname_prefix_data + r'train/B/*.png'))
flnames_images_B += sorted(glob.glob(flname_prefix_data + r'val/B/*.png'))
flnames_images_B = sorted(flnames_images_B)


flnames_images_chng = sorted(glob.glob(flname_prefix_data + r'train/label/*.png'))
flnames_images_chng += sorted(glob.glob(flname_prefix_data + r'val/label/*.png'))
flnames_images_chng = sorted(flnames_images_chng)


listOfAll123 = list(zip(flnames_images_A,flnames_images_B,flnames_images_chng))


if __name__ == '__main__':

    print ("Starting Chopping...")   
    mywriter = WriteDataRecordIO(listOfAll123)
    mywriter.chop_all()
    print ("Done!")


