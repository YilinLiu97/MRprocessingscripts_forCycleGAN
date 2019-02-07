import sys
import re
import numpy as np
import nibabel as nib
from scipy.misc import imread,imsave,imresize,imrotate
from scipy.ndimage import zoom
import os
from os import listdir
from os import path
from os.path import isfile,join
import tensorflow as tf


def rotate(slice,n_images=6):
    start_angle, end_angle = 0,360
    iterate_at = (end_angle - start_angle) / (n_images - 1)

    rotate_slices = []
    for index in range(n_images):
        degrees_angle = start_angle + index * iterate_at
        slice = imrotate(slice,degress_angle)
        rotate_slices.append(slice)

    return rotate_slices


def resize(image, new_shape, interpolation="linear"):
    image = reorder_img(image, resample=interpolation)
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)
    new_data = resample_to_spacing(image.get_data(), image.header.get_zooms(), new_spacing,
                                   interpolation=interpolation)
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3, 3] += calculate_origin_offset(new_spacing, image.header.get_zooms())
    return new_img_like(image, new_data, affine=new_affine)

def clipped_zoom(img,zoom_factor):

    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = imresize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
 #   assert result.shape[0] == height and result.shape[1] == width
    return result

def save_slices(vol,foldername,ID, istrain=True):
    """ Function to save row of image slices """
    num = 0
    for i, slice in enumerate(vol[1]):
        slice_1 = vol[:,i,:]

        if istrain:
           scaling_factors = [0.8,1.2,1]
           rotation_angles = [0,30,45,60,90,180,270]

           rotate = np.random.choice(2)*2-1
           scale = np.random.choice(2)*2-1

           if rotate > 0:
              slice_1 = imrotate(slice_1,np.random.choice(rotation_angles,p=[0.4,0.1,0.1,0.1,0.1,0.1,0.1]))
           if scale > 0:
              slice_1 = clipped_zoom(slice_1,np.random.choice(scaling_factors,p=[0.25,0.25,0.5]))

           imsave('{}/{}slice_{}.jpg'.format(foldername,ID,num),np.array(slice_1))
           num = num+1

           print slice_1.shape
           print '...The slice was successfully saved as {}slice_{}.jpg'.format(ID,num)
        else:
           slice_1 = imrotate(slice_1,90)
           imsave('{}/{}slice_{}.jpg'.format(foldername,ID, num),np.array(slice_1))
           num = num+1

           print slice_1.shape
           print '...The slice was successfully saved as {}slice_{}.jpg'.format(ID,num)



dataset_path = 'trainB_orig'
foldername = 'trainB'   #the folder to save the slices
istrain = True
#########################################################

filenames = listdir(dataset_path)
filenames.sort(key=lambda f: int(filter(str.isdigit, f)))
print(filenames)

#testA_files = filenames[3:]
#print(trainA_files)

for f in filenames:
    vol = nib.load(join(dataset_path,f)).get_data()
    ID = re.split('(\d+)',f)[1]

    save_slices(vol,foldername,ID,istrain)

