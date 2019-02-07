import sys
import re
import numpy as np
import nibabel as nib

from nilearn.image import reorder_img, new_img_like
from scipy.misc import imread, imsave,imresize, imrotate
import os
from os import listdir
from os import path
from os.path import isfile,join
import fnmatch

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def saveNifti(image,imageName,affine):
    niiToSave = nib.Nifti1Image(image,affine)
    niiToSave.set_data_dtype(np.dtype(np.int16))
    nib.save(niiToSave,imageName)

    print("... Image successfully saved as ", imageName)

def reconstruct(IDs, slices_path, affine):
    """ Function to recontruct a full 3D volume from a stack of slices """
    sliceNames = listdir(slices_path)
    print sliceNames
    sliceNames.sort(key=lambda f: int(filter(str.isdigit, f)))
    for ID in IDs:
        pattern = '{}slice'.format(ID)
        print pattern
        slices = []  #np.empty((191,171))
        for name in sliceNames:
          #  print name
            if (name, re.match(pattern,name)):
                print name
                slice = imread(join(slices_path,name))
                print slice.shape
                slice = imrotate(slice,-90)
                slice = imresize(slice,(191,171))

                slices.append(slice)
        print np.array(slices).shape
        slices = np.transpose(slices,(1,0,2))
        saveNifti(np.array(slices),str(ID),affine)
        print np.array(slices).shape

# reconstruct the full volume
vol = nib.load('/study/utaut2/Yilin/ISMRM_Dataset/Training/subject205.nii')
affine = vol.affine

#save_trainslices(vol,'205')

slices_path = '/study/utaut2/Yilin/pytorch-CycleGAN-and-pix2pix/datasets/MR/tmp'
reconstruct([205],slices_path,affine)


