# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 21:51:16 2021

@author: Ernesto

Image processing module for ECE419

The purpose of this module is to compile/construct a collection of image 
processing utilities as I progress through the course
"""
import sys
import numpy as np
from tabulate import tabulate
import cv2 
import exifread

def get_img_info(img):
    '''
    Function to extract and print image information
    
    Inputs:
        - img: image represented as ndarray 
    Outputs:
        - None, but the function prints the image dimensions, the data type, 
          and the size in bytes
    '''
    if isinstance(img,np.ndarray):
        #create dict to store image info
        info_dict = dict()
        
        info_dict['Shape'] = [img.shape]
        info_dict['Data type'] = [img.dtype]
        info_dict['Bytes'] = [img.itemsize * np.prod(img.shape)]
        
        #print info to display
        print("Image Information\n\n",tabulate(info_dict, headers = "keys", tablefmt = 'github'))
        
        #return info_dict
    else: 
        raise Exception('Image is not array type')



# %%
'''
READING IMAGE EXIF INFORMATION / METADATA
I WILL BE USING EXIFREAD FROM: https://pypi.org/project/ExifRead/

Matlab has a built-in function iminfo that returns metadata for image files. 
My plan is to wrap some of exifread's functionality into a utility function to
extract meaningful information from image files
'''
def get_exif_data(filename_with_path):
    '''
    READING IMAGE EXIF INFORMATION / METADATA
    I WILL BE USING EXIFREAD FROM: HTTPS://PYPI.ORG/PROJECT/EXIFREAD/

    MATLAB HAS A BUILT-IN FUNCTION IMINFO THAT RETURNS METADATA FOR IMAGE FILES. 
    MY PLAN IS TO WRAP SOME OF EXIFREAD'S FUNCTIONALITY INTO A UTILITY FUNCTION TO
    EXTRACT MEANINGFUL INFORMATION FROM IMAGE FILES
    
    INPUT:
        FILENAME_WITH_PATH EXPECTS A STRING WITH THE PATH TO THE FILENAME
    ''' 
    
    #THIS FUNCTION DOES NOT WORK FOR ALL IMAGES... I NEED TO EXPLORE MORE HOW
    #TO CONSISTENTLY EXTRACT METADATA...
    
    #Open file
    with open(filename_with_path, 'rb') as f:
        #read_data = f.read()
        #return Exif tags
        tags = exifread.process_file(f)
        return tags 
    
