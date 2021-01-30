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

import skimage
import skimage.util
from skimage import io
import matplotlib.pyplot as plt

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
        info_dict['Range'] = [np.min(img), np.max(img)]
        
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

# %%
#INTENSITY TRANSFORMATIONS FUNCTIONS

def log_xfm(img, args_dict):
    #get img class
    img_cls = img.dtype
    
    #convert image to float for processing
    if 'float' not in str(img_cls):
        img = img.astype(np.float64)    #note that for the log xfm, we do
                                            #not scale to [0,1]
        
    #check if the constant parameter was passed to the log xfm
    if 'C' in args_dict.keys():
        C = args_dict['C']
    else:
        C = 1
        
    #perorm log xfm
    out_img = C*np.log(1+img)
        
    #check if an output class was passed in the args_dict
    if 'out_cls' in args_dict.keys():
        if args_dict['out_cls'] == 'uint8':
            out_img = skimage.img_as_ubyte(out_img/np.max(out_img))
        elif args_dict['out_cls'] == 'uint16':
            out_img = skimage.img_as_uint(out_img/np.max(out_img))
        else:
            Exception("Unsupported output class")
    #else, if dictionary is empty, convert to uint8
    elif not bool(args_dict):
        out_img = skimage.img_as_ubyte(out_img/np.max(out_img))
        
    return out_img


def stretch_xfm(img, args_dict):
    #convert image to float image
    img = skimage.img_as_float(img)
    
    #if args_dict is empty, assign defaults
    if not bool(args_dict):
        slope = 4.0
        k = np.mean(img)
        #k = 0.5
    elif len(args_dict.keys()) == 2:
        slope = args_dict['slope'] 
        k = args_dict['k']
    else:
        Exception("Incorrect number of inputs for stretch method")
    
    #out_img = 1/(1+np.power((k/img+1e-6),slope))
    out_img = 1/(1+np.power(np.divide(k,img+1e-10),slope))
    #I MAY HAVE TO WRITE THIS AS A TABLE LOOK-UP/INTERPOLATION OPERATION INSTEAD
    
    #rescale and convert back to uint8
    out_img = skimage.img_as_ubyte(out_img/np.max(out_img))

    return out_img

def specified_xfm(img, args_dict):
    #convert image to floating point
    img = skimage.img_as_float(img)
    
    #parse through arguments
    #check if args_dict is empty
    if not bool(args_dict):
        Exception("Not enough inputs")
    else:
        #extract intensity transformation function
        txfun = args_dict['txfun']
        
        #create intensity vector for interpolation within range [0,1]
        r = np.linspace(1e-10,1,num=256)
        
        #interpolate 
        out_img = np.interp(img, r, txfun)
        
    #convert image back to uint8 and return
    return skimage.img_as_ubyte(out_img)



def intensity_xfms(img, method, args_dict):
    '''
    Function to perform intensity transformatin on grayscale images.
    Adapted from Digital Image Processing Using Matlab, 3rd ed by Gonzalez et al.

    Parameters
    ----------
    img : Typically uint8, although most functions will convert to float img
        img object as ndarray.
    method : string
        String describing intensity transformation to be performed.
    args_dict : dict
        Dictionary with input arguments depending on the intensity xfm.
        
        if method == 'log', args_dict = {'C': val1, 'out_cls': cls_str}
        where C is the constant multiplier in the log transformation and 
        out_cls is either uint8 or uint16 representing output img class

        if method == 'stretch', args_dict = {'k':val1, 'slope':val2}
        where k and slope are the functional parameters in the transformation
        out_intensity = 1/(1+ (k/img)^slope))
    
        if method == 'specified', args_dict = {'txfun'} where txfun represents
        an array of intensity values to interpolate with
        
        if method == 'neg', return the photgraphic negative of the image
    
    Returns
    -------
    out_img: uint8 or uint16 (for log xfm) transformed according to the selected
    method (see desciptions above)
   '''
    if method == 'log':
        #the log transformation handles image classes differently than the other
        #xfms, so let the log_transform function handle that and then return
        out_img = log_xfm(img, args_dict)
    
    elif method == 'neg':
        out_img = skimage.util.invert(img)
        
    elif method == 'stretch':
        out_img = stretch_xfm(img, args_dict)
    
    elif method == 'specified':
        out_img = specified_xfm(img, args_dict)
    
    return out_img



