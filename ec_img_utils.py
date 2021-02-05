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
import scipy
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
    if isinstance(img, np.ndarray):
        # create dict to store image info
        info_dict = dict()

        info_dict['Shape'] = [img.shape]
        info_dict['Data type'] = [img.dtype]
        info_dict['Bytes'] = [img.itemsize * np.prod(img.shape)]
        info_dict['Range'] = [np.min(img), np.max(img)]

        # print info to display
        print("Image Information\n\n", tabulate(info_dict, headers="keys", tablefmt='github'))

        # return info_dict
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

    # THIS FUNCTION DOES NOT WORK FOR ALL IMAGES... I NEED TO EXPLORE MORE HOW
    # TO CONSISTENTLY EXTRACT METADATA...

    # Open file
    with open(filename_with_path, 'rb') as f:
        # read_data = f.read()
        # return Exif tags
        tags = exifread.process_file(f)
        return tags

    # %%


# INTENSITY TRANSFORMATIONS FUNCTIONS

def log_xfm(img, args_dict):
    # get img class
    img_cls = img.dtype

    # convert image to float for processing
    if 'float' not in str(img_cls):
        img = img.astype(np.float64)  # note that for the log xfm, we do
        # not scale to [0,1]

    # check if the constant parameter was passed to the log xfm
    if 'C' in args_dict.keys():
        C = args_dict['C']
    else:
        C = 1

    # perorm log xfm
    out_img = C * np.log(1 + img)

    # check if an output class was passed in the args_dict
    if 'out_cls' in args_dict.keys():
        if args_dict['out_cls'] == 'uint8':
            out_img = skimage.img_as_ubyte(out_img / np.max(out_img))
        elif args_dict['out_cls'] == 'uint16':
            out_img = skimage.img_as_uint(out_img / np.max(out_img))
        else:
            Exception("Unsupported output class")
    # else, if dictionary is empty, convert to uint8
    elif not bool(args_dict):
        out_img = skimage.img_as_ubyte(out_img / np.max(out_img))

    return out_img


def stretch_xfm(img, args_dict):
    # convert image to float image
    img = skimage.img_as_float(img)

    # if args_dict is empty, assign defaults
    if not bool(args_dict):
        slope = 4.0
        k = np.mean(img)
        # k = 0.5
    elif len(args_dict.keys()) == 2:
        slope = args_dict['slope']
        k = args_dict['k']
    else:
        Exception("Incorrect number of inputs for stretch method")

    # out_img = 1/(1+np.power((k/img+1e-6),slope))
    out_img = 1 / (1 + np.power(np.divide(k, img + 1e-10), slope))
    # I MAY HAVE TO WRITE THIS AS A TABLE LOOK-UP/INTERPOLATION OPERATION INSTEAD

    # rescale and convert back to uint8
    out_img = skimage.img_as_ubyte(out_img / np.max(out_img))

    return out_img


def specified_xfm(img, args_dict):
    # convert image to floating point
    img = skimage.img_as_float(img)

    # parse through arguments
    # check if args_dict is empty
    if not bool(args_dict):
        raise Exception("Not enough inputs")

    else:
        # extract intensity transformation function
        txfun = args_dict['txfun']

        # create intensity vector for interpolation within range [0,1]
        r = np.linspace(1e-10, 1, num=256)

        # interpolate
        out_img = np.interp(img, r, txfun)

    # convert image back to uint8 and return
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
        # the log transformation handles image classes differently than the other
        # xfms, so let the log_transform function handle that and then return
        out_img = log_xfm(img, args_dict)

    elif method == 'neg':
        out_img = skimage.util.invert(img)

    elif method == 'stretch':
        out_img = stretch_xfm(img, args_dict)

    elif method == 'specified':
        out_img = specified_xfm(img, args_dict)

    return out_img


# %%
# Frequency Domain Filtering utility functions
def dftuv(M, N):
    # set up range of values
    u = np.arange(0, M)
    v = np.arange(0, N)

    # compute the indices for use in meshgrid corresponding to the positive
    # frequency points in a 2D FT

    idx = u > M / 2
    u[idx] = u[idx] - M

    idy = v > N / 2
    v[idy] = v[idy] - N

    # compute the meshgrid arrays
    U, V = np.meshgrid(u, v)

    return U, V


def next_pwr_2(n):
    return np.ceil(np.log2(n))


def padded_size(params_dict):
    '''
    

    Parameters
    ----------
    params_dict : TYPE
        DESCRIPTION.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    padded_dim : TYPE
        DESCRIPTION.

    '''
    # check if the parameter dictionary is empty
    if not bool(params_dict):
        raise Exception("Invalid input arguments")

    else:
        # Go through cases
        if len(params_dict.keys()) == 1 and 'img_dim' in params_dict.keys():
            # check if the input dimensions vector is an array
            if type(params_dict['img_dim']) != np.ndarray:
                dims = np.array(params_dict['img_dim'])

            # compute output padded dimensions
            padded_dim = 2 * dims

        elif len(params_dict.keys()) == 2 and 'img_dim' in params_dict.keys() and 'pwr2' in params_dict.keys():
            # check if the input dimensions vector is an array
            if type(params_dict['img_dim']) != np.ndarray:
                dims = np.array(params_dict['img_dim'])

            # Extract max dimension in case image is not a square
            max_dim = np.max(params_dict['img_dim'])

            # compute next power of two
            out_dim = np.power(next_pwr_2(2 * max_dim), 2)

            # create padded_dim array
            padded_dim = np.array([out_dim, out_dim])

        elif len(params_dict.keys()) == 2 and 'img_dim' in params_dict.keys() and 'krnl_dim' in params_dict.keys():
            # check if the input dimensions vector is an array
            if type(params_dict['img_dim']) != np.ndarray:
                dims = np.array(params_dict['img_dim'])

            # check if the input dimensions vector is an array
            if type(params_dict['krnl_dim']) != np.ndarray:
                dims = np.array(params_dict['krnl_dim'])

            out_dim = params_dict['img_dim'] + params_dict['krnl_dim'] - 1
            padded_dim = 2 * np.ceil(out_dim / 2)

        elif len(params_dict.keys()) == 3:
            if type(params_dict['img_dim']) != np.ndarray:
                img_dims = np.array(params_dict['img_dim'])

            # check if the input dimensions vector is an array
            if type(params_dict['krnl_dim']) != np.ndarray:
                krnl_dims = np.array(params_dict['krnl_dim'])

            max_dim = np.max([img_dims, krnl_dims])

            # compute next pwr of two
            out_dim = np.power(2, next_pwr_2(2 * max_dim))

            # generate padded_dim vector
            padded_dim = np.array([out_dim, out_dim])

    return padded_dim


def dft_filt(img, H, pad_method):
    # Get image and kernel dimensions
    img_dim = img.shape
    krnl_dim = H.shape

    # Get padding dimensions from padded_size
    padded_dim = padded_size({"img_dim": img_dim, "krnl_dim": krnl_dim, "pwr2": True})

    # Pad img to the size of the transfer function, using the default or the specified pad_method
    img_padded = np.pad(img, (int(padded_dim[0] - img_dim[0]),int(padded_dim[1] - img_dim[1])) , mode=pad_method)

    # Compute the FFT of the input image
    F = scipy.fft.fft2(img_padded)

    # Perform filtering. The resulting image is back in the spatial domain
    g = scipy.fft.ifft2(F * H)

    # Crop to the original size
    g = g[0:img_dim[0], 0:img_dim[1]]

    return g


def lp_filter(flt_type, M, N, D0, n=1):
    # Protect against upper case
    flt_type = flt_type.lower()

    # Create meshgrid
    U, V = dftuv(M, N)

    # compute the distances in the meshgrid
    D = np.hypot(U, V)

    # Go through different filter type cases
    if flt_type == 'ideal':
        H = D <= D0
    elif flt_type == 'butterworth':
        H = lp_butterworth(M, N, D0, n)
    elif flt_type == 'gaussian':
        H = lp_gaussian(M, N, D0)
    else:
        raise Exception("Filter type unknown")

    return H


def lp_butterworth(M, N, D0, n):
    # Create meshgrid
    U, V = dftuv(M, N)

    # compute the distances in the meshgrid
    D = np.hypot(U, V)

    # Compute xfr function
    H = 1 / (1 + (D / D0) ** (2 * n))

    return H


def lp_gaussian(M, N, D0):
    '''
    Function to return a frequency domain transfer function H(u,v) of size 
    M x N for a lowpass Gaussian filter with cutoff frequency (std dev) at D0.
    
    Adapted from Digital Image Processing Using Matlab, 3rd ed. by Gonzalez et al.
    
    Parameters
    ----------
    M : int
        Vertical dimension (e.g., M x N) of filter transfer function
    N : int
        Horizontal dimension (e.g., M x N) of filter transfer function
    D0 : float or int
        Cutoff frequency (std dev). Must be positive

    Returns
    -------
    Transfer function H of float type

    '''
    # generate the meshgrid from the input dimensions using function dftuv
    U, V = dftuv(M, N)

    # compute the distances from the origin to each point in the grid
    D = np.sqrt(U ** 2 + V ** 2)

    # compute Gaussian transer function
    H = np.exp(-np.power(D, 2) / (2 * np.power(D0, 2)))

    return H
