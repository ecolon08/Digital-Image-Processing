# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 21:51:16 2021

@author: Ernesto

Image processing module for ECE419

The purpose of this module is to compile/construct a collection of image 
processing utilities as I progress through the course
"""
import numpy as np
import scipy
from tabulate import tabulate
import exifread

import skimage
import skimage.util
import skimage.restoration
from scipy.ndimage import convolve
from scipy.ndimage import correlate
import skimage.filters
import skimage.transform
from skimage import morphology

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
    """
    Function to compute meshgrid frequency matrices that are useful for computing frequency domain filter transfer
    functions that can be used with dft_filt(). U and V are both of size M x N.

    Parameters
    ----------
    M : int
        Vertical dimension (e.g., M x N) of filter transfer function
    N : int
        Horizontal dimension (e.g., M x N) of filter transfer function

    Returns
    -------
    U, V : ndarray
        meshgrid frequency matrices.
    """

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
    """
    Function to compute the next power of two given an input number n.

    Parameters
    ----------
    n : int or float
        Number to compute the next power of two M such that 2^M >= n.
    Returns
    -------
    M : float
        Power of 2 such that 2^M >= n.
    """
    M = np.ceil(np.log2(n))

    return M


def padded_size(params_dict):
    """
    Function to compute padded sizes useful for FFT-based filtering.

    If only the img_dim is passed in the dictionary, the function returns 2*img_dim

    If in addition to img_dim, the pwer2 flag is passed, the function computes the padded dimensions that are the next
    closest power of 2. This case is useful to speed up FFT computations.

    If both img_dim and krnl_dim are passed, the function returns the sum of the array dimensions
    img_dim + krnl_dim - 1

    Lastly, if img_dim, krnl_dim, and pwr2 are passed, the function computes the maximum dimension from the two arrays
    and returns the next closest power of 2.

    Parameters
    ----------
    params_dict : dict
        Valid keys-value pairs:
            img_dim : array-like with input image's dimensions
            krnl_dim : array-like with input kernel's dimensions
            pwr2 : Boolean flag
    Raises
    ------
    Exception
        The function raises an exception if the params_dict contains invalid inputs.

    Returns
    -------
    padded_dim : array-like
        padded dimensions computed.
    """
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
            out_dim = np.power(2, next_pwr_2(2 * max_dim))

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
    """
    Function to perform frequency domain filtering given an input image and transfer functon.

    Adapted from Digital Image Processing Using Matlab, 3rd ed. by Gonzalez et al.

    Parameters
    ----------
    img : any valid image type supported by skimage
    H : ndarray
        Frequency domain filter transfer function. The function assumes that H already has the correct dimensions needed
        for filtering and has not been fftshifted
    pad_method : string        Padding method for input image. Valid options are outlined in the np.pad documentation:
        https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        modestr or function, optional

        One of the following string values or a user supplied function.

        ‘constant’ (default)
        Pads with a constant value.

        ‘edge’
        Pads with the edge values of array.

        ‘linear_ramp’
        Pads with the linear ramp between end_value and the array edge value.

        ‘maximum’
        Pads with the maximum value of all or part of the vector along each axis.

        ‘mean’
        Pads with the mean value of all or part of the vector along each axis.

        ‘median’
        Pads with the median value of all or part of the vector along each axis.

        ‘minimum’
        Pads with the minimum value of all or part of the vector along each axis.

        ‘reflect’
        Pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.

        ‘symmetric’
        Pads with the reflection of the vector mirrored along the edge of the array.

        ‘wrap’
        Pads with the wrap of the vector along the axis. The first values are used to pad the end and the end values are used to pad the beginning.

        ‘empty’
        Pads with undefined values.

    Returns
    -------
    Transfer function H of float type
    """

    # Get image and kernel dimensions
    img_dim = img.shape
    krnl_dim = H.shape

    #compute origin shift
    orig_shft = ((krnl_dim[0] - img_dim[0]) // 2, (krnl_dim[1] - img_dim[1]) // 2)

    # check that dimensions will match
    x_delta = 0
    y_delta = 0

    if (2 * orig_shft[0] + img_dim[0]) - krnl_dim[0] != 0:
        x_delta = int(krnl_dim[0] - (2 * orig_shft[0] + img_dim[0]))
    if (2 * orig_shft[1] + img_dim[1]) - krnl_dim[1] != 0:
        y_delta = int(krnl_dim[1] - (2 * orig_shft[1] + img_dim[1]))


    # Get padding dimensions from padded_size
    #padded_dim = padded_size({"img_dim": img_dim, "krnl_dim": krnl_dim, "pwr2": True})

    # Pad img to the size of the transfer function, using the default or the specified pad_method
    #img_padded = np.pad(img, ((krnl_dim[0] - img_dim[0]) // 2, (krnl_dim[1] - img_dim[1]) // 2),
    #                    mode=pad_method)
    img_padded = np.pad(img, ((orig_shft[0], orig_shft[0] + x_delta), (orig_shft[1], orig_shft[1] + y_delta)), mode=pad_method)

    # Compute the FFT of the input image
    F = scipy.fft.fft2(img_padded)

    # Perform filtering. The resulting image is back in the spatial domain
    g = scipy.fft.ifft2(F * H)

    # Crop to the original size
    if img_dim == krnl_dim:
        g = skimage.img_as_float(np.real(g))
    else:
        g = skimage.img_as_float(np.real(g[orig_shft[0] - 1:orig_shft[0] + img_dim[0] - 1, orig_shft[1] - 1:orig_shft[1] + img_dim[1] - 1]))
    #g = skimage.img_as_float(np.real(g))

    return g


def lp_filter(flt_type, M, N, D0, n=1):
    """
    Function to compute lowpass filter frequency domain transfer functions H_LP(u,v) of size
    M x N with cutoff frequency (std dev) at D0.

    Adapted from Digital Image Processing Using Matlab, 3rd ed. by Gonzalez et al.

    Parameters
    ----------
    flt_type : string
        Valid values are:
        - 'ideal' for brickwall filter with cutoff at D0
        - 'butterwoth'for Butterworth transfer function
        - 'gaussian' for Gaussian LPF transfer function

    M : int
        Vertical dimension (e.g., M x N) of filter transfer function
    N : int
        Horizontal dimension (e.g., M x N) of filter transfer function
    D0 : float or int
        Cutoff frequency (std dev). Must be positive
    Returns
    -------
    Transfer function H of float type
    """
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


def hp_filter(flt_type, M, N, D0, n=1):
    """
    Function to compute highpass filter frequency domain transfer functions H_LP(u,v) of size
    M x N with cutoff frequency (std dev) at D0.

    Adapted from Digital Image Processing Using Matlab, 3rd ed. by Gonzalez et al.

    Parameters
    ----------
    flt_type : string
        Valid values are:
        - 'ideal' for brickwall filter with cutoff at D0
        - 'butterwoth'for Butterworth transfer function
        - 'gaussian' for Gaussian LPF transfer function

    M : int
        Vertical dimension (e.g., M x N) of filter transfer function
    N : int
        Horizontal dimension (e.g., M x N) of filter transfer function
    D0 : float or int
        Cutoff frequency (std dev). Must be positive
    Returns
    -------
    Transfer function H of float type
    """
    # Generate highpass filter transfer function from the lowpass representation
    H = 1.0 - lp_filter(flt_type, M, N, D0, n)

    return H

def lp_gaussian(M, N, D0):
    """
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
    """
    # generate the meshgrid from the input dimensions using function dftuv
    U, V = dftuv(M, N)

    # compute the distances from the origin to each point in the grid
    D = np.sqrt(U ** 2 + V ** 2)

    # compute Gaussian transer function
    H = np.exp(-np.power(D, 2) / (2 * np.power(D0, 2)))

    return H


def lp_butterworth(M, N, D0, n):
    """
    Function to return a frequency domain transfer function H(u,v) of size
    M x N for a lowpass Butterworth filter with cutoff frequency D0 and order.

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
    """
    # Create meshgrid
    U, V = dftuv(M, N)

    # compute the distances in the meshgrid
    D = np.hypot(U, V)

    # Compute xfr function
    H = 1 / (1 + (D / D0) ** (2 * n))

    return H


def bandfilter(params_dict):

    # extract parameters from the params_dict input
    if 'n' not in params_dict.keys():
        n = 1
    else:
        n = params_dict['n']

    if 'W' and 'C0' not in params_dict.keys():
        raise Exception("No W or C0 parameters specified")

    W = params_dict['W']
    C0 = params_dict['C0']

    # Use dftuv to set up the meshgrid arrays
    U, V = dftuv(params_dict['M'], params_dict['N'])

    # compute the distances in the meshgrid
    D = np.hypot(U,V)

    # Go through different cases for the three different filter types
    if params_dict['type'] == 'ideal':
        H = ideal_reject(D, C0, W)
    elif params_dict['type'] == 'butterworth':
        H = bttr_reject(D, C0, W, n)
    elif params_dict['type'] == 'gaussian':
        H = gauss_reject(D, C0, W)
    else:
        raise Exception("Unknown filter type")

    # convert to bandpass if specified
    if params_dict['bandpass_flag'] == True:
        H = 1 - H

    return H


def ideal_reject(D, C0, W):

    # compute points inside the inner boundary of the reject band
    RI = D <= C0 - (W/2)

    # compute points inside the inner boundary of the reject band
    RO = D >= C0 + (W/2)

    # ideal bandreject transfer function
    H = np.logical_or(RI, RO).astype(np.float)

    return H


def bttr_reject(D, C0, W, n):
    H = 1 / (1 + ((D*W)/(D**2 - C0**2 + 1e-6))**(2*n))

    return H


def gauss_reject(D, C0, W):
    H = 1 - np.exp(-((D**2 - C0**2)/(D*W + 1e-8))**2)
    #H = 1 - exp(-((D. ^ 2 - C0 ^ 2). / (D. * W + eps)). ^ 2);
    return H

#######################################
# IMAGE RESTORATION AND RECONSTRUCTION
#######################################


def imnoise(img, params_dict):
    """
    Function to add noise to an input image.

    Adapted from Digital Image Processing Using Matlab, 3rd ed. by Gonzalez et al.

    Parameters
    ----------
    img : M x N ndarray-like image

    params_dict : dict
        Valid keys-value pairs:
            type : string describing filter type. Valid types are uniform, gaussian, salt_pepper, lognormal, rayleigh,
                   exponential, and erlang
            a : float defining distribution center
            b : float defining distribution scale

    Returns
    -------
    img_noisy: array-like image corrupted by additive noise
    noise: array-like with additive noise that corrupted the image
    """

    # convert image to float
    img = skimage.img_as_float(img)

    # get img size
    M, N = img.shape

    # convert noise type to lower case to protect against uppercase
    type = params_dict['type'].lower()

    # extract a and b parameters
    if 'a' in params_dict.keys():
        a = params_dict['a']

    if 'b' in params_dict.keys():
        b = params_dict['b']

    # go through cases
    if type == 'uniform':
        if 'a' not in params_dict.keys():
            a = 0
        if 'b' not in params_dict.keys():
            b = 1

        noise = a + (b - a) * np.random.rand(M, N)

        img_noisy = img + noise

    elif type == 'gaussian':
        if 'a' not in params_dict.keys():
            a = 0
        if 'b' not in params_dict.keys():
            b = 1

        noise = a + b * np.random.randn(M, N)

        img_noisy = img + noise

    elif type == 'salt_pepper':
        if 'a' not in params_dict.keys():
            a = 0.05
        if 'b' not in params_dict.keys():
            b = 0.05

        noise = salt_pepper(M, N, a, b)

        img_noisy = img

        # set the values of 'salt' to 1
        img_noisy[noise == 1] = 1

        # set the values of 'pepper' to 0
        img_noisy[noise == 0] = 0

    elif type == 'lognormal':
        if 'a' not in params_dict.keys():
            a = 1
        if 'b' not in params_dict.keys():
            b = 0.25

        noise = np.exp(b*np.random.randn(M, N) + a)

        img_noisy = img + noise

    elif type == 'rayleigh':
        if 'a' not in params_dict.keys():
            a = 0
        if 'b' not in params_dict.keys():
            b = 1

        noise = a + (-b*np.log(1 - np.random.rand(M, N)))**(0.5)

        img_noisy = img + noise

    elif type == 'exponential':
        if 'a' not in params_dict.keys():
            a = 1

        noise = np.random.exponential(scale=a, size=(M, N))

        img_noisy = img + noise

    elif type == 'erlang':
        if 'a' not in params_dict.keys():
            a = 2
        if 'b' not in params_dict.keys():
            b = 5

        noise = erlang(M, N, a, b)

        img_noisy = img + noise
    else:
        raise Exception("Unknown distribution type")

    # scale image to full range and return
    img_noisy = img_noisy/np.max(img_noisy)

    return img_noisy, noise


def salt_pepper(M, N, a, b):
    """
    Function to generate salt and pepper noise

    Adapted from Digital Image Processing Using Matlab, 3rd ed. by Gonzalez et al.

    Parameters
    ----------
    M : int
        Vertical dimension (e.g., M x N) of filter transfer function
    N : int
        Horizontal dimension (e.g., M x N) of filter transfer function
    a : float
        Defining distribution center
    b : float
        Defining distribution scale

    Returns
    -------
    sp_arr: array-like with salt and pepper additive noise
    """

    # check to make sure that probabilities are valid
    if (a + b) > 1:
        raise Exception("The sum (a + b) must not exceed 1")

    # generate array to populate with salt and pepper noise
    sp_arr = 0.5*np.ones((M, N))

    # Generate an array of uniformly distributed random numbers in the range (0,1). Then Pp*(M*N) of them will have
    # values <= b. We set these coordinates points to 0. Similarly, Ps*(M*N) points will have values in the range
    # > b and <= (a + b). These are set to 1

    X = np.random.rand(M, N)

    # get pepper mask
    pepper_mask = np.where(X <= b, True, False)

    # set values to zero
    sp_arr[pepper_mask] = 0

    # get salt mask
    salt_mask = np.where(np.logical_and(X > b, X < (a + b)), True, False)

    # set values to one
    sp_arr[salt_mask] = 1

    return sp_arr

#test_sp = salt_pepper(100, 100, 0.05, 0.05)


def erlang(M, N, a, b):
    """
    Function to generate Erlang noise

    Adapted from Digital Image Processing Using Matlab, 3rd ed. by Gonzalez et al.

    Parameters
    ----------
    M : int
        Vertical dimension (e.g., M x N) of filter transfer function
    N : int
        Horizontal dimension (e.g., M x N) of filter transfer function
    a : float
        Defining distribution center
    b : float
        Defining distribution scale

    Returns
    -------
    R: array-like with Erlang additive noise
    """

    # check that b is a positive integer
    if b != np.round(b) or b <= 0:
        raise Exception("Parameter b must be a positive integer for Erlang")

    k = -1/a
    R = np.zeros((M, N))

    for j in range(b):
        R = R + k*np.log(1 - np.random.rand(M, N))

    return R

#test_erlang = erlang(100000,1,0.05,4)

def first_second_moment(img):
    # save original image type

    #convert image to uint8 in case it is not already of uint8 type
    img = skimage.img_as_ubyte(img)

    # compute normalized image histogram with 256 bins
    bins = np.arange(0, 257, 1)
    hist, hist_ctrs = np.histogram(img, bins=bins, density=True)

    # computing moments
    mean = np.sum(bins[:-1] * hist)
    second_moment = np.sum(np.power((bins[:-1] - mean), 2) * hist)

    return mean, second_moment


def spatial_fitler(img, params_dict):

    #convert image to float
    img = skimage.img_as_float(img)

    # define default values
    m = 3
    n = 3
    q = 1.5
    d = 2

    if m != n:
        raise Exception("Kernel dimensions mismatch")

    if 'm' in params_dict.keys():
        m = params_dict["m"]
    if 'n' in params_dict.keys():
        n = params_dict["n"]
    if 'q' in params_dict.keys():
        q = params_dict["q"]
    if 'd' in params_dict.keys():
        d = params_dict["d"]



    # arithmetic mean
    if params_dict["type"] == "amean":
        # skimage has a non-local means denoising function. We can take advantage of that function here.
        # alternatively, we can convolve our image with a kernel of ones of size m x n divided by m * n
        img_flt = skimage.restoration.denoise_nl_means(img, patch_size=m)

    if params_dict["type"] == "gmean":
        img_flt = geo_mean(img, m, n)

    if params_dict["type"] == "hmean":
        img_flt = harmonic_mean(img, m, n)

    if params_dict["type"] == "chmean":
        img_flt = contra_harmonic_filter(img, m, n, q, d)

    if params_dict["type"] == "max":
        img_flt = skimage.filters.rank.maximum(skimage.img_as_ubyte(img), selem=np.ones((m, n)))

    if params_dict["type"] == "min":
        img_flt = skimage.filters.rank.minimum(skimage.img_as_ubyte(img), selem=np.ones((m, n)))

    return img_flt


def geo_mean(img, m, n):
    # convert image to float
    img = skimage.img_as_float(img)

    img_flt = np.exp(convolve(np.log(img + 1e-10), np.ones((m, n), dtype=np.float), mode='reflect')) ** (1/(m*n))

    return img_flt


def harmonic_mean(img, m, n):
    # conver image to float
    img = skimage.img_as_float(img)

    # convolve image
    img_flt = (m * n) / convolve(1 / (img + 1e-10), np.ones((m, n)), mode='reflect')

    return img_flt


def contra_harmonic_filter(img, m, n, q, d):
    # convert image to float
    img = skimage.img_as_float(img)
    #img = img / np.max(img)

    # filter image
    img_flt = convolve(img ** (q + 1), np.ones((m, n)), mode='nearest')
    img_flt = img_flt / (convolve((img + 1e-10) ** q, np.ones((m, n)), mode='nearest') + 1e-10)

    #f = imfilter(g. ^ (q + 1), ones(m, n), 'replicate');
    #f = f. / (imfilter(g. ^ q, ones(m, n), 'replicate') + eps);

    return img_flt


def get_motion_kernel(length, angle):

    # Some references: https://www.mathworks.com/help/images/ref/fspecial.html#d122e72691
    # https://stackoverflow.com/questions/40305933/how-to-add-motion-blur-to-numpy-array
    # I use skimage instead of OpenCV to get the rotated kernel

    # create kernel of size (len, len) with ideal line segment along the middle row (e.g., len // 2 --> int division)
    krnl = np.zeros((length, length))

    # insert ideal line segment
    krnl[length // 2, :] = np.ones(length)

    # normalize the kernel by the size dim
    krnl = krnl / length
    #krnl = krnl / np.sum(krnl)

    # rotate krnl by specified angle
    krnl_rotated = skimage.transform.rotate(krnl, angle=angle)

    return krnl_rotated


def pad_to_same_size(img, krnl):

    # get img and kernel dimensions
    img_dim = img.shape
    krnl_dim = krnl.shape

    #compute pad size
    x_pad = (img_dim[0] - krnl_dim[0]) // 2
    y_pad = (img_dim[1] - krnl_dim[1]) // 2

    #protecting against odd image dimensions
    x_delta = 0
    y_delta = 0
    if (2 * x_pad + krnl_dim[0]) - img_dim[0]  != 0:
        x_delta = int(img_dim[0] - (2 * x_pad + krnl_dim[0]))
    if (2 * y_pad + krnl_dim[1]) - img_dim[1] != 0:
        y_delta = int(img_dim[1] - (2 * y_pad + krnl_dim[1]))

    krnl_pad = np.pad(krnl, [(x_pad, x_pad + x_delta), (y_pad, y_pad + y_delta)], 'constant')

    return krnl_pad


def custom_wiener_filt(img, psf, nspr):

    # compute the FFT of the input image
    img_fft = scipy.fft.fft2(img)

    # compute the Fourier transform of the kernel

    # first pad the kernel
    krnl_pad = pad_to_same_size(img, psf)

    # FFT the kernel
    krnl_fft = scipy.fft.fft2(krnl_pad)

    # compute inverse filter xfr function (eqn. 5-14 from DIPUM)
    krnl_pwr_spec = np.abs(krnl_fft)

    # compute the wiener filter
    wiener_filt = (krnl_pwr_spec / (krnl_pwr_spec + nspr)) * (1 / krnl_fft)

    # filter the image
    img_f_hat = wiener_filt * img_fft

    # compute the IFFT
    restored_ifft = skimage.img_as_float(np.real(scipy.fft.ifft2(img_f_hat)))

    # ifftshift
    restored_ifft = scipy.fft.ifftshift(restored_ifft)

    return restored_ifft


######################################
# COLOR IMAGE PROCESSING
######################################

def my_rgb2xyz(arr):
    """
    Function to convert from RGB to XYZ
    @param arr: ndarray-like with shape P x 3 with RGB coordinate points
    @return: ndarray-like with shape P x 3 with XYZ coordinate points
    """

    assert arr.shape[-1] == 3

    # define transformation matrix
    rgb_to_xyz_xfm = np.array([[2.768892, 1.751748, 1.130160],
                               [1.000000, 4.590700, 0.060100],
                               [0.000000, 0.056508, 5.594292]])

    # compute transformation
    xyz_arr = (rgb_to_xyz_xfm @ arr.T).T

    return xyz_arr


def my_xyz2rgb(arr):
    """
    Function to convert from XYZ RGB
    @param arr: ndarray-like with shape P x 3 with XYZ coordinate points
    @return: ndarray-like with shape P x 3 with RGB coordinate points
    """

    assert arr.shape[-1] == 3

    # define transformation matrix
    rgb_to_xyz_xfm = np.array([[2.768892, 1.751748, 1.130160],
                               [1.000000, 4.590700, 0.060100],
                               [0.000000, 0.056508, 5.594292]])

    xyz_to_rgb_xfm = np.linalg.inv(rgb_to_xyz_xfm)

    # compute transformation
    #xyz_arr = (rgb_to_xyz_xfm @ arr.T).T
    rgb_arr = (xyz_to_rgb_xfm @ arr.T).T

    return rgb_arr


def my_rgb2hsi(rgb):

    # convert image to float
    img_float = skimage.img_as_float(rgb)

    # extract rgb bands
    red = img_float[:, :, 0]
    green = img_float[:, :, 1]
    blue = img_float[:, :, 2]

    # implement the conversion equations
    numer = 0.5 * ((red - green) + (red - blue))
    denom = np.sqrt((red - green)**2 + (red - blue)*((green - blue)))

    theta = np.arccos(numer / (denom + 1e-10))

    H = np.where(blue > green, 2*np.pi - theta, theta)
    H = H / (2 * np.pi)

    numer = np.minimum(np.minimum(red, green), blue)
    denom = red + green + blue

    denom = np.where(denom == 0, 1e-10, denom)

    S = 1 - (3 * numer) / denom

    H = np.where(S == 0, 0, H)

    I = (red + green + blue) / 3

    # concatenate
    hsi = np.dstack((H, S, I))

    return hsi


def my_hsi2rgb(hsi):
    # convert image to float
    hsi = skimage.img_as_float(hsi)

    # extract the individual HSI component images
    H = hsi[:, :, 0] * 2 * np.pi
    S = hsi[:, :, 1]
    I = hsi[:, :, 2]

    # Implement the conversion equations
    red = np.zeros((hsi.shape[0], hsi.shape[1]))
    green = np.zeros((hsi.shape[0], hsi.shape[1]))
    blue = np.zeros((hsi.shape[0], hsi.shape[1]))

    # red-green sector (0 <= H < 2*pi/3)
    idx_mask = np.where(((0 <= H) & (H < (2 * np.pi) / 3)), True, False)
    blue[idx_mask] = I[idx_mask] * (1.0 - S[idx_mask])
    red[idx_mask] = I[idx_mask] * (1.0 + (S[idx_mask] * np.cos(H[idx_mask])) / np.cos((np.pi/3) - H[idx_mask]))
    green[idx_mask] = 3 * I[idx_mask] - (red[idx_mask] + blue[idx_mask])

    # blue-green sector
    idx_mask = np.where((2 * (np.pi/3) <= H) & (H < 4 * (np.pi/3)), True, False)
    red[idx_mask] = I[idx_mask] * (1 - S[idx_mask])
    green[idx_mask] = I[idx_mask] * (1 + S[idx_mask] * np.cos(H[idx_mask] - (2 * (np.pi/3))) / np.cos(np.pi - H[idx_mask]))
    blue[idx_mask] = 3 * I[idx_mask] - (red[idx_mask] + green[idx_mask])

    # blue-red sector
    idx_mask = np.where((4 * (np.pi/3) <= H) & (H < 2 * np.pi), True, False)
    green[idx_mask] = I[idx_mask] * (1 - S[idx_mask])
    blue[idx_mask] = I[idx_mask] * (1 + S[idx_mask] * np.cos(H[idx_mask] - 4 * (np.pi/3)) / np.cos(5 * (np.pi/3) - H[idx_mask]))
    red[idx_mask] = 3 * I[idx_mask] - (green[idx_mask] + blue[idx_mask])

    # concatenate
    rgb = np.dstack((red, green, blue))
    rgb = np.maximum(np.minimum(rgb, 1), 0)

    return rgb


def spatial_filt_3d(img, krnl, conv_mode):
    """
    Function to perform spatial filtering for 3D images given a filter kernel
    @param img: 3D ndarray-like image
    @param krnl: 2D ndarray-like filter kernel
    @param conv_mode: mode parameter for correlation/convolution that determines how the image is extended beyond
                      boundaries
    @return: 3D ndarray-like filtered image
    """
    # convert image to float
    img = skimage.img_as_float(img)

    # convolve image
    chan_1_conv = correlate(img[:, :, 0], krnl, mode=conv_mode)
    chan_2_conv = correlate(img[:, :, 1], krnl, mode=conv_mode)
    chan_3_conv = correlate(img[:, :, 2], krnl, mode=conv_mode)

    # concatenate filtered image
    flt_img = np.dstack((chan_1_conv, chan_2_conv, chan_3_conv))

    return flt_img


def normalize_zero_one(img):
    """
    Function to normalize intensity values between [0,1]
    @param img: ndarray-like image
    @return: ndarray-like image normalized between 0 and 1
    """
    # convert image to float
    img = skimage.img_as_float(img)

    # normalize
    img_normalized = (img - img.min()) / (img.max() - img.min())

    return img_normalized


def colorgrad(img, T=0):
    """
    Function to compute the gradient of a color image. Adapted from DIPUM 3rd edition - Toolbox
    @param img:3D ndarray-like input image
    @param T: threshold value, defaults to 0
    @return: ndarray-like with vector_grad (gradient computed directly in RGB vector space)
             max_dir_arr: ndarray-like with the maximum gradient direction per pixel
             ppg: ndarray-like with the gradient computed on per-plane or per-channel basis
    """
    # convert image to float
    skimage.img_as_float(img)

    # compute the gradients
    red_x = skimage.filters.sobel(img[:, :, 0], axis=0, mode='reflect')
    red_y = skimage.filters.sobel(img[:, :, 0], axis=1, mode='reflect')

    green_x = skimage.filters.sobel(img[:, :, 1], axis=0, mode='reflect')
    green_y = skimage.filters.sobel(img[:, :, 1], axis=1, mode='reflect')

    blue_x = skimage.filters.sobel(img[:, :, 2], axis=0, mode='reflect')
    blue_y = skimage.filters.sobel(img[:, :, 2], axis=1, mode='reflect')

    # compute the parameters of the vector gradient
    g_xx = np.power(red_x, 2) + np.power(green_x, 2) + np.power(blue_x, 2)
    g_yy = np.power(red_y, 2) + np.power(green_y, 2) + np.power(blue_y, 2)
    g_xy = red_x * red_y + green_x * green_y + blue_x * blue_y

    # compute gradient direction
    max_dir = 0.5 * np.arctan( (2 * g_xy) / (g_xx - g_yy + 1e-10))

    # compute the square of the magnitude
    grad_mag = 0.5 * ((g_xx + g_yy) + (g_xx - g_yy) * np.cos(2 * max_dir) + 2 * g_xy * np.sin(2 * max_dir))

    # now repeat for angle + pi/2 and select the maximum at each point
    max_dir_2 = max_dir + (np.pi / 2)

    grad_mag_2 = 0.5 * ((g_xx + g_yy) + (g_xx - g_yy) * np.cos(2 * max_dir_2) + 2 * g_xy * np.sin(2 * max_dir_2))

    grad_mag = np.sqrt(grad_mag + 1e-10)
    grad_mag_2 = np.sqrt(grad_mag_2 + 1e-10)

    # create the gradient image by picking the maximum between the two and normalize between [0,1]
    vector_grad = normalize_zero_one(np.maximum(grad_mag, grad_mag_2))

    # select the corresponding angles. Where grad_mag_2 > grad_mag, pick values from max_dir + pi/2
    max_dir_arr = np.where(max_dir_2 > max_dir, max_dir_2 + (np.pi / 2), max_dir)

    # compute the per-plane gradients
    red_grad = np.hypot(red_x, red_y)
    green_grad = np.hypot(green_x, green_y)
    blue_grad = np.hypot(blue_x, blue_y)

    # form the composite per plane gradient
    ppg = normalize_zero_one(red_grad + green_grad + blue_grad)

    if T != 0:
        vector_grad = vector_grad > T
        ppg = ppg > T

    return vector_grad, max_dir_arr, ppg

def color_space_conv(img, method):
    """
    Function to perform color conversion specified by method.
    @param img: ndarray like color image
    @param method: color space conversion method. Valid arguments are:
                - rgb2cmy
                - cmy2rgb
                - rgb2cmyk
                - cmyk2rgb
    @return:
    """

    # assert that we have a color image
    if img.shape[-1] < 3:
        raise Exception("Not a color image")

    # convert image to float and normalize, just in case
    img = normalize_zero_one(img)

    # step through different methods

    # protect against uppercase
    method = method.lower()

    if method == 'rgb2cmy':
        out_img = np.ones(img.shape) - img

    elif method == 'cmy2rgb':
        out_img = np.ones(img.shape) - img

    elif method == 'rgb2cmyk':
        # first convert to cmy space
        cmy_img = np.ones(img.shape) - img

        # compute K
        K = np.minimum(np.minimum(cmy_img[:, :, 0], cmy_img[:, :, 1]), cmy_img[:, :, 2])

        K_masked = np.ma.masked_array(K, K == 1)

        # denom = np.ones(img.shape[0:2]) - K
        denom = np.ones(img.shape[0:2]) - K_masked

        # find instances where C = 0, M = 0, and Y = 0 and set K = 1 there
        #zero_mask = (cmy_img[:, :, 0] == 0) & (cmy_img[:, :, 1] == 0) & (cmy_img[:, :, 2] == 0)
        #K[zero_mask] = 1

        # otherwise, compute the C, M, K components
        #K_masked = np.ma.masked_array(K, zero_mask)

        denom = np.ones(img.shape[0:2]) - K_masked
        #denom = np.ones(img.shape[0:2]) - K

        C = ((cmy_img[:, :, 0] - K) / denom).filled(fill_value=0)
        M = ((cmy_img[:, :, 1] - K) / denom).filled(fill_value=0)
        Y = ((cmy_img[:, :, 2] - K) / denom).filled(fill_value=0)

        #C = (cmy_img[:, :, 0] - K) / denom
        #M = (cmy_img[:, :, 1] - K) / denom
        #Y = (cmy_img[:, :, 2] - K) / denom

        # Set true black pixels
        #C = np.where(K == 1, 0, C)
        #M = np.where(K == 1, 0, M)
        #Y = np.where(K == 1, 0, Y)

        # compose the CMYK image
        out_img = np.dstack((C, M, Y, K))

    elif method == 'cmyk2rgb':
        # first convert cmyk to cmy
        C = img[:, :, 0] * (np.ones(img[:,:,0].shape) - img[:, :, -1]) + img[:, :, -1]
        M = img[:, :, 1] * (np.ones(img[:,:,1].shape) - img[:, :, -1]) + img[:, :, -1]
        Y = img[:, :, 2] * (np.ones(img[:,:,1].shape) - img[:, :, -1]) + img[:, :, -1]

        # recompose CMY image
        cmy_img = np.dstack((C, M, Y))

        # convert to rgb
        out_img = np.ones(cmy_img.shape) - cmy_img

    return out_img


#################################
# MORPHOLOGICAL IMAGE PROCESSING
#################################

def my_morph_recost(init_marker, mask, strel):
    """
    Function to compute the morphological reconstruction given an initial marker image, a mask, and a structuring
    element. The algorithm is defined in Section 10.5 of DIPUM, 3rd edition

    @param init_marker: ndarray-like with initial marker image to start the reconstruction process
    @param mask: ndarray-like with initial mask image to reconstruct
    @param strel: ndarray-like with initial structuring element
    @return h: ndarray-like reconstructed image
    """

    # convert image to boolean
    img = skimage.img_as_bool(mask)

    # initialize marker
    h = skimage.img_as_bool(init_marker)

    flag = True
    while flag:

        # compute dilation of marker image
        dil = morphology.dilation(h, strel)

        # compute reconstruction by dilation
        h_k_1 = np.logical_and(dil, img)

        # repeat until h_{k+1} == h_{k}
        if np.sum(np.abs(skimage.img_as_ubyte(h_k_1) - skimage.img_as_ubyte(h))) == 0:
            flag = False
        else:
            h = h_k_1

    return h


def custom_hole_fill(img):
    """
    Function to fill in holes in an image using morphological reconstruction.
    The algorithm is defined in Section 10.5 of DIPUM, 3rd edition

    @param img: ndarray-like with image to fill holes for
    @return holes_fill: ndarray-like image with filled holes
    @return holes: ndarray-like image with holes
    """
    # convert image to boolean
    img = skimage.img_as_bool(img)

    # compute image complement. We will need it later
    img_compl = np.logical_not(img)

    # structuring element
    strel = np.array([[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]], dtype=np.uint8)

    # generate initial marker image
    marker = np.zeros(img.shape)

    # top border
    marker[0, :] = 1 - img[0, :]

    # bottom border
    marker[-1, :] = 1 - img[-1, :]

    # left border
    marker[:, 0] = 1 - img[:, 0]

    # right border
    marker[:, -1] = 1 - img[:, -1]

    marker = skimage.img_as_bool(marker)

    flag = True
    while flag:

        # dilate the marker image with structuring element
        marker_dil = morphology.dilation(marker, strel)

        # compute reconstruction by dilation
        holes_fill = np.logical_and(marker_dil, img_compl)

        # repeat until image does not change anymore
        if np.sum(np.abs(skimage.img_as_ubyte(holes_fill) - skimage.img_as_ubyte(marker))) == 0:
            flag = False
        else:
            marker = holes_fill

    # fill hole
    hole = np.logical_and(np.logical_not(holes_fill), img_compl)

    return np.logical_not(holes_fill), hole


def custom_hit_or_miss(img, b1_strel, b2_strel):
    """
    Function to compute morphological hit or miss transform for pattern matching
    @param img: ndarray like boolean image
    @param b1_strel: ndaray like structuring element with pattern to match in foreground
    @param b2_strel: ndaray like structuring element with pattern to match in background
    @return: ndarray like boolean image with hit or miss transform
    """
    # convert image to boolean
    img = skimage.img_as_bool(img)

    # compute complement
    img_compl = np.logical_not(img)

    # Compute the hit or miss transform
    hmt_foreground = morphology.binary_erosion(img, selem=b1_strel)
    hmt_background = morphology.binary_erosion(img_compl, selem=b2_strel)
    hmt = np.logical_and(hmt_foreground, hmt_background)

    return hmt


def custom_morph_boundary_detection(img, strel):
    """
    Function to extract the boundary of a binary image
    @param img: ndarray-like binary image
    @param strel: ndarray-like structuring element used to extract boundary
    @return: ndarray-like binary image with boundary
    """
    # convert image to uint8 type since we will be doing mathematical operations on it
    img = skimage.img_as_ubyte(img)

    img_erode = skimage.img_as_ubyte(morphology.erosion(img, selem=strel))

    # extract boundary
    boundary = skimage.img_as_bool(img - img_erode)

    return boundary