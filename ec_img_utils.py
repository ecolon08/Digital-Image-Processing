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

    # Get padding dimensions from padded_size
    #padded_dim = padded_size({"img_dim": img_dim, "krnl_dim": krnl_dim, "pwr2": True})

    # Pad img to the size of the transfer function, using the default or the specified pad_method
    #img_padded = np.pad(img, ((krnl_dim[0] - img_dim[0]) // 2, (krnl_dim[1] - img_dim[1]) // 2),
    #                    mode=pad_method)
    img_padded = np.pad(img, ((orig_shft[0], orig_shft[0]), (orig_shft[1], orig_shft[1])), mode=pad_method)

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

    return img_flt


def geo_mean(img, m, n):
    # convert image to float
    img = skimage.img_as_float(img)

    img_flt = np.exp(convolve(np.log(img), np.ones((m, n), dtype=np.float), mode='reflect')) ** (1/(m*n))

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
    img_flt = img_flt / (convolve(img ** q, np.ones((m, n)), mode='nearest') + 1e-10)

    #f = imfilter(g. ^ (q + 1), ones(m, n), 'replicate');
    #f = f. / (imfilter(g. ^ q, ones(m, n), 'replicate') + eps);

    return img_flt








































