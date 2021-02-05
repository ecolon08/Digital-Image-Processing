# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 07:22:37 2021

@author: Ernesto

Test scripts for ec_img_utils
"""
import sys
sys.path.append('..')

import ec_img_utils
import numpy as np
from skimage import io

#testing padded_size()

#case 1 - no input arguments - should get an exception
#test_dict = {}
#out = ec_img_utils.padded_size(test_dict)
#Pass

#case 2 - only img_dim in params_dict
img_dim = [2,2]
out = ec_img_utils.padded_size({'img_dim': img_dim})
print(out)
print(type(out))

#case 3 - only img_dim passed and pwr2 argument passed
out = ec_img_utils.padded_size({'img_dim': img_dim, 'pwr2': True})
img_dim = np.array([5,7])
out = ec_img_utils.padded_size({'img_dim': img_dim, 'pwr2': True})
#passed

#case 4 - passing img_dim and krnl_dim
img_dim = np.array([5,6])
krnl_dim = np.array([3,3])
out = ec_img_utils.padded_size({'img_dim': img_dim, 'krnl_dim': krnl_dim})
#passed

#Case 5 - passing  passing img_dim and krnl_dim and pwr2
img_dim = np.array([5,6])
krnl_dim = np.array([3,3])
out = ec_img_utils.padded_size({'img_dim': img_dim, 'krnl_dim': krnl_dim, 'pwr2':True})
#passed

print(type((2,2)))


img_tst_pttrn = io.imread('./images/testpattern512.tif')
# Compute padding dimensions
tst_pttrn_pad_dim = ec_img_utils.padded_size({"img_dim": img_tst_pttrn.shape})

#Ideal LPI
D0 = 50
tst_HLPI = ec_img_utils.lp_filter('ideal', tst_pttrn_pad_dim[0], tst_pttrn_pad_dim[1], D0)

#filter the image
tst_LPI = ec_img_utils.dft_filt(img_tst_pttrn, tst_HLPI, 'constant')
