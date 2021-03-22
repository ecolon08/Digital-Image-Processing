# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 10:59:49 2021

@author: Ernesto
"""

# import libraries
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import skimage
from matplotlib import colors

# load sample image
img_yeast = io.imread('./images/yeast_USC.tif')
#img_book = io.imread('./images/book-cover.tif')
#img_flowers = io.imread('./images/flowers-red.tif')
img_chalk = io.imread('./images/chalk-RGB.tif')
img_square = io.imread('./images/RGB-blue-yellow-squares.tif')
img_home = io.imread('./images/home.jpg')

img = img_yeast
#img = img_book
#img = img_flowers
#img = img_chalk
#img = img_square
#img = img_home

# specify number of means
k = 3


# Step 1) Initialize the algorithm. Specify an initial set of means
#means = np.random.choice(np.unique(img.ravel()), size=k)


if len(img.shape) > 2:
    
    unique_pix = np.unique(img.reshape(-1, img.shape[2]), axis=0)

    # generate idx to get pixels from
    #idx_x = np.random.uniform(low=0, high=img.shape[0], size=k).astype(int)
    #idx_y = np.random.uniform(low=0, high=img.shape[1], size=k).astype(int)
    
    
    #means = np.dstack((means_red, means_green, means_blue))
    #means = img[idx_x, idx_y, :]
    if unique_pix.shape[0] == k:
        means = unique_pix
    else:
        idx = np.random.randint(low=0, high=unique_pix.shape[0], size=k)
        means = unique_pix[idx, :]
    #means = np.random.choice(unique_pix, size=k)

else:
    means = np.random.choice(np.unique(img.ravel()), size=k) #.reshape(-1, 1)

#means = np.array([[0, 0, 229],
#                  [229, 229, 229],
#                  [0, 0, 0],
#                  [229, 229, 0]])


num_iter = 15
means_delta = list()
means_list = list()
means_list.append(means.copy())

for i in range(num_iter):
    
    # I want to vectorize the algorithm as much as possible, create mean array stacks
    means_stack = list()    
    
    if len(img.shape) > 2:
            
        for mean in range(k):
            #means_stack.append(means[mean] * np.ones(img.shape))
            #means_stack.append(np.expand_dims(means[:, :, mean], axis=0) * np.ones(img.shape))
            #means_stack.append(np.expand_dims(means[mean, :, :], axis=0) * np.ones(img.shape))
            means_stack.append(np.expand_dims(means[mean, :], axis=0) * np.ones(img.shape))
        
        means_stack = np.dstack(means_stack)    
        
        norms_list = list()
        norms_stack = (np.tile(img, reps=k) - means_stack) ** 2
        #norms_stack = (np.tile(np.expand_dims(img, axis=-1), reps=k) - means_stack) ** 2
        #test_lst = []
        
        #for mean in range(k):
        #    # generate indices
        #    idx = np.arange(3) + 3 * mean
        #    test_lst.append(idx)
        
        for mean in range(k):
            # generate indices
            idx = np.arange(3) + 3 * mean
            norms_list.append(np.sum(norms_stack[:, :, idx], axis=-1))
        
        norms_stack = np.dstack(norms_list)
        
        clustered = np.argmin(norms_stack, axis = -1)
        
        #plt.figure()
        #plt.imshow(clustered)
        
        #mask = np.where(clustered == 0, 1, 0)
        
        #mask = np.tile(np.expand_dims(mask,axis=-1), reps=3)
        
        #cluster_ma = np.ma.masked_array(img, mask.astype(bool))
        
        #cluster_means = np.ma.mean(cluster_ma, axis = (0, 1))
        
        #means[0, :] = cluster_means.reshape(1, -1)
        
        for cluster in range(k):
            # create masks for each mean
            
            #mask = np.where(clustered == cluster, 1, 0)
            mask = np.where(clustered == cluster, 1, 0)
            
            mask = np.tile(np.expand_dims(mask,axis=-1), reps=3)
        
            # update the means vector
            #print(np.sum(img[mask.astype(bool)]), np.sum(mask))
            #means[cluster] = np.sum(img[mask]) / (np.sum(mask) + 1e-10)
            #means[cluster] = np.sum(img[mask.astype(bool)]) / (np.sum(mask) + 1e-10)
            cluster_ma = np.ma.masked_array(img, mask.astype(bool))
                
            # compute mean vector
            cluster_means = np.ma.mean(cluster_ma, axis = (0, 1))
                
            #means[:, :, cluster] = cluster_means.reshape(1, -1)
            #means[cluster, :, :] = cluster_means.reshape(1, -1)
            means[cluster, :] = cluster_means.reshape(1, -1)
        
        
        
        means_list.append(means.copy())
        
        means_delta.append(np.sqrt(np.sum((means_list[i + 1] - means_list[i])**2)))

    else: 
        for mean in range(k):
            means_stack.append(means[mean] * np.ones(img.shape))
            #means_stack.append(np.expand_dims(means[:, :, mean], axis=0) * np.ones(img.shape))
    
        means_stack = np.dstack(means_stack)    
    
        norms_stack = (np.expand_dims(img, axis=-1) - means_stack)**2
        
        clustered = np.argmin(norms_stack, axis = -1) 
    
        for cluster in range(k):
            # create masks for each mean
            
            mask = np.where(clustered == cluster, 1, 0)
            #mask = np.where(clustered == cluster, 1, 0)
            
            #mask = np.tile(np.expand_dims(mask,axis=-1), reps=3)
        
            # update the means vector
            #print(np.sum(img[mask.astype(bool)]), np.sum(mask))
            #means[cluster] = np.sum(img[mask]) / (np.sum(mask) + 1e-10)
            means[cluster] = np.sum(img[mask.astype(bool)]) / (np.sum(mask) + 1e-10)
            
            #cluster_ma = np.ma.masked_array(img, mask.astype(bool))
                
            # compute mean vector
            #cluster_means = np.ma.mean(cluster_ma, axis = (0, 1))
                
            #means[:, :, cluster] = cluster_means.reshape(1, -1)
            #means[cluster, :, :] = cluster_means.reshape(1, -1)
        # Step 2) Assign samples to clusters: Assign each sample to the cluster set 
        # whose mean is the closest (ties are resolved arbitrarily, but samples are 
        # assigned to only one cluster)
        
        # first compute distance to clusters
        
        # Step 3) Update the cluster centers (means)
        # Now I need to cycle through the means
        
        means_list.append(means.copy())

        means_delta.append(np.sqrt(np.sum((means_list[i + 1] - means_list[i])**2)))
    
    if i > 0:
        if ((means_delta[i - 1] - means_delta[i]) == 0):
            print("Soppted at Iter: ", i)
            break

plt.figure()
plt.imshow(img)
plt.figure()
plt.plot(means_delta)
plt.figure()
plt.imshow(clustered)
plt.show()

#from matplotlib import cm
#color_arr = plt.cm.get_cmap('hsv', k)

#plt.figure()
#plt.imshow(img)
#plt.imshow(clustered, cmap=color_arr)

# assemble labeled image
#test = skimage.color.label2rgb(clustered, image=img, colors = ((1/255) * means).tolist())
test = skimage.color.label2rgb(clustered, image=img)
plt.figure()
plt.imshow(test)