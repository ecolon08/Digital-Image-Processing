# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 10:09:02 2021

@author: Ernesto

K-Means Clustering Sandbox
"""

# import libraries
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import skimage
from matplotlib import colors

# load sample image
img_yeast = io.imread('./images/yeast_USC.tif')
img_book = io.imread('./images/book-cover.tif')
img_flowers = io.imread('./images/flowers-red.tif')
img_chalk = io.imread('./images/chalk-RGB.tif')
img_square = io.imread('./images/RGB-blue-yellow-squares.tif')

#img = img_yeast
#img = img_book
#img = img_flowers
#img = img_chalk
img = img_square

# specify number of means
k = 4


# Step 1) Initialize the algorithm. Specify an initial set of means
means = np.random.choice(np.unique(img.ravel()), size=k)
#means = np.array([45, 120, 181])


idx_x = np.random.uniform(low=0, high=img.shape[1], size=3).astype(int)
idx_y = np.random.uniform(low=0, high=img.shape[2], size=3).astype(int)



if len(img.shape) > 2:
    # generate idx to get pixels from
    smallest_dim = np.sort(img.shape)[1]
    #idx = np.random.uniform(low=0, high=smallest_dim, size=(k,2)).astype(int)

    idx_x = np.random.uniform(low=0, high=img.shape[0], size=k).astype(int)
    idx_y = np.random.uniform(low=0, high=img.shape[1], size=k).astype(int)

    #idx = np.expand_dims(idx, axis=-1)
    #means_red = np.random.choice(np.unique(img[:, :, 0].ravel()), size=k).reshape(-1, 1)
    #means_green = np.random.choice(np.unique(img[:, :, 1].ravel()), size=k).reshape(-1, 1)
    #means_blue = np.random.choice(np.unique(img[:, :, 2].ravel()), size=k).reshape(-1, 1)
    
    #means = np.dstack((means_red, means_green, means_blue))
    means = img[idx_x, idx_y, :]
    
    #means = np.expand_dims(means, axis=1)
    #red_idx = np.tile(np.array([True, False, False]), reps=k)
    #green_idx = np.tile(np.array([False, True, False]), reps=k)
    #blue_idx = np.tile(np.array([False, False, True]), reps=k)
    
else:
    means = np.random.choice(np.unique(img.ravel()), size=k)


num_iter = 15
means_delta = list()
means_list = list()
means_list.append(means.copy())

for i in range(num_iter):
#old_means = means
    
    # I want to vectorize the algorithm as much as possible, create mean array stacks
    means_stack = list()    
    
    if len(img.shape) > 2:
        
        for mean in range(k):
            #means_stack.append(means[mean] * np.ones(img.shape))
            #means_stack.append(np.expand_dims(means[:, :, mean], axis=0) * np.ones(img.shape))
            means_stack.append(np.expand_dims(means[mean, :, :], axis=0) * np.ones(img.shape))
    
        means_stack = np.dstack(means_stack)    
        
        norms_list = list()
        norms_stack = (np.tile(img, reps=k) - means_stack) ** 2
        
        for mean in range(k):
            # generate indices
            idx = np.arange(3) + mean
            norms_list.append(np.sum(norms_stack[:, :, idx], axis=-1))
        
        norms_stack = np.dstack(norms_list)
        
        clustered = np.argmin(norms_stack, axis = -1)
        
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
            means[cluster, :, :] = cluster_means.reshape(1, -1)
        
    
        #if means_delta[i] < 5:
        #    print("Soppted at Iter: ", i)
        #   break
            
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
    #norms_list = list()
    
    #for mean in range(k):
    #    norms_list = np.sqrt((img - means[mean])**2)
    
    #norms_stack = np.sqrt((np.expand_dims(img, axis=-1) - means_stack)**2)
  
    '''  
    if len(img.shape) > 2:
        norms_list = list()
        norms_stack = (np.tile(img, reps=k) - means_stack) ** 2
        
        for mean in range(k):
            # generate indices
            idx = np.arange(3) + mean
            norms_list.append(np.sum(norms_stack[:, :, idx], axis=-1))
        
        norms_stack = np.dstack(norms_list)
    else:
        norms_stack = (np.expand_dims(img, axis=-1) - means_stack)**2
    '''
    #test_1 = norms_stack[:, :, 0]
    #test_2 = np.sqrt((img - means_stack[:, :, 0])**2)
    #np.isclose(test_1, test_2)
    #print(np.all(np.isclose(test_1, test_2)))
    
    # find the argmin along the means axis of the stack (axis = -1)
    # clustered = np.argmin(norms_stack, axis=-1).astype(np.uint8)
 
    ''' 
    if len(img.shape) > 2:
    
        
        red_clustered = np.argmin(norms_stack[:, :, red_idx], axis=-1)
        green_clustered = np.argmin(norms_stack[:, :, green_idx], axis=-1)
        blue_clustered = np.argmin(norms_stack[:, :, blue_idx], axis=-1)
        
        clustered = np.dstack((red_clustered, green_clustered, blue_clustered))
        
        for cluster in range(k):
            # create masks for each mean
            mask = np.where(clustered == np.tile(cluster, reps=k), 1, 0)
        
            # update the means vector
            print(np.sum(img[mask.astype(bool)]), np.sum(mask))
            #means[cluster] = np.sum(img[mask]) / (np.sum(mask) + 1e-10)
            #means[cluster] = np.sum(img[mask.astype(bool)]) / (np.sum(mask) + 1e-10)
            #a = np.sum(img[mask.astype(bool)]) / (np.sum(mask) + 1e-10)
            cluster_ma = np.ma.masked_array(img, mask.astype(bool))
            
            # compute mean vector
            cluster_means = np.ma.mean(cluster_ma, axis = (0, 1))
            
            means[:, :, cluster] = cluster_means.reshape(1, -1)
    '''
    #clustered = np.argmin(norms_stack, axis = -1)      
    #else:
    #clustered = np.argmin(norms_stack, axis=-1)
    
    # Step 3) Update the cluster centers (means)
    # Now I need to cycle through the means
    
    #mask = np.where(clustered == 0,)
    #mask = np.where(clustered == 1, 1, 0)
    #print(np.sum(mask))    
    #print(np.unique(mask, return_counts=True))
    
    #test = np.sum(img[mask.astype(bool)])
    ''' 
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
        means[cluster, :, :] = cluster_means.reshape(1, -1)
    '''    
    means_list.append(means.copy())
    #means_delta.append(np.sqrt((means - old_means)**2))
    means_delta.append(np.sqrt(np.sum((means_list[i + 1] - means_list[i])**2)))
    
    if means_delta[i] < 5:
        print("Soppted at Iter: ", i)
        break
    

# update means
#means_in_clustered, counts_per_mean = np.unique(clustered, return_counts=True)

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