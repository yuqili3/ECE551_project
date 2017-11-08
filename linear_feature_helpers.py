# Below is a bunch of helper functions

import numpy as np
from scipy.signal import convolve2d

# extract low resolution image features
# 1st and 2nd derivtive
# input: a low res image patches, size n*n 
# output: a n*n*4 feature, 
def extr_lIm_feat(lIm):
    r,c = lIm.shape
    hf1 = np.array([[-1,0,1]]) # horizontal
    vf1 = np.array([[-1],[0],[1]]) # vertical
    hf2 = np.array([[1,0,-2,0,1]]) # horizontal 
    vf2 = np.array([[1],[0],[-2],[0],[1]]) # vertical
    lImfeat = np.zeros((r,c,4))
    lImfeat[:,:,0] = convolve2d(lIm,hf1,mode='same',boundary='symm')
    lImfeat[:,:,1] = convolve2d(lIm,vf1,mode='same',boundary='symm')    
    lImfeat[:,:,2] = convolve2d(lIm,hf2,mode='same',boundary='symm')    
    lImfeat[:,:,3] = convolve2d(lIm,vf2,mode='same',boundary='symm')
    return lImfeat


#i = 1
#feat = extr_lIm_feat(patches[i,:,:])
#plt.figure
#plt.imshow(patches[i,:,:])
#plt.figure
#for j in range(4):
#    plt.subplot(2,2,j+1)
#    plt.imshow(feat[:,:,j])
    
