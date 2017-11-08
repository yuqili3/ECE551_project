import numpy as np
import glob
from scipy.misc import imread,imresize
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, rgb2ycbcr, ycbcr2rgb
from sklearn.feature_extraction import image
import linear_feature_helpers

def rnd_sample_patch(img_path, img_type, patch_size, num_patch, upscale):
    train_img_list = glob.glob(img_path+'*.'+ img_type)
    img_num = len(train_img_list)
    print('In total: '+str(img_num) +' images found!')
    n_per_img = np.zeros(img_num)
    
    for i in range(img_num):
        img = imread(train_img_list[i])
        n_per_img[i] = img.size
    
#    img = imread(train_img_list[1])
#    fig1=plt.figure()
#    plt.imshow(img , interpolation ="nearest", cmap="gray")
    n_per_img = np.floor(n_per_img*num_patch/sum(n_per_img))
    n_per_img = n_per_img.astype(int)
    Xh = np.array([]).reshape(0,patch_size**2)
    Xl = np.array([]).reshape(0,4*patch_size**2)
    for i in range(img_num):
        num_patch_in_img = n_per_img[i]
        img = imread(train_img_list[i])
        H,L = sample_patches(img, patch_size,num_patch_in_img,upscale)
        Xh = np.vstack((Xh,H)) 
        Xl = np.vstack((Xl,L)) 
    
    return Xh,Xl

def sample_patches(img, patch_size,num_patch_in_img,upscale):
    himg = rgb2gray(img) # convert to grayscale image
    limg = imresize(himg, 1/upscale, interp ='bicubic') # without blurring
    limg = imresize(limg, himg.shape, interp = 'bicubic') # convert back to the original size
    h_patch = image.extract_patches_2d(himg, (patch_size,patch_size)) 
    # extract patches in raster scan order
    patch_idx = np.random.choice(range(h_patch.shape[0]),num_patch_in_img, replace=False) 
    # generate idx to form pairs
    h_patch = h_patch[patch_idx,:,:] # randomly select patches in himg
    H = h_patch.reshape(h_patch.shape[0],-1) # vectorize each patch
    H -= H.mean(axis=1,keepdims=True)
    # restoring patch use H.reshape(H.shape[0],patch_size,patch_size)
    
    # apply feature extracter to the LR image rather than patches
    limg_feat = linear_feature_helpers.extr_lIm_feat(limg)
    l_patch = image.extract_patches_2d(limg_feat, (patch_size,patch_size)) 
    # returns num_patches*size*size*4
    l_patch = l_patch[patch_idx,:,:,:] # choose the same index to form pairs
    L = np.array([]).reshape(0,4*patch_size**2)
    for i in range(l_patch.shape[0]):
        L = np.vstack((L,l_patch[i,:,:,:].reshape(1,-1)))
    return H,L

def patch_pruning(Xh,Xl):
    pvars = np.var(Xh,axis = 1) # compute the varinace of each patch
    threshold = np.percentile(pvars,10) # throw away last 10%
    mask = (pvars > threshold)
    Xh = Xh[mask,:]
    Xl = Xl[mask,:]
    return Xh,Xl
    
    
    
    
    
    
    
    