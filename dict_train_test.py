import numpy as np
import dict_helpers
from sklearn.preprocessing import normalize
import efficient_sparse_coding


dict_size   = 512;          # dictionary size
lambda_     = 0.15;         # sparsity regularization
patch_size  = 3;            # image patch size
num_patch   = 10000;       # number of patches to sample,say 100,000 from 30 images
upscale     = 2;            # upscaling factor
iteration_num = 10;          # number of alternating iterations, say 40
img_path = 'ECE551/551_Data/Training/t'
img_type = 'bmp'
dic_path = 'ECE551/551_Data/Dictionary'

def train_coupled_dict(Xh,Xl,dict_size,lambda_,iteration_num):
    hdim,ldim = Xh.shape[1], Xl.shape[1]
    Xh = normalize(Xh,norm='l2',axis=1) # normalize so that each patch (row) have unit norm
    Xl = normalize(Xl,norm='l2',axis=1)
    X = np.hstack((np.sqrt(hdim)*Xh, np.sqrt(ldim)*Xl))
    X = normalize(X,norm='l2',axis=1) # dim here is: num_patch * (hdim+ldim)
    X = X.T # dim here is: (hdim+ldim) * num_patch
    # now min 
    D = efficient_sparse_coding.dict_training(X,dict_size,lambda_,iteration_num)
    Dh = D[0:hdim,:]
    Dl = D[hdim:,:]
    # returned dict is not normalized
    # and of size hdim* num_patch and ldim* num_patch
    return Dh,Dl

Xh,Xl = dict_helpers.rnd_sample_patch(img_path, img_type, patch_size, num_patch, upscale)
Xh,Xl = dict_helpers.patch_pruning(Xh,Xl)
Dh,Dl = train_coupled_dict(Xh,Xl,dict_size,lambda_,iteration_num)

dict_path = dic_path+'/D_'+str(dict_size)+'_'+str(lambda_)+'_'+str(patch_size)+'_'+str(upscale)+'_Dh.out'
np.savetxt(dict_path, Dh)
dict_path = dic_path+'/D_'+str(dict_size)+'_'+str(lambda_)+'_'+str(patch_size)+'_'+str(upscale)+'_Dl.out'
np.savetxt(dict_path, Dl)
