import numpy as np
from numpy.random import rand
from sklearn.preprocessing import normalize
import L1LS_featuresign
import L2LS_learn_basis_dual

def dict_training(X, num_bases, lambda_, num_iters):
    '''
    minimize ||X - BS||_2^2 +lambda_*||S||_1
    s.t. ||Dj||^2 <= 1
    
    alternate between Z and D to get 
    X           -data samples, column wise (1 col = 1 sample, )
    num_bases   -number of bases
    lambda_     -sparsity regularization
    num_iters   -number of iterations
    
    '''
    L, N = X.shape
    M = num_bases
    S = np.zeros((M, N)) # initialize S
    B = rand(L,M) # initialize B
    B = normalize(B, norm = 'l2', axis = 0) # each col will have unit norm
    for t in range(num_iters):
        for i in range(N):
            S[:,i] = L1LS_featuresign.L1LS_feature_sign_search(B,X[:,i],lambda_)
        sparsity = sum(sum(S!=0))/S.size
        print('sparsity in S :'+str(sparsity))
        B = L2LS_learn_basis_dual.L2LS_learn_basis_dual(X,S,1)
        print('# col norm > 1: ' + str(sum(np.linalg.norm(B,axis=0)>=1)))
        print('# col norm = 0: ' + str(sum(np.linalg.norm(B,axis=0)<=1e-10)))      
        obj = np.linalg.norm(X-B@S)**2 + lambda_*sum(sum(abs(S)))
        print('Iteration '+str(t)+' :finished. Objective func = '+str(obj))
    return B