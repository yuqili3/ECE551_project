"""
L2-constrained quadratic minimization Lagrange Dual.
"""

import numpy as np
from numpy.random import rand, randn
from scipy.sparse import random
from scipy.optimize import least_squares


def fobj(dual_lambda, SSt, XSt, X, c, trXXt): 
    # the onjective function to be minimized
    L= XSt.shape[0]
    M= len(dual_lambda);
    
    SSt_lam_inv = np.linalg.inv(SSt + np.diag(dual_lambda))
    if L > M:
        f = np.trace(SSt_lam_inv @ (XSt.T @ XSt)) - trXXt + c*sum(dual_lambda)
    else:
        f = np.trace(XSt @ SSt_lam_inv @ XSt.T) - trXXt + c*sum(dual_lambda)
    return f

def jac(dual_lambda, SSt, XSt, X, c, trXXt): 
    # the jacobian of objective function df/d(dual_lambda)
    # some of the arguments were not used but least_squares requires the same
    # argument as in objective_func
    M = len(dual_lambda)
    J = np.zeros((1, M ))
    SSt_lam_inv = np.linalg.inv(SSt + np.diag(dual_lambda))
    J[0,:] = -np.linalg.norm(XSt @ SSt_lam_inv, axis = 0)
#    for k in range(M):
#        J[0,k] = - np.linalg.norm(XSt @ (SSt_lam_inv[:,k]) )
    J += c
    return J

def hes(dual_lambda, SSt, XSt): 
    # the hessian of objective function df/d(dual_lambda)
    M = len(dual_lambda)
    H = np.zeros((M, M ))
    SSt_lam_inv = np.linalg.inv(SSt + np.diag(dual_lambda))
    A = XSt @ SSt_lam_inv
    H = (A.T @ A) * SSt_lam_inv 
    H *= -2
    return H
def L2LS_learn_basis_dual(X, S, l2norm):
    """
    minimize_B   0.5*||X - B*S||^2
    subject to   ||B(:,j)||_2 <= l2norm, forall j=1...size(S,1)
    Solve a least square minimization with quadratic constraints using
    lagrange dual of Lee et al (2006).
    Parameters----------
    signal X : array_like, 2-dimensional
        The signal being decomposed as a sparse linear combination
        of the columns of the dictionary.
    solution S : ndarray, 1-dimensional, optional
        Pre-allocated vector to use to store the solution.
    L2 norm constraint l2norm: float
        the maximum of column norm of B can take
    Returns-------
    dictionary B : array_like, 2-dimensional
        The dictionary of basis functions from which to form the
        sparse linear combination.
    References----------
    .. [1] H. Lee, A. Battle, R. Raina, and A. Y. Ng. "Efficient
       sparse coding algorithms". Advances in Neural Information
       Processing Systems 19, 2007.
    """
    L,N = X.shape # L: # of features, N: # of samples
    M = S.shape[0] # M: # of dict_size
    SSt = S @ S.T
    XSt = X @ S.T
    dual_lambda = 10*abs(rand(M))
    c = l2norm**2
    trXXt = sum(sum(X @ X.T))
    
    lb = np.zeros(len(dual_lambda)) # lower bound of dual_lambda
    
    res = least_squares(fobj, dual_lambda, jac=jac, bounds = (lb, np.inf),
                                args = (SSt, XSt, X, c, trXXt),
                                ftol=1e-10,xtol = 1e-10, gtol = 1e-10,
                                verbose =0)
    dual_lambda = res.x
    Bt = np.linalg.inv(SSt+np.diag(dual_lambda)) @ XSt.T
    B = Bt.T
    
    return B


#B = L2LS_learn_basis_dual(randn(45,1000), random(512,1000), 1)


