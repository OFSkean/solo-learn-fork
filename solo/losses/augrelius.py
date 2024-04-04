import torch
import repitl.kernel_utils as ku
import repitl.matrix_itl as itl
import repitl.difference_of_entropies as dent

# returns H(X | Y)
def conditional_entropy(X, Y, kernel_type, alpha):

    # create kernel
    sigma = (X.shape[1] / 2) ** 0.5
    if kernel_type == 'gaussian':
        Kx = ku.gaussianKernel(X, X, sigma)
        Ky = ku.gaussianKernel(Y, Y, sigma)
    elif kernel_type == 'linear':
        Kx = (X.T @ X).double()
        Ky = (Y.T @ Y).double()
    else:
        raise NotImplementedError('Kernel type not implemented')

    conde = itl.matrixAlphaJointEntropy([Kx, Ky], alpha=alpha) - itl.matrixAlphaEntropy(Ky, alpha=alpha)

    return conde

# computes conditional entropy for pairs of elements in X and Y
# X[0] compared with Y[0], X[1] compared with Y[1], etc.
def multiview_conditional_entropy_loss(X, Y, kernel_type, alpha):
    assert len(X) == len(Y)

    total_conde = 0
    for view in range(len(X)):
        total_conde += conditional_entropy(X[view], Y[view], kernel_type, alpha)

    total_conde = total_conde / len(X)
    return total_conde