import torch
import repitl.kernel_utils as ku
import repitl.matrix_itl as itl
import repitl.difference_of_entropies as dent

# returns H(X | Y)
def conditional_entropy(X, Y):
    sigmax = torch.sqrt(torch.tensor(X.shape[1]/2))
    sigmay = torch.sqrt(torch.tensor(Y.shape[1]/2))

    Kx = ku.gaussianKernel(X, X, sigmax)
    Ky = ku.gaussianKernel(Y, Y, sigmay) #kwargs['sigma_y'])

    conde = itl.matrixAlphaJointEntropy([Kx, Ky], 2) - itl.matrixAlphaEntropy(Ky, 2)

    return conde