####### Test functions usable also for gradient methods (with grad calculated using torch.autograd) #####
import torch
import numpy as np
from functools import wraps

def torch_wrapper(func):
    '''
    Handles torch-numpy conversion. If applied to a function of (c_,x_), when calling f = tf1(c_, grad_=True) 
    it returns the function of x_ that returns tensors, resp. numpy arrays if grad_=False.
    '''
    @wraps(func)
    def wrapper(c_, grad_=False):
        if isinstance(c_, np.ndarray):
            c_ = torch.as_tensor(c_)
        
        def wrapped_func(x_):
            if isinstance(x_, np.ndarray):
                x_ = torch.as_tensor(x_)
            if x_.ndim == 1 or x_.ndim == 0:
                x_ = x_.reshape(1, x_.shape[0])
            val_ = func(x_, c_)
            return val_ if grad_ else val_.detach().numpy().reshape(1,)

        return wrapped_func
    
    return wrapper

# Example of using the decorator with a function
@torch_wrapper
def tf1(x_, c_):
    '''
    Euclidean norm to the power of 4.
    '''
    return torch.norm(x_ - c_, p=2, dim=1) ** 4

@torch_wrapper
def tf2(x_, c_):
    '''
    norm(sum(exp(x_-c_)), 2)**8
    '''
    return torch.norm(torch.exp(x_ - c_), p=2, dim=1) ** 8

@torch_wrapper
def tf3(x_, c_):
    '''
    var(x_-c_)
    '''
    x_ = x_ - c_
    m = torch.mean(x_, dim=1)
    N = x_.shape[1]
    return torch.sum((x_ - m) ** 2) / N

@torch_wrapper
def tf4(x_, c_):
    '''
    Gaussian exp(0.5 * (x-mu).T @ (x-mu))
    '''
    x_ = x_ - c_
    #x_t = torch.transpose(x_.clone(), 0, 1)
    return torch.exp(0.5 * torch.matmul(x_, x_.T)).reshape(1,)

@torch_wrapper
def tf5(x_, c_):
    '''
    exp(0.01 * (sum(exp(0.5*(x_-c_))) + tf1(c_)(x_))
    '''
    return torch.exp(0.01 * (torch.sum(torch.exp(0.5 * (x_ - c_)), dim=1) + torch.norm(x_ - c_, p=2, dim=1) ** 4))

@torch_wrapper
def tf6(x_, c_):
    '''
    2-norm(A(x-c))^6 for A np.cumsum( (c_ c_ ... c_), axis = 1) -> A symmetric
    '''
    d = len(c_)
    x_ = x_ - c_
    A = torch.repeat_interleave(c_, d).reshape(d, d)
    A = torch.cumsum(A, dim=1, dtype=torch.float64)  # returns a symmetric matrix
    #Ax = torch.sum(torch.mul(A,x_), axis=1) # A @ x
    return torch.norm(torch.sum(torch.mul(A,x_), axis=1), p=2, dim=0) ** 6


@torch_wrapper # TODO: this function used to be non-smooth -> adjust scripts accorsingly??? used to be max!!
def tf7(x_, c_):
    '''
    (x-c).T @ A @ (x-c) for A spd.
    '''
    d = len(c_)
    x_ = (x_ - c_).reshape(x_.shape[1])
    A = torch.repeat_interleave(c_, d).reshape(d, d)
    A = torch.cumsum(A, dim=1, dtype=torch.float64)  # returns a symmetric matrix
    A = torch.matmul(A, A.T) + 1.1*torch.eye(d) # make positive definite 
    return x_ @ A @ x_ #torch.matmul(torch.matmul(x_, A), x_.T)# x.TA @ x

# Non-smooth functions
@torch_wrapper
def tf8(x_, c_):
    '''
    1-norm(x_-c_)
    '''
    x_ = x_ - c_
    return torch.sum(torch.abs(x_))

@torch_wrapper
def tf9(x_, c_):
    '''
    tv_1(x_-c_)
    '''
    x_ = x_ - c_
    return torch.sum(torch.abs(x_[:, 1:] - x_[:, :-1]))

@torch_wrapper
def tf10(x_, c_):
    '''
    tv_2(x_-c_)
    '''
    x_ = x_ - c_
    return torch.sum(torch.norm(x_[:, 1:] - x_[:, :-1], p=2, dim=1))

# Smooth, only used for BOUNDED methods 
@torch_wrapper
def tf11(x_, c_):
    '''
    log(sum(exp(x_-c_)))
    '''
    return torch.log(torch.sum(torch.exp(x_ - c_), dim=1))