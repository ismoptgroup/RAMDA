# prox_fns

import torch

from torch import Tensor

def reweight_fn(p: Tensor, lambda_: float, dim: tuple):
    '''
    Reweighting regularization hyperparameter lambda_ according to the group size for the group-LASSO norm regularization
    '''
    if dim == (0,2) or dim == (0):
        group_size = p.numel()/p.shape[1]
    elif dim == (1,2) or dim == (1):
        group_size = p.numel()/p.shape[0]
    lambda_ *= group_size**0.5

    return lambda_

def prox_group_lasso(p: Tensor, alpha: Tensor, lambda_: float, dim: tuple):
    lambda_ = reweight_fn(p=p, lambda_=lambda_, dim=dim)

    threshold = alpha*lambda_
    norm = torch.nn.functional.relu(torch.linalg.norm(p, dim=dim, keepdim=True).sub(threshold))
    p.mul_(norm.div(norm.add(threshold)))

    return norm

def prox_nuclear_norm(p: Tensor, alpha: float, lambda_: float, dim: tuple = None):
    threshold = alpha*lambda_
    U, S, V_T = torch.linalg.svd(p, full_matrices=False)
    norm = torch.nn.functional.relu(S.sub(threshold))
    p.copy_(U.matmul(norm.diagflat()).matmul(V_T))
        
    return norm

def subproblem_objective(p: Tensor, 
                         p0: Tensor, 
                         v: Tensor, 
                         denom: Tensor, 
                         regularization: Tensor,
                         alpha: float, 
                         lambda_: float,
                         dim: tuple,
                         reweight: bool):
    '''
    The subproblem objective below follows formula (2) of
    https://proceedings.neurips.cc/paper/2021/hash/cc3f5463bc4d26bc38eadc8bcffbc654-Abstract.html
    Jihun Yun, Aurelie C. Lozano, Eunho Yang
    Adaptive Proximal Gradient Methods for Structured Neural Networks, NeurIPS 2021,
    which is
    \inprod{v}{p-p0} + \frac{1}{2 \alpha} \inprod{p-p0}{diag(denom) (p-p0)} + \lambda \psi(p),
    where \psi is the regularization function, \inprod is the inner product, and
    diag is the diagonal matrix of the given vector.
    '''
    if reweight:
        lambda_ = reweight_fn(p=p, lambda_=lambda_, dim=dim)
        
    diff = p.sub(p0)
    diff_denom = diff.mul(denom)
    objective = (diff.mul(v).sum()+
                 diff_denom.mul(diff).sum().mul(1/(2*alpha))+
                 regularization.mul(lambda_)).sum().item()
        
    return diff_denom, objective
