# solvers

import torch

from torch import Tensor
from typing import Callable
from Core.prox_fns import prox_group_lasso, prox_nuclear_norm, subproblem_objective

'''
See the comments of the function subproblem_objective in prox_fns.py 
if you want to understand the arguments: p, p0, v, denom, alpha and lambda_. 
These arguments construct the subproblem objective function. Also, 
see the comments about arguments in optimizer.py 
if you want to understand the arguments: max_iters, rtol, dim and dim_.
Thses arguments are passed from the optimizer. 
'''

class pgd(object):
    def __init__(self, 
                 p: Tensor, 
                 p0: Tensor, 
                 v: Tensor, 
                 denom: Tensor, 
                 alpha: float, 
                 lambda_: float,
                 reweight: bool,
                 prox_fn: Callable,
                 subproblem_objective: Callable,
                 max_iters: int, 
                 rtol: tuple,
                 dim: tuple = None):
        self.p = p 
        self.p0 = p0
        self.v = v 
        self.denom = denom
        self.alpha = alpha
        self.lambda_ = lambda_  
        self.reweight = reweight
        self.prox_fn = prox_fn
        self.subproblem_objective = subproblem_objective
        self.dim = dim
        self.max_iters = max_iters
        self.rtol = rtol
        self.eta = self.denom.amax(dim=dim, keepdim=True).reciprocal()
        self.diff_denom = None
        self.regularization = None
    
    def step(self):
        subprob_grad = self.v.add(self.diff_denom.div(self.alpha))
        self.p.addcmul_(subprob_grad, self.eta, value=-1)
        self.regularization = self.prox_fn(p=self.p, alpha=self.eta, lambda_=self.lambda_, dim=self.dim) 
        
    def solve(self):
        self.regularization = self.p.sum() # The initial subproblem objective value does not need to be specified, as we enforce the number of iterations to be larger than 1 and will not use the initial objective value in any calculation.
        self.diff_denom, previous = self.subproblem_objective(p=self.p, p0=self.p0, v=self.v, denom=self.denom, regularization=self.regularization, alpha=self.alpha, lambda_=self.lambda_, dim=self.dim, reweight=self.reweight)
        for i in range(self.max_iters): 
            self.step()
            self.diff_denom, current = self.subproblem_objective(p=self.p, p0=self.p0, v=self.v, denom=self.denom, regularization=self.regularization, alpha=self.alpha, lambda_=self.lambda_, dim=self.dim, reweight=self.reweight)
          
            # early stopping
            if (previous-current)/(abs(current)+1.0) < self.rtol and i > 0:
                break
                                
            previous = current   

        return self.regularization
            
def pgd_solver_group_lasso(p: Tensor, 
                           p0: Tensor, 
                           v: Tensor, 
                           denom: Tensor, 
                           alpha: float, 
                           lambda_: float, 
                           dim: tuple, 
                           dim_: tuple, 
                           max_iters: int, 
                           rtol: tuple):  
    '''
    We mask out the parameters that will be zero at the optimal solution using Theorem 1 of
    https://arxiv.org/abs/2102.03869
    Tristan Deleu, Yoshua Bengio
    Structured Sparsity Inducing Adaptive Optimizers for Deep Learning, 2021
    '''
    # compute mask
    mask = torch.linalg.norm(denom.mul(p0).sub(v), dim=dim, keepdim=True).gt(alpha*lambda_)
     
    # initialize parameters
    p_tilde = torch.zeros_like(p)
    
    if mask.any().item():
        # select parameters
        if dim == (0,2) or dim == (0):
            index = mask.float().nonzero(as_tuple=True)[1]
        elif dim == (1,2) or dim == (1):
            index = mask.float().nonzero(as_tuple=True)[0]            
        p = p.index_select(dim_, index)
        v = v.index_select(dim_, index)
        p0 = p0.index_select(dim_, index)
        denom = denom.index_select(dim_, index) 
        
        # solve subproblem by pgd
        solver = pgd(p=p, 
                     p0=p0, 
                     v=v, 
                     denom=denom, 
                     alpha=alpha, 
                     lambda_=lambda_,
                     reweight=True,
                     prox_fn=prox_group_lasso,
                     subproblem_objective=subproblem_objective,
                     dim=dim, 
                     max_iters=max_iters, 
                     rtol=rtol)
        
        norm = solver.solve()
        
        # assign parameters value
        p_tilde.index_copy_(dim_, index, p)
        
    return p_tilde, norm

def pgd_solver_nuclear_norm(p: Tensor, 
                            p0: Tensor, 
                            v: Tensor, 
                            denom: Tensor, 
                            alpha: float, 
                            lambda_: float, 
                            dim: tuple, 
                            dim_: tuple, 
                            max_iters: int, 
                            rtol: tuple): 

    solver = pgd(p=p, 
                 p0=p0, 
                 v=v, 
                 denom=denom, 
                 alpha=alpha, 
                 lambda_=lambda_,
                 reweight=False,
                 prox_fn=prox_nuclear_norm,
                 subproblem_objective=subproblem_objective,
                 max_iters=max_iters, 
                 rtol=rtol)
    
    norm = solver.solve()
        
    return p, norm
