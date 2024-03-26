# optimizer

import torch
import math

from typing import Callable, Iterable, Tuple

from Core.prox_fns import prox_group_lasso, prox_nuclear_norm
from Core.solvers import pgd_solver_group_lasso, pgd_solver_nuclear_norm

# Check if the tensor is 4D or Conv2D.
def is_4d(dim):
    if dim == (0,2,3) or dim == (1,2,3):
        reshape = True
        if dim == (0,2,3):
            dim = (0,2)
        elif dim == (1,2,3):
            dim == (1,2)
    else:
        reshape = False

    return reshape, dim

# Reshape a 4D tensor into a 3D tensor. In other words, flatten a 2D kernel into a 1D vector.
def _4d_to_3d(p, grad, reshape):
    if reshape:
        p_ = p.view(p.shape[0], p.shape[1], -1)
        grad = grad.view(grad.shape[0], grad.shape[1], -1)
    else:
        p_ = p.clone().detach()

    return p_, grad

# Reshape a 3D tensor into a 4D tensor. That is, reshape a flattened tensor back to its original shape.
def _3d_to_4d(p_shape, p_tilde, reshape):
    if reshape:
        p_tilde = p_tilde.view(p_shape)
    else:
        p_tilde = p_tilde.clone().detach()
        
    return p_tilde
    

class RMDA(torch.optim.Optimizer):
    '''
    Arguments:
        params (iterable): 
            iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): 
            learning rate (default: 1e-1).
        momentum (float): 
            momentum value in  the range (0,1] (default: 1e-1). momentum = 1
            corresponds to the case that no momentum is used, while momentum =
            0 corresponds to the case that the iterate is not updated at all.
        lambda_ (float): 
            regularization weight (default: 0.0).
        prox_fn (Callable):
            proximal operator of the corresponding regularization (default:
            prox_group_lasso), the iterate is updated inplace and the return
            value is the objective value of the regularization function
            evaluated at the updated iterate
        dim (tuple):
            dimensions over which to operate (default: None).
        dim_ (tuple):
            the complement of dim (default: None).
    '''
    def __init__(self, 
                 params: Iterable[torch.nn.parameter.Parameter],
                 lr: float = 1e-1, 
                 momentum: float = 1e-1, 
                 lambda_: float = 0.0, 
                 prox_fn: Callable = prox_group_lasso, 
                 dim: tuple = None, 
                 dim_: tuple = None):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum <= 0.0 or momentum > 1.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if lambda_ < 0.0:
            raise ValueError("Invalid lambda_ value: {}".format(lambda_))

        defaults = dict(lr=lr, 
                        momentum=momentum, 
                        lambda_=lambda_, 
                        prox_fn=prox_fn, 
                        dim=dim, 
                        dim_=dim_)

        super(RMDA, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            lambda_ = group['lambda_']
            prox_fn = group['prox_fn']
            dim = group['dim']
            dim_ = group['dim_']
            
            reshape, dim = is_4d(dim=dim)
            
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                p_, grad = _4d_to_3d(p=p, grad=grad, reshape=reshape)

                state = self.state[p]
                if len(state) == 0:
                    step = state['step'] = 0
                    alpha = state['alpha'] = 0.0
                    p0 = state['initial_point'] = p_.clone().detach()
                    grad_sum = state['grad_sum'] = torch.zeros_like(p_, memory_format=torch.preserve_format)
                    state['regularization'] = None
                else:  
                    p0 = state['initial_point']
                    grad_sum = state['grad_sum']
                    
                state['step'] += 1
                step = state['step']
                scaling = step**0.5
                state['alpha'] += lr*scaling
                alpha = state['alpha']

                grad_sum.add_(grad, alpha=lr*scaling)

                p_tilde = p0.sub(grad_sum, alpha=1/scaling)

                if lambda_ != 0.0 and dim is not None:                       
                    state['regularization'] = prox_fn(p=p_tilde, alpha=1/scaling, lambda_=alpha*lambda_, dim=dim)

                p_tilde = _3d_to_4d(p_shape=p.shape, p_tilde=p_tilde, reshape=reshape)
                    
                if momentum != 1.0:
                    p.mul_(1.0-momentum).add_(p_tilde, alpha=momentum)
                else:
                    p.copy_(p_tilde)

        return loss

class RAMDA(torch.optim.Optimizer):
    '''
    Arguments:
        params (iterable): 
            iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): 
            learning rate (default: 1e-2).
        momentum (float): 
            momentum value in  the range (0,1] (default: 1e-1). momentum = 1
            corresponds to the case that no momentum is used, while momentum =
            0 corresponds to the case that the iterate is not updated at all.
        lambda_ (float): 
            regularization weight (default: 0.0).
        solver (Callable):
            subproblem solver of the corresponding regularization (default:
            pgd_solver_group_lasso), the iterate is updated inplace and the
            return value is the objective value of the regularization function
            evaluated at the updated iterate
        epsilon (float): 
            term added to the denominator in the diagonal preconditioner to improve numerical stability. (default: 1e-6).
        max_iters (int):
            maximum iterations of the subproblem solver. (default: 100) 
        rtol (float):
            relative tolerance for early stopping in the subproblem solver. (default: 1e-8)
        dim (tuple):
            dimensions over which to operate (default: None).
        dim_ (tuple):
            the complement of dim (default: None).
    '''
    def __init__(self, 
                 params: Iterable[torch.nn.parameter.Parameter],
                 lr: float = 1e-2, 
                 momentum: float = 1e-1, 
                 lambda_: float = 0.0,  
                 solver: Callable = pgd_solver_group_lasso, 
                 epsilon: float = 1e-6, 
                 max_iters: int = 100, 
                 rtol: float = 1e-8,
                 dim: tuple = None, 
                 dim_: tuple = None):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum <= 0.0 or momentum > 1.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if lambda_ < 0.0:
            raise ValueError("Invalid lambda_ value: {}".format(lambda_))
        if epsilon < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        if max_iters < 1:
            raise ValueError("Invalid max_iters value: {}".format(max_iters))
        if rtol < 0.0:
            raise ValueError("Invalid rtol value: {}".format(rtol))

        defaults = dict(lr=lr, 
                        momentum=momentum,
                        lambda_=lambda_, 
                        solver=solver, 
                        epsilon=epsilon, 
                        max_iters=max_iters,
                        rtol=rtol,
                        dim=dim, 
                        dim_=dim_)

        super(RAMDA, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            lambda_ = group['lambda_']
            solver = group['solver']
            epsilon = group['epsilon']
            max_iters = group['max_iters']
            rtol = group['rtol']
            dim = group['dim']
            dim_ = group['dim_']
            
            reshape, dim = is_4d(dim=dim)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                
                p_, grad = _4d_to_3d(p=p, grad=grad, reshape=reshape)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['alpha'] = 0.0
                    p0 = state['initial_point'] = p_.clone().detach()
                    grad_sum = state['grad_sum'] = torch.zeros_like(p_, memory_format=torch.preserve_format)
                    grad_sum_sq = state['grad_sum_sq'] = torch.zeros_like(p_, memory_format=torch.preserve_format)
                    state['regularization'] = None
                else:
                    p0 = state['initial_point']
                    grad_sum = state['grad_sum']
                    grad_sum_sq = state['grad_sum_sq']
                state['step'] += 1
                step = state['step']
                step_size = lr*step**0.5
                state['alpha'] += step_size
                alpha = state['alpha']
                                    
                grad_sum.add_(grad, alpha=step_size)

                grad_sum_sq.add_(grad.square(), alpha=step_size)

                denom = grad_sum_sq.pow(1/3).add(epsilon)

                if lambda_ != 0.0 and dim is not None:                      
                    p_tilde, state['regularization'] = solver(p=p_, p0=p0, v=grad_sum, denom=denom, alpha=1.0, lambda_=alpha*lambda_, dim=dim, dim_=dim_, max_iters=max_iters, rtol=rtol)
                else:
                    p_tilde = p0.addcdiv(grad_sum, denom, value=-1.0)
                    
                p_tilde = _3d_to_4d(p_shape=p.shape, p_tilde=p_tilde, reshape=reshape)

                if momentum != 1.0:
                    p.mul_(1.0-momentum).add_(p_tilde, alpha=momentum)
                else:
                    p.copy_(p_tilde)
                
        return loss
    
class ProxSGD(torch.optim.Optimizer):
    '''
    Arguments:
        params (iterable): 
            iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): 
            learning rate (default: 1e-1).
        momentum (float): 
            momentum value in  the range (0,1] (default: 1e-1). momentum = 1
            corresponds to the case that no momentum is used, while momentum =
            0 corresponds to the case that only the latest stochastic gradient
            is used.
        lambda_ (float): 
            regularization weight (default: 0.0).
        prox_fn (Callable):
            proximal operator of the corresponding regularization (default:
            prox_group_lasso), the iterate is updated inplace and the return
            value is the objective value of the regularization function
            evaluated at the updated iterate
        dim (tuple):
            dimensions over which to operate (default: None).
        dim_ (tuple):
            the complement of dim (default: None).
    '''
    def __init__(self, 
                 params: Iterable[torch.nn.parameter.Parameter],
                 lr: float = 1e-1, 
                 momentum: float = 1e-1, 
                 lambda_: float = 0.0,
                 prox_fn: Callable = prox_group_lasso, 
                 dim: tuple = None, 
                 dim_: tuple = None):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum <= 0.0 or momentum > 1.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if lambda_ < 0.0:
            raise ValueError("Invalid lambda_ value: {}".format(lambda_))

        defaults = dict(lr=lr, 
                        momentum=momentum, 
                        prox_fn=prox_fn, 
                        lambda_=lambda_, 
                        dim=dim, 
                        dim_=dim_)

        super(ProxSGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            lambda_ = group['lambda_']
            prox_fn = group['prox_fn']
            dim = group['dim']
            dim_ = group['dim_']

            reshape, dim = is_4d(dim=dim)

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                
                p_, grad = _4d_to_3d(p=p, grad=grad, reshape=reshape)
                    
                state = self.state[p]
                if len(state) == 0:
                    buf = state['momentum_buffer'] = torch.clone(grad).detach().mul(momentum)
                    state['regularization'] = None
                else:
                    buf = state['momentum_buffer']
                    buf.mul_(1.0-momentum).add_(grad, alpha=momentum)
                grad = buf

                p_.sub_(grad, alpha=lr)

                if lambda_ != 0.0 and dim is not None:                 
                    state['regularization'] = prox_fn(p=p_, alpha=lr, lambda_=lambda_, dim=dim) 

                p_ = _3d_to_4d(p_shape=p.shape, p_tilde=p_, reshape=reshape)
                    
                p.copy_(p_)

        return loss
    
class ProxAdamW(torch.optim.Optimizer):
    '''
    Arguments:
        params (iterable): 
            iterable of parameters to optimize or dicts defining parameter groups.
        lr (float): 
            learning rate (default: 1e-3).
        betas (Tuple): 
            coefficient used for computing the running average of the stochastic gradient and its square (default: (0.9, 0.999)).
        epsilon (float): 
            term added to the denominator in the diagonal preconditioner to improve numerical stability. (default: 1e-8).
        weight_decay (float):
            decoupled weight decay (default: 0.0).
        amsgrad (bool):
            whether to use the AMSGrad variant of this algorithm (from the paper On the Convergence of Adam and Beyond) (default: False)
        lambda_ (float): 
            regularization weight (default: 0.0).
        solver (Callable):
            subproblem solver of the corresponding regularization (default:
            pgd_solver_group_lasso), the iterate is updated inplace and the
            return value is the objective value of the regularization function
            evaluated at the updated iterate
        max_iters (int):
            maximum iterations of the subproblem solver. (default: 100) 
        rtol (float):
            relative tolerance for early stopping in the subproblem solver. (default: 1e-8)
        dim (tuple):
            dimensions over which to operate (default: None).
        dim_ (tuple):
            the complement of dim (default: None).
    '''
    def __init__(self, 
                 params: Iterable[torch.nn.parameter.Parameter],
                 lr: float = 1e-3, 
                 betas: Tuple[float, float] = (0.9, 0.999),
                 epsilon: float = 1e-8, 
                 weight_decay: float = 0.0, 
                 amsgrad: bool = False, 
                 lambda_: float = 0.0, 
                 solver: Callable = pgd_solver_group_lasso, 
                 max_iters: int = 100, 
                 rtol: float = 1e-8,
                 dim: tuple = None, 
                 dim_: tuple = None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= epsilon:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= lambda_:
            raise ValueError("Invalid lambda_ value: {}".format(lambda_))
        if max_iters < 1:
            raise ValueError("Invalid max_iters value: {}".format(max_iters))
        if rtol < 0.0:
            raise ValueError("Invalid rtol value: {}".format(rtol))
            
        defaults = dict(lr=lr, 
                        betas=betas, 
                        epsilon=epsilon,
                        weight_decay=weight_decay, 
                        amsgrad=amsgrad, 
                        lambda_=lambda_,
                        solver=solver, 
                        max_iters=max_iters, 
                        rtol=rtol,
                        dim=dim, 
                        dim_=dim_)
        
        super(ProxAdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ProxAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            
    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            epsilon = group['epsilon']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']
            lambda_ = group['lambda_']
            solver = group['solver']
            max_iters = group['max_iters']
            rtol = group['rtol']
            dim = group['dim']
            dim_ = group['dim_']
            
            reshape, dim = is_4d(dim=dim)
            
            for p in group['params']:
                if p.grad is None:
                    continue

                p.mul_(1.0-lr*weight_decay)

                grad = p.grad
                
                p_, grad = _4d_to_3d(p=p, grad=grad, reshape=reshape)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    exp_avg = state['exp_avg'] = torch.zeros_like(p_, memory_format=torch.preserve_format)
                    exp_avg_sq = state['exp_avg_sq'] = torch.zeros_like(p_, memory_format=torch.preserve_format)
                    if amsgrad:
                        max_exp_avg_sq = state['max_exp_avg_sq'] = torch.zeros_like(p_, memory_format=torch.preserve_format)
                    state['regularization'] = None
                else:
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    if amsgrad:
                        max_exp_avg_sq = state['max_exp_avg_sq']
                state['step'] += 1
                step = state['step']
                
                bias_correction1 = 1.0-beta1**step
                bias_correction2 = 1.0-beta2**step

                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt()/math.sqrt(bias_correction2)).add_(epsilon)
                else:
                    denom = (exp_avg_sq.sqrt()/math.sqrt(bias_correction2)).add_(epsilon)

                step_size = lr/bias_correction1
                    
                if lambda_ != 0.0 and dim is not None:                    
                    p_tilde, state['regularization'] = solver(p=p_, p0=p_.clone().detach(), v=exp_avg.mul(step_size), denom=denom, alpha=1.0, lambda_=lr*lambda_, dim=dim, dim_=dim_, max_iters=max_iters, rtol=rtol)
                    
                    p_.copy_(p_tilde)
                else:
                    p_.addcdiv_(exp_avg, denom, value=-step_size)
                    
                p_ = _3d_to_4d(p_shape=p.shape, p_tilde=p_, reshape=reshape)
                    
                p.copy_(p_)
                
        return loss
