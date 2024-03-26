# scheduler

import torch

class multistep_param_scheduler(object):
    def __init__(self, name: str = None, optimizer: torch.optim.Optimizer = None,
                 milestones: list = None, gamma: float = 1e-1):
        self.name = name
        self.milestones = milestones
        self.gamma = gamma
        self.momentum_step_count = 1
        if self.name == "RMDA" or self.name == "RAMDA":
            for param_group in optimizer.param_groups:
                self.momentum = param_group['momentum']      
            
    def step(self, optimizer: torch.optim.Optimizer = None, epoch: int = None):
        if self.name == "RMDA" or self.name == "RAMDA":
            if epoch+1 in self.milestones:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= self.gamma
                    
                    with torch.no_grad():
                        for p in param_group['params']:
                            optimizer.state[p]['step'] = 0
                            optimizer.state[p]['alpha'] = 0.0
                            optimizer.state[p]['initial_point'].copy_(p.clone().detach())
                            optimizer.state[p]['grad_sum'].copy_(torch.zeros_like(p, memory_format=torch.preserve_format))
                            if 'grad_sum_sq' in optimizer.state[p]:
                                optimizer.state[p]['grad_sum_sq'].copy_(torch.zeros_like(p, memory_format=torch.preserve_format)) 

                    if 'epsilon' in param_group:
                        param_group['epsilon'] *= self.gamma

        else:
            if epoch+1 in self.milestones:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= self.gamma
                    
    def momentum_step(self, optimizer: torch.optim.Optimizer = None, epoch: int = None):          
        if self.name == "RMDA" or self.name == "RAMDA":
            if epoch+1 > self.milestones[-1]:
                self.momentum_step_count += 1
                for param_group in optimizer.param_groups:
                    param_group['momentum'] = min(self.momentum*self.momentum_step_count**0.5, 1.0)
        else:
            pass