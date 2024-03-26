# grouping for the group-LASSO-norm regularization

import torch

def group_model(model: torch.nn.Module, name: str, lambda_: float):
    if name == "Linear":
        optimizer_grouped_parameters = [
        {
            "names": [],
            "params": [],
            "lambda_": lambda_,
            "dim": (0),
            "dim_": (1)
        },
        {
            "names": [],
            "params": [],
            "lambda_": 0.0,
            "dim": None,
            "dim_": None
        }]  
        
        for n, p in model.named_parameters():
            if p.ndim == 2:
                optimizer_grouped_parameters[0]["names"].append(n)
                optimizer_grouped_parameters[0]["params"].append(p)
            else:
                optimizer_grouped_parameters[1]["names"].append(n)
                optimizer_grouped_parameters[1]["params"].append(p)
                
    elif name == "ConvNet":
        optimizer_grouped_parameters = [
        {
            "names": [],
            "params": [],
            "lambda_": lambda_,
            "dim": (0,2,3),
            "dim_": (1)
        },
        {
            "names": [],
            "params": [],
            "lambda_": lambda_,
            "dim": (0),
            "dim_": (1)
        },
        {
            "names": [],
            "params": [],
            "lambda_": 0.0,
            "dim": None,
            "dim_": None
        }]

        for n, p in model.named_parameters():
            if p.ndim == 4:
                optimizer_grouped_parameters[0]["names"].append(n)
                optimizer_grouped_parameters[0]["params"].append(p)
            elif p.ndim == 2:
                optimizer_grouped_parameters[1]["names"].append(n)
                optimizer_grouped_parameters[1]["params"].append(p)
            else:
                optimizer_grouped_parameters[2]["names"].append(n)
                optimizer_grouped_parameters[2]["params"].append(p)
    # TODO
    else:
        pass

    return optimizer_grouped_parameters
