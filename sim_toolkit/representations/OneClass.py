# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
  
  ----------------------------------------- 
  One-class representations
  ----------------------------------------- 

"""

from __future__ import absolute_import, division, print_function

import numpy as np
import sys

import logging
import torch
import torch.nn as nn

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
from .networks import *  

from torch.autograd import Variable
import random

# One-class loss functions
# ------------------------


def OneClassLoss(outputs, c): 
    
    dist   = torch.sum((outputs - c) ** 2, dim=1)
    loss   = torch.mean(dist)
    
    return loss


def SoftBoundaryLoss(outputs, R, c, nu):
    
    dist   = torch.sum((outputs - c) ** 2, dim=1)
    scores = dist - R ** 2
    loss   = R ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
    
    scores = dist 
    loss   = (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
    
    return loss


LossFns    = dict({"OneClass": OneClassLoss, "SoftBoundary": SoftBoundaryLoss})

# Base network
# ---------------------

class BaseNet(nn.Module):
    
    """Base class for all neural networks."""

    def __init__(self):
        
        super().__init__()
        
        self.logger  = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None  # representation dimensionality, i.e. dim of the last layer

    def forward(self, *input):
        
        """Forward pass logic
        
        :return: Network output
        """
        raise NotImplementedError

    def summary(self):
        
        """Network summary."""
        
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params         = sum([np.prod(p.size()) for p in net_parameters])
        
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)


def get_radius(dist:torch.Tensor, nu:float):
    
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    
    return np.quantile(np.sqrt(dist.clone().data.float().numpy()), 1 - nu)

class OneClassLayer(BaseNet):

    def __init__(self, params=None, hyperparams=None, seed=42):
        
        super().__init__()
        
        # set all representation parameters - remove these lines
        
        self.rep_dim        = params["rep_dim"] 
        self.input_dim      = params["input_dim"]
        self.num_layers     = params["num_layers"]
        self.num_hidden     = params["num_hidden"]
        self.activation     = params["activation"]
        self.dropout_prob   = params["dropout_prob"]
        self.dropout_active = params["dropout_active"]  
        self.loss_type      = params["LossFn"]
        self.train_prop     = params['train_prop']
        self.learningRate   = params['lr']
        self.epochs         = params['epochs']
        self.warm_up_epochs = params['warm_up_epochs']
        self.weight_decay   = params['weight_decay']
        if torch.cuda.is_available():
            self.device     = torch.device('cuda') # Make this an option
        else:
            self.device     = torch.device('cpu')
        # set up the network
        
        self.model          = build_network(network_name="feedforward", params=params).to(self.device)
        self.seed = seed

        # create the loss function

        self.c              = hyperparams["center"].to(self.device)
        self.R              = hyperparams["Radius"]
        self.nu             = hyperparams["nu"]

        self.loss_fn        = LossFns[self.loss_type]

        
    def forward(self, x):
        
        x                   = self.model(x)
        
        return x
    
    
    def fit(self, x_train, verbosity=True):

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.optimizer      = torch.optim.AdamW(self.model.parameters(), lr=self.learningRate, weight_decay = self.weight_decay)
        self.X              = torch.tensor(x_train.reshape((-1, self.input_dim))).float()
        
        if self.train_prop != 1:
            x_train, x_val = x_train[:int(self.train_prop*len(x_train))], x_train[int(self.train_prop*len(x_train)):]
            inputs_val = Variable(x_val.to(self.device)).float()
        
        self.train_losses     = []
        self.val_losses       = []
        
        print("Training the One-Class (OC) Network... ", flush=True)
        for epoch in range(self.epochs):
            
            # Converting inputs and labels to Variable
            inputs = Variable(x_train).to(self.device).float()
            
            self.model.zero_grad()

            self.optimizer.zero_grad()

            # get output from the model, given the inputs
            outputs = self.model(inputs)

            # get loss for the predicted output
            
            if self.loss_type=="SoftBoundary":
                
                self.loss = self.loss_fn(outputs=outputs, R=self.R, c=self.c, nu=self.nu) 
                
            elif self.loss_type=="OneClass":
                
                self.loss = self.loss_fn(outputs=outputs, c=self.c) 
            
            #self.c    = torch.mean(torch.tensor(outputs).float(), dim=0)
            
            # get gradients w.r.t to parameters
            self.loss.backward(retain_graph=True)
            self.train_losses.append(self.loss.item())
        
            # update parameters
            self.optimizer.step()
            
            if (epoch >= self.warm_up_epochs) and (self.loss_type=="SoftBoundary"):
                
                dist   = torch.sum((outputs - self.c) ** 2, dim=1)
                #self.R = torch.tensor(get_radius(dist, self.nu))
            
            if self.train_prop != 1.0:
                with torch.no_grad():
                    
                    # get output from the model, given the inputs
                    outputs = self.model(inputs_val)
        
                    # get loss for the predicted output
                    
                    if self.loss_type=="SoftBoundary":
                        
                        loss_val = self.loss_fn(outputs=outputs, R=self.R, c=self.c, nu=self.nu) 
                        
                    elif self.loss_type=="OneClass":
                        
                        loss_val = self.loss_fn(outputs=outputs, c=self.c).detach.cpu().numpy()
                    
                    self.val_losses.append(loss_val.item())
            
            if verbosity:
                print(f"\rEpoch {epoch+1}/{self.epochs} ", end="", flush=True)
        if verbosity:
            print()    

        return {"train": list(self.train_losses), "val": list(self.val_losses)}        
                
    def save_losses(self, path_train="OC_train_losses.npy", path_val="OC_val_losses.npy", verbosity=True):
        import numpy as np
        np.save(path_train, np.array(self.train_losses, dtype=float))
        if verbosity:
            print(f"-> Saved training losses to: {path_train}")
        if len(self.val_losses) > 0:
            np.save(path_val, np.array(self.val_losses, dtype=float))
            if verbosity:
                print(f"-> Saved validation losses to: {path_val}")