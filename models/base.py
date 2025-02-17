from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable, Type
import abc

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch import Tensor


############################################################################
##### Base classes                                                     #####
##### - FittableModule: A nn.Module with .fit(X, y) support            #####
##### - Sequential: chaining together multiple FittableModules         #####
##### - make_fittable: turns type nn.Module into FittableModule        #####
############################################################################


class FittableModule(nn.Module):
    def __init__(self):
        """Base class that wraps nn.Module with a .fit(X, y) 
        and .fit_transform(X, y) method. Requires subclasses to
        implement .forward, as well as either .fit or .fit_transform.
        """
        super(FittableModule, self).__init__()
    

    def fit(self, X: Tensor, y: Tensor, **kwargs):
        """Fits the model given training data X and targets y.

        Args:
            X (Tensor): Training data, shape (N, D).
            y (Tensor): Training targets, shape (N, d).
        """
        self.fit_transform(X, y)
        return self
    

    def fit_transform(self, X: Tensor, y: Tensor) -> Tensor:
        """Fit the module and return the transformed data."""
        self.fit(X, y)
        return self(X)
    

    # @abc.abstractmethod
    # def forward(self, X: Tensor) -> Tensor:
    #     """Forward pass of the model."""
    #     pass



def make_fittable(module_class: Type[nn.Module]) -> Type[FittableModule]:
    """Converts a nn.Module class into a FittableModule class,
    with a .fit method that does nothing."""
    class FittableModuleWrapper(FittableModule, module_class):
        def __init__(self, *args, **kwargs):
            FittableModule.__init__(self)
            module_class.__init__(self, *args, **kwargs)
        
        def fit(self, X: Tensor, y: Tensor, **kwargs):
            self.to(X.device)
            return self
        
    return FittableModuleWrapper


Tanh = make_fittable(nn.Tanh)
ReLU = make_fittable(nn.ReLU)
Identity = make_fittable(nn.Identity)
FittableSequential = make_fittable(nn.Sequential)


##########################################
#### Logistic Regression and RidgeCV  ####
####      classifiers/regressors      ####
##########################################



class RidgeModule(FittableModule):
    def __init__(self, l2_reg: float = 1e-3, **kw_args):
        super(RidgeModule, self).__init__()
        self.l2_reg = l2_reg
        
    
    def fit(self, X: Tensor, y: Tensor, **kwargs):
        """Fit the Ridge model with a fixed l2_reg.
        Assumes X is of appropriate scale"""
        self.X_mean = X.mean(dim=0, keepdim=True)
        self.y_mean = y.mean(dim=0, keepdim=True)
        self.y_std = torch.clamp(y.std(dim=0, keepdim=True), 1e-6)
        
        # Only center X, normalize y
        X_centered = X - self.X_mean
        y_normalized = (y - self.y_mean) / self.y_std
        
        # Create linear layer
        N, D = X.shape
        N, d = y.shape
        self.linear = nn.Linear(D, d, bias=False)
        self.to(X.device)
        
        # Solve the ridge regression
        A = X_centered.T @ X_centered + self.l2_reg * N * torch.eye(D, device=X.device)
        B = X_centered.T @ y_normalized
        self.linear.weight.data = torch.linalg.solve(A, B).T
        
        # Compute bias term accounting for y's normalization
        self.b = self.y_mean - (self.X_mean @ self.linear.weight.T) * self.y_std
        print("W", self.linear.weight)
        print("b", self.b)
        print("self.X_mean", self.X_mean)
        print("self.y_mean", self.y_mean)
        print("self.y_std", self.y_std)
        print("EXTRA", self.X_mean @ self.linear.weight.T * self.y_std)
        return self
    
    
    def forward(self, X: Tensor) -> Tensor:
        return (self.linear(X) * self.y_std) + self.b
    
    

class RidgeLBFGS(FittableModule):
    def __init__(self, 
                 l2_reg: float = 1e-1,
                 lr: float = 0.1,
                 max_iter: int = 300,
                 batch_size: int = 32,
                 **kwargs
                 ):
        super(RidgeLBFGS, self).__init__()
        self.l2_reg = l2_reg
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
    
    
    def fit(self, 
            X: Tensor,
            y: Tensor, 
            init_top: Optional[FittableModule] = None,
            **kwargs):
        """Fit the Ridge model with a fixed l2_reg"""
        self.X_mean = X.mean(dim=0, keepdim=True)
        self.y_mean = y.mean(dim=0, keepdim=True)
        self.y_std = torch.clamp(y.std(dim=0, keepdim=True), 1e-6)
        
        # Center X, normalize y with std
        X_centered = X - self.X_mean
        y_normalized = (y - self.y_mean) / self.y_std
        
        # Create linear layer
        N, D = X.shape
        N, d = y.shape
        self.linear = nn.Linear(D, d, bias=False)
        if init_top is not None:
            self.linear.weight.data = init_top.linear.weight.data.clone()
        self.to(X.device)
        
        # Train
        with torch.enable_grad():
            optimizer = torch.optim.LBFGS(self.linear.parameters(), lr=self.lr, max_iter=self.max_iter, history_size=20)
            def closure():
                optimizer.zero_grad()
                loss = torch.nn.functional.mse_loss(
                    self.linear(X_centered), y_normalized, reduction="sum"
                ) / N
                loss += self.l2_reg * torch.norm(self.linear.weight)**2
                loss.backward()
                print("loss", loss)
                return loss
            optimizer.step(closure)

        # Compute bias term accounting for y's normalization
        self.b = self.y_mean - (self.X_mean @ self.linear.weight.T) * self.y_std
        print("W", self.linear.weight)
        print("b", self.b)
        print("self.X_mean", self.X_mean)
        print("self.y_mean", self.y_mean)
        print("self.y_std", self.y_std)
        return self
    
    
    def forward(self, X: Tensor) -> Tensor:
        return (self.linear(X) * self.y_std) + self.b





class RidgeSGD(FittableModule):
    def __init__(self, 
                 l2_reg: float = 1e-1,
                 lr: float = 0.1,
                 max_iter: int = 300,
                 batch_size: int = 1000,
                 tol: float = 1e-4,  # Relative tolerance for early stopping
                 patience: int = 20,   # Number of epochs to wait for improvement
                 AdamClass = torch.optim.Adam,
                 **kwargs
                 ):
        super(RidgeSGD, self).__init__()
        self.l2_reg = l2_reg
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tol = tol
        self.patience = patience
        self.AdamClass = AdamClass
    
    def fit(self, 
            X: Tensor,
            y: Tensor, 
            init_top: Optional[FittableModule] = None,
            **kwargs):
        """Fit the Ridge model with a fixed l2_reg"""
        self.X_mean = X.mean(dim=0, keepdim=True)
        self.y_mean = y.mean(dim=0, keepdim=True)
        self.y_std = torch.clamp(y.std(dim=0, keepdim=True), 1e-6)
        
        # Center X, normalize y with std
        X_centered = X - self.X_mean
        y_normalized = (y - self.y_mean) / self.y_std
        
        # Create linear layer
        N, D = X.shape
        N, d = y.shape
        self.linear = nn.Linear(D, d, bias=False)
        if init_top is not None:
            self.linear.weight.data = init_top.linear.weight.data.clone()
        self.to(X.device)
        
        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(X_centered, y_normalized)
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=True
        )

        # Early stopping variables
        best_loss = float("inf")
        best_weights = None
        patience_counter = 0
        
        # Train
        with torch.enable_grad():
            optimizer = self.AdamClass(self.linear.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=2*self.l2_reg)
            
            for epoch in range(self.max_iter):
                epoch_loss = 0.0
                for batch_X, batch_y in loader:
                    optimizer.zero_grad()
                    outputs = self.linear(batch_X)
                    loss = torch.nn.functional.mse_loss(outputs, batch_y, reduction="sum") / N
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss/len(loader)
                print(f"Epoch {epoch}, Loss: {avg_loss}")
                
                # Early stopping check
                if best_loss == float('inf'):
                    best_loss = avg_loss
                    best_weights = self.linear.state_dict()
                    patience_counter = 0
                else:
                    relative_improvement = (best_loss - avg_loss) / best_loss 
                    print(f"Epoch {epoch}, Loss: {avg_loss}, Rel. Improvement: {relative_improvement}")
                    if relative_improvement > self.tol:
                        best_loss = avg_loss
                        best_weights = self.linear.state_dict()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    if best_weights is not None:
                        self.linear.load_state_dict(best_weights)
                    break

        # Compute bias term
        self.b = self.y_mean - (self.X_mean @ self.linear.weight.T) * self.y_std
        return self


    def forward(self, X: Tensor) -> Tensor:
        return (self.linear(X) * self.y_std) + self.b



class LogisticRegressionAdam(FittableModule):
    def __init__(self, 
                 batch_size = 512,
                 num_epochs = 30,
                 lr = 0.01,):
        super(LogisticRegressionAdam, self).__init__()
        self.model = None
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr

    def fit(self, X: Tensor, y: Tensor, **kwargs):
        # Determine input and output dimensions
        input_dim = X.size(1)
        if y.dim() > 1 and y.size(1) > 1:
            output_dim = y.size(1)
            y_labels = torch.argmax(y, dim=1)
            criterion = nn.CrossEntropyLoss()
        else:
            output_dim = 1
            y_labels = y.squeeze()
            criterion = nn.BCEWithLogitsLoss()

        # Define the model
        self.model = nn.Linear(input_dim, output_dim)
        device = X.device
        self.model.to(device)

        # DataLoader
        dataset = torch.utils.data.TensorDataset(X, y_labels)
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
        )

        # Training loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self(X), y

    def forward(self, X: Tensor) -> Tensor:
        return self.model(X)



class LogisticRegression(FittableModule):
    def __init__(self,
                 n_classes: int = 10,
                 l2_lambda: float = 0.001,
                 lr: float = 1.0,
                 max_iter: int = 300,
                 ):
        super(LogisticRegression, self).__init__()
        self.n_classes = n_classes
        self.l2_lambda = l2_lambda
        self.lr = lr
        self.max_iter = max_iter
        self.linear = None


    def fit(self, 
            X: Tensor, 
            y: Tensor,
            init_top: Optional[FittableModule] = None,
            **kwargs
            ):
        """Fits a logistic regression model with L2 regularization
        via the LBFGS method.
        
        Args:
            X (Tensor): Training data, shape (N, D).
            y (Tensor): Training targets, shape (N, d).
            init_top (Optional[FittableModule]): Initial weights and bias to use. Defaults to None."""

        D = X.size(1)
        if self.n_classes > 2:
            #cross entropy
            loss_fn = nn.functional.cross_entropy #this is with logits
            self.linear = nn.Linear(D, self.n_classes).to(X.device)
        else:
            #binary cross entropy for n_classes=2
            loss_fn = nn.functional.binary_cross_entropy_with_logits
            self.linear = nn.Linear(D, 1).to(X.device)
        self.to(X.device)

        # No onehot encoding
        if self.n_classes > 2:
            y_labels = torch.argmax(y, dim=1)
        else:
            y_labels = y

        # Initialize weights and bias
        if init_top is not None:
            self.linear.weight.data = init_top.linear.weight.data.clone()
            self.linear.bias.data = init_top.linear.bias.data.clone()
        
        with torch.enable_grad():
            # Optimize
            optimizer = torch.optim.LBFGS(self.linear.parameters(), lr=self.lr, max_iter=self.max_iter)
            def closure():
                optimizer.zero_grad()
                logits = self.linear(X)
                loss = loss_fn(logits, y_labels)
                loss += self.l2_lambda * torch.sum(self.linear.weight**2)
                loss.backward()
                return loss
            optimizer.step(closure)
        return self


    def forward(self, X: Tensor) -> Tensor:
        return self.linear(X)