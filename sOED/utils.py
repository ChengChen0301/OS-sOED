import numpy as np
import torch
import torch.nn as nn
# import warnings
# warnings.filterwarnings("ignore")

# logpdf of independent normal distribution.
# x is of size (n_sample, n_param).
# loc and scale are int or numpy.ndarray of size n_param.
# output is of size n_sample.
def norm_logpdf(x, loc=0, scale=1):
    logpdf = (-np.log(np.sqrt(2 * np.pi) * scale) 
              - (x - loc) ** 2 / 2 / scale ** 2)
    return logpdf.sum(axis=-1)

# pdf of independent normal distribution.
def norm_pdf(x, loc=0, scale=1):
    return np.exp(norm_logpdf(x, loc, scale))

# logpdf of uniform distribution.
def uniform_logpdf(x, low=0, high=1):
    return np.log(uniform_pdf(x, low, high))

# pdf of uniform distribution.
def uniform_pdf(x, low=0, high=1):
    pdf = ((x >= low) * (x <= high)) / (high - low)
    return pdf.prod(axis=1)

# Construct neural network
class Net(nn.Module):
    def __init__(self, dimns, activate, bounds):
        super().__init__()
        layers = []
        for i in range(len(dimns) - 1):
            layers.append(nn.Linear(dimns[i], dimns[i + 1]))
            if i < len(dimns) - 2:
                layers.append(activate)
        self.net = nn.Sequential(*layers)
        self.bounds = torch.from_numpy(bounds)
        self.has_inf = torch.isinf(self.bounds).sum()

    def forward(self, x):
        x = self.net(x)
        if self.has_inf:
            x = torch.maximum(x, self.bounds[:, 0])
            x = torch.minimum(x, self.bounds[:, 1])
        else:
            x = (torch.sigmoid(x) * (self.bounds[:, 1] - 
                                     self.bounds[:, 0]) + self.bounds[:, 0])
        return x


class Net_classify(nn.Module):
    def __init__(self, dimns, activate):
        super().__init__()
        layers = []
        # Build all layers except the last one
        for i in range(len(dimns) - 2):  # Note: -2 instead of -1
            layers.append(nn.Linear(dimns[i], dimns[i + 1]))
            layers.append(activate)

        # Add final layer with sigmoid activation
        layers.append(nn.Linear(dimns[-2], dimns[-1]))
        layers.append(nn.Sigmoid())  # Final activation for binary classification

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Net_categorical(nn.Module):
    def __init__(self, dimns, activate):
        super().__init__()
        layers = []
        # Build all layers except the last one
        for i in range(len(dimns) - 2):
            layers.append(nn.Linear(dimns[i], dimns[i + 1]))
            layers.append(activate)
            
        # Add final layer (logits)
        layers.append(nn.Linear(dimns[-2], dimns[-1]))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        logits = self.net(x)
        # Return log probabilities directly using log_softmax
        # This is numerically more stable than log(softmax(x))
        return torch.nn.functional.log_softmax(logits, dim=1)
    
    def get_action_prob(self, x):
        # Get regular probabilities if needed (e.g. for sampling)
        return torch.exp(self.forward(x))
    
    def get_log_prob(self, x, actions):
        """
        Get log probability for specific actions
        
        Parameters
        ----------
        x : torch.Tensor
            Input states of shape (batch_size, input_dim)
        actions : torch.Tensor
            Actions of shape (batch_size,) containing indices of chosen actions
            
        Returns
        -------
        torch.Tensor
            Log probabilities of shape (batch_size,)
        """
        # Get log probabilities for all actions - shape (batch_size, n_categories)
        log_probs = self.forward(x)
        # Get log probs of chosen actions
        return torch.gather(log_probs, 1, actions.unsqueeze(1)).squeeze(1)


class Net_prob_categorical(nn.Module):
    def __init__(self, dimns, activate):
        super().__init__()
        layers = []
        # Build all layers except the last one
        for i in range(len(dimns) - 2):
            layers.append(nn.Linear(dimns[i], dimns[i + 1]))
            layers.append(activate)
            
        # Add final layer (logits)
        layers.append(nn.Linear(dimns[-2], dimns[-1]))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        logits = self.net(x)
        # Return log probabilities directly using log_softmax
        # This is numerically more stable than log(softmax(x))
        return torch.nn.functional.softmax(logits, dim=1)
