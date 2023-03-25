import numpy as np
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import functorch as ft

""" 
For the most part, this is a porting in PyTorch from here:
https://github.com/UCLA-StarAI/SIMPLE/blob/main/DVAE/DVAE-SIMPLE.ipynb
The backpropagation part exploits functorch to perform the JVP of the gradient operation.
"""

def log1mexp(x: Tensor) -> Tensor:
    """Numerically accurate evaluation of log(1 - exp(x)) for x < 0.
    See [Maechler2012accurate]_ for details.
    Taken from here: https://github.com/pytorch/pytorch/issues/39242
    """
    mask = -math.log(2) < x  # x < 0
    return torch.where(
        mask,
        (-x.expm1()).log(),
        (-x.exp()).log1p(),
    )

def log_sigmoid(logits):
  # This does not work with our implementation because F.logsigmoid is not implemented
  # in functorch.jvp. :-(
  return torch.clamp(F.logsigmoid(logits), max=-1e-7, min=-float('inf'))

def log_sigmoid_meh(logits):
  # Meh version (not numerically stable)
  return torch.clamp(torch.log(1.0/(1.0 + torch.exp(-logits))), max=-1e-7, min=-float('inf'))

def logaddexp(x1, x2):
    delta = torch.where(x1 == x2, 0., x1 - x2)
    return torch.maximum(x1, x2) + F.softplus(-delta.abs())

def log_pr_exactly_k(logp, logq, k):
    
    batch_size = logp.shape[0]
    n = logp.shape[1]
    
    state = torch.ones((batch_size, k+2)) * -float('inf')
    state[:, 1] = 0

    a = [state]
    
    for i in range(1, n+1):
        
        state = torch.cat([
            torch.ones([batch_size, 1]) * -float('inf'), 
            logaddexp(
                state[:, :-1] + logp[:, i-1:i], 
                state[:, 1:] + logq[:, i-1:i]
            )
        ], 1)
        
        a.append(state)
    a = torch.stack(a).transpose(1, 0)
    return a

def marginals(theta, k):
    # Note that this is different wrt the original code, since we are only returning the marginals, not their gradients.
    log_p = log_sigmoid_meh(theta)
    log_p_complement = log1mexp(log_p) 
    a = log_pr_exactly_k(log_p, log_p_complement, k)
    return a

def sample(a, probs):
    
    n = a.shape[-2] - 1
    k = a.shape[-1] - 1
    bsz = a.shape[0]
    
    j = torch.ones((bsz,)) * k
    samples = torch.zeros_like(probs)
    
    for i in torch.arange(n, 0, -1):
        
        # Unnormalized probabilities of Xi and -Xi
        full = torch.ones((bsz,)) * (i-1)
        p_idx = torch.stack([full, j-1], axis=1).long()
        z_idx = torch.stack([full + 1, j], axis=1).long()
      
        #p = gather_nd_torch(batch_dim=1, indices=p_idx, params=a)
        p = a[list((torch.arange(a.size(0)), *p_idx.T.chunk(2)))][0]
        z = a[list((torch.arange(a.size(0)), *z_idx.T.chunk(2)))][0]

        p = (p + probs[:, i-1]) - z
        q = log1mexp(p)

        # Sample according to normalized dist.
        X = torch.distributions.bernoulli.Bernoulli(logits=(p-q)).sample()

        # Pick next state based on value of sample
        j = torch.where(X>0, j - 1, j)

        # Concatenate to samples
        samples[:, i-1] = X
    
    # Our samples should always satisfy the constraint
    # tf.debugging.assert_equal(tf.math.reduce_sum(samples, axis=-1), k-1)
    
    return samples.float()

class SIMPLE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, k):
        logp = log_sigmoid(logits)
        logq = log1mexp(logp)
        
        a = log_pr_exactly_k(logp, logq, k)
        samples_p = sample(a, logp)

        ctx.save_for_backward(logits)
        ctx.k = k
        return samples_p

    @staticmethod
    def backward(ctx, grad_output):
        logits = ctx.saved_tensors
        _, tangent_out = ft.jvp(ft.grad(lambda logits: marginals(logits, 2)[:, -1, ctx.k+1:ctx.k+2].sum()), logits, (grad_output,))
        return tangent_out, None
        
def simple_sampler(logits, k=1):
    return SIMPLE.apply(logits, k)

class SIMPLESampler(nn.Module):
    def __init__(self, k=1):
        super().__init__()
        self.k = k

    def forward(self, logits):
        return simple_sampler(logits, self.k)


