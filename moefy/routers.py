import torch
from torch import nn
from torch.nn.functional import softmax, softplus, gumbel_softmax, normalize
from einops import einsum, repeat, rearrange
from torch.distributions import Normal
from typing import Literal
from .samplers import simple_sampler

"""
Routers for a Mixture of Experts of Transformers.
A router is an nn.Module that given an input (sequence, embedding) returns a 
mask (num experts, sequence, embedding), 
such that mask[exp_id]*input is the input for expert exp_id.

Inspired by https://github.com/davidmrau/mixture-of-experts.
""" 


class TopkRouter(nn.Module):
    def __init__(
        self,
        num_experts: int, 
        input_dim: int, 
        k: int, 
        noise : Literal['gumbel', 'gaussian', 'simple'] = 'gumbel',
        aux_criterion: Literal['entropy', 'prob'] = 'entropy',
        expert_choice: bool = False
        ) -> None:
        super().__init__()

        assert not (aux_criterion == 'prob' and noise != 'gaussian'), "Probability auxiliary loss supported only for Gaussian noise"
        assert k < num_experts or expert_choice, f"The 'k' value for top-k routing ({self.k}) has to be < number of experts ({self.num_experts})"


        self.num_experts = num_experts
        self.input_dim = input_dim
        self.k = k
        self.noise = noise
        self.aux_criterion = aux_criterion
        self.expert_choice = expert_choice

        # initialize expert embeddings
        self.expert_embs = nn.Parameter(torch.empty((self.input_dim, self.num_experts), requires_grad=True))
        torch.nn.init.orthogonal_(self.expert_embs, gain=1)

        if self.noise == 'gaussian':
            self.routing_noise = nn.Parameter(torch.randn(input_dim, num_experts), requires_grad=True)
            self.register_buffer("mean", torch.tensor([0.0]))
            self.register_buffer("std", torch.tensor([1.0]))
    

    @staticmethod
    def cv_squared(x):
        """
        CV squared as per as per https://arxiv.org/abs/1701.06538. 
        """
        eps = 1e-10
        if x.dim() == 0:
            return x
        if x.shape[0]==1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)
        
     
    def balancing_loss(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """
        Balancing loss as per https://arxiv.org/abs/1701.06538. 
        Implementation from https://github.com/davidmrau/mixture-of-experts.
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    
    
    @staticmethod
    def entropy_loss(routing_matrix):
        return torch.special.entr(normalize(routing_matrix.sum(dim=0), dim=0, eps=1e-6))
    

    def forward(self, x):
        """
        Compute routing scores multiplying each input token with expert embeddings, and return a mask for
        the input for each expert.

        Args:
            x : torch.Tensor -> unbatched input tokens of shape (seq, embedding_dim)
        Returns:
            routing_matrix: torch.Tensor -> routing matrix of shape (exp, seq, dim), where the input for expert i 
                can be obtained by multiplying the initial input sequence by routing_matrix[i]
            aux_loss: torch.Tensor -> an auxiliary balancing loss
        """
        # init auxiliary loss
        aux_loss = torch.tensor(0.0, device=x.device)

        # unbatch input
        b, s, d = x.shape
        x = rearrange(x, "b s d -> (b s) d")

        # compute routing scores
        routing_scores = einsum(x, self.expert_embs, "seq dim, dim exp -> seq exp")
        if self.expert_choice:
            routing_scores = routing_scores.t() 
        

        # add noise
        if self.training:
            if self.noise == 'gumbel':
                routing_matrix = gumbel_softmax(routing_scores, hard=False, dim=-1)
            elif self.noise == 'simple':
                routing_matrix = simple_sampler(routing_scores, k = self.k)
            elif self.noise == 'gaussian':
                raw_noise_stddev = einsum(x, self.routing_noise, "seq dim, dim exp -> seq exp")
                raw_noise_stddev = raw_noise_stddev.t() if self.expert_choice else raw_noise_stddev
                noise_stddev = ((softplus(raw_noise_stddev)) + 1e-2) 
                noisy_routing_matrix = routing_scores + (torch.randn_like(routing_scores) * noise_stddev)  
                routing_matrix = softmax(noisy_routing_matrix, dim=-1)
            else:
                routing_matrix = softmax(routing_scores, dim=-1)
        else:
            routing_matrix = softmax(routing_scores, dim=-1) 
        
        # get topk routing scores
        # we need topk+1 in case we are computing auxiliary loss
        top_logits, top_indices = routing_matrix.topk(self.k + 1, dim=-1)
        top_k_logits, top_k_indices = top_logits[:, :self.k], top_indices[:, :self.k]

        
        # compute aux loss
        if self.training:
            if self.aux_criterion == 'entropy':
                aux_loss = self.entropy_loss(routing_matrix=routing_matrix)
            elif self.aux_criterion == 'prob':
                _top_logits, _ = routing_scores.topk(self.k + 1, dim=-1)
                aux_loss = (self.balancing_loss(routing_scores, noisy_routing_matrix, noise_stddev, _top_logits)).sum(0)
            

        # build up a routing mask such that mask[i] * x is input for expert i
        zeros = torch.zeros_like(routing_matrix, requires_grad=True) 
        experts_masks = zeros.scatter(1, top_k_indices, top_k_logits)  
        experts_masks = experts_masks.t() if self.expert_choice else experts_masks
        experts_masks = rearrange(experts_masks, "(b seq) exp -> exp b seq ()", b=b)

        return experts_masks, aux_loss, routing_matrix.t() if self.expert_choice else routing_matrix