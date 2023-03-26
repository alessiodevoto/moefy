import torch
import torch.nn as nn
from einops import rearrange, repeat, pack, einsum, reduce
from typing import List, Literal, Optional
from .routers import TopkRouter
from abc import ABC, abstractmethod



class MoEBlock(nn.Module, ABC):
    def __init__(
                self,
                experts: List, 
                input_dim: int, 
                k: int ,
                hole: bool, 
                noise : Literal['gumbel', 'gaussian'] = 'gaussian',
                aux_criterion :  Literal['entropy', 'prob'] = 'prob',
                aggregation: Literal['mean', 'sum'] = 'sum',
                expert_choice: bool = False,
                layer_id: Optional[int] = None,
                *args,
                **kwargs
                ):
        """
        Args:
            input_dim: int -> size of input tokens
            k: int = 1 -> k for top-k token routing
            hole: bool = False -> whether to add and expert that drops tokens. 
            noise : Literal['gumbel', 'gaussian'] = 'gaussian' -> what kind of noise to add to the routing scores.
            aux_criterion :  Literal['entropy', 'prob'] = 'prob' -> auxiliary loss to be used for load balancing.  
            aggregation: Literal['mean', 'sum'] = 'sum' -> whether output tokens should summed or averaged  
            layer_id: int = None -> an id to keep track od this block if inside a bigger model
            expert_choice: bool = False -> experts choose tokens?  
        """
        super(MoEBlock, self).__init__()
        self.layer_id = layer_id
        self.input_dim = input_dim
        self.k = k
        self.noise = noise
        self.hole = hole
        self.aux_criterion = aux_criterion
        self.num_experts = len(experts) + 1 if hole else len(experts)
        self.aggregation = aggregation
        self.expert_choice = expert_choice

        # routing mechanism
        self.router = TopkRouter(
            num_experts = self.num_experts,
            input_dim = self.input_dim,
            k = self.k,
            noise = self.noise,
            aux_criterion=self.aux_criterion,
            expert_choice=self.expert_choice
        )

        # initialize experts list
        if self.hole:
            experts.append(Hole())
        self.experts = nn.ModuleList(experts)
        self._init_experts()
    

    def forward(self, *args, **kwargs): # kword args just for compatiblity with nn.MultiheadAttention
        """
        This is only self attention, so key and value are neglected. 
        Returns the result of the MoE of nn.MultiheadAttetnion + a dummy mask of zeros, which is not intended to be used.
        """
        
        x = args[0]
        b, s, d = x.size()
        self.batch_size = b

        # collapse batch 
        # get routing matrix and aux loss
        # x = rearrange(x, "batch seq dim -> (batch seq) dim")
        self.exp_masks, self.aux_loss, self.rout_matrix = self.router(x)
        routing_mask = torch.gt(self.exp_masks.expand(-1, -1, -1, self.input_dim), 0)
        
        # we forward all masked inputs to experts and collect them here
        expert_outs = []
        num_experts = self.num_experts - int(self.hole)
        for i in range(num_experts):
            
            expert_input = x * routing_mask[i]
            expert_out = self.expert_forward(self.experts[i], expert_input)
            
            # multiply each token by routing score
            expert_out = expert_out * self.exp_masks[i].expand(expert_out.shape)
            expert_outs.append(expert_out)

        
        expert_outs, _ = pack(expert_outs, "* b s d") 
        expert_outs = reduce(expert_outs, "e b s d -> b s d", reduction=self.aggregation)
        
        return expert_outs
    

    @abstractmethod
    def expert_forward(self, expert: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        This functions prepares an expert-specific forward given a masked expert input.
        Warning: only single tensor input - single tensor output supported so far.
        Args:
            expert: an expert in a MoE
            x: input for expert
        Returns:
            out: torch.Tensor -> expert output
        """
        pass


    @property
    def routing_matrix(self):
        return self.rout_matrix
    
    @property
    @torch.no_grad()
    def batched_routing_matrix(self):
        batched_routing_matrix = rearrange(self.rout_matrix,  "(batch seq) dim -> batch seq dim", batch=self.batch_size)
        return batched_routing_matrix
    

    @property
    def experts_masks(self):
        unbatched_routing_matrix = rearrange(self.exp_masks, "exp batch seq dim -> exp (batch seq) dim", batch=self.batch_size)
        return unbatched_routing_matrix
        
    
    @property
    @torch.no_grad()
    def batched_experts_masks(self):
        return self.exp_masks
    

    @torch.no_grad()
    def experts_load(self):
        """
        Return tensor expert_token_idx, such that expert_token_idx[i][j] is how many patches in position j did expert i receive. 
        """
        batched_routing_matrix = self.batched_experts_masks
        expert_token_idx = (batched_routing_matrix[..., 0]!=0).sum(1)
        return expert_token_idx.detach()
    
    
    def abs_experts_load(self):
        return self._experts_load().sum(-1)


    def __moerepr__(self):
        layer_id = self.layer_id if self.layer_id is not None else 'unk'
        return f"layer {layer_id}/ {type(self).__name__} ({self.num_experts} experts)"
    
    
    def _init_experts(self):
        for expert_idx, elem in enumerate(self.experts):
            elem.expert_id = expert_idx
            elem.mixture_id = self.__moerepr__()
            elem.is_expert = True


class Hole(nn.Module):
    """
    An expert that drops the input received and always returns zero.
    """
    def __init__(self) -> None:
        super().__init__()
    def forward(self, *args, **kwargs):
        # get first element in input sequence and return zeros like that element
        return torch.zeros(1)