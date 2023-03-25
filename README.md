## Custom MoE blocks

Create a custom MoE block from any torch Module, by subclassing the abstract class `MoEBlock`.
- Define the list of experts and override the `expert_forward` method. 
- The module will automatically deal with routing the tokens and computing balancing losses (if needed).

For example, to create a MoE of Linear layers.

```python

from moefy.moefy import MoEBlock

class CustomMoEBlock(MoEBlock):
    def __init__(self, input_dim, output_dim, num_experts, *args, **kwargs):
       
        # initialize experts 
        # beware that input_dim must be provided as argument to superclass
        experts = [nn.Linear(
            in_features=input_dim,
            out_features=output_dim) for _ in range(num_experts )]
        super().__init__(input_dim = input_dim, experts=experts, *args, **kwargs)

    # eaxh of the experts shall be called like this
    def expert_forward(self, expert: nn.Module, x: torch.Tensor) -> torch.Tensor:
        out =  expert(x)
        return out

```

Provide arguments about the routing mechanism,like `expert choice`, token `aggregation`, `noise` that must be added to routing matrix and so on ... 

```python
lin_moe = CustomMoEBlock(
    k = 1,
    num_experts=3,
    input_dim=4, 
    output_dim=5,
    expert_choice=False,           # Should experts choose tokens ? 
    hole=True,                     # Should we have an expert that drops tokens ?
    noise = 'gumbel',              # Noise to be added to routing matrix ['simple', 'gaussian']
    aux_criterion = 'entropy',     # Balancing loss ['prob', 'entropy']
    aggregation = 'sum',           # How do we aggregate tokens ['mean', 'sum'] 
    layer_id = 0                   # just to keep track of this layer's position inside a bigger model
    )

lin_moe
```


## Visualize experts load in any MoE block

It is possible to visualize the expert load for each Mixture, that is, which and how many tokens are distributed to each expert.

```python
from utils import display_experts_load

display_experts_load(moe_attn_block, expert_loads)
display_abs_experts_load(moe_attn_block, expert_loads)
```

![load](https://github.com/alessiodevoto/moe-transformer/blob/main/images/expert_load.png)
![total_load](https://github.com/alessiodevoto/moe-transformer/blob/main/images/total_expert_load.png)

## Visualize patches in Vision model


```python

from torch.nn import Sequential, Linear
from einops.layers.torch import Rearrange
from moefy.utils import image_through_experts
from matplotlib import pyplot as plt

patch_size = 60
hidden_dim = 96 

# a simple transformer like model for vision
moe_model = Sequential(
    Rearrange("b c (h s1) (w s2) -> b (h w) (s1 s2 c)", s1=patch_size, s2=patch_size), # make patches
    Linear(patch_size * patch_size * 3, hidden_dim),
    CustomMoEBlock(k=1, num_experts=4,input_dim=hidden_dim, output_dim=hidden_dim, hole=False, layer_id=1),
    Linear(hidden_dim, hidden_dim),
    CustomMoEBlock(k=1, num_experts=3,input_dim=hidden_dim, output_dim=hidden_dim, hole=True, layer_id=3),
    Linear(hidden_dim, 10)
)

```

One can visualize how tokens are distributed across experts like so:

```python 
image = plt.imread("./images/owl.jpg")
image_through_experts(image=image, model=moe_model, patch_size=patch_size)
```

![attention](https://github.com/alessiodevoto/moe-transformer/blob/main/images/experts_att.png)
![mlp](https://github.com/alessiodevoto/moe-transformer/blob/main/images/experts_mlp.png)
![vitblock](https://github.com/alessiodevoto/moe-transformer/blob/main/images/experts_vitblock.png)


#### Code TODOs
[ ] add requirements 
[ ] change maplotlib to plotly 