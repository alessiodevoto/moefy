import math
import torch
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
from typing import Union
from .moefy import MoEBlock
from torch import nn
# from moe_transformer.moe_vit import MoEVisionTransformer
from contextlib import contextmanager



@contextmanager
def evaluating(net):
    '''Temporarily switch to evaluation mode.'''
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


@torch.no_grad()
def make_expert_mask(expert_patches: torch.Tensor, imsize: int, patch_size: int):
    """
    Args:
        expert_patches: tensor containing the "unrolled" sequence indices processed by an expert.
    
    Returns:
        mask: torch.Tensor: mask to mask image
    """
    patches_per_side = (imsize//patch_size)
    mask = np.zeros((imsize, imsize))
    
    # if expert received no patch, don't mask the image
    if expert_patches.dim() == 0:
        expert_patches = expert_patches.unsqueeze(0)
    if expert_patches.nelement() == 0:
        return mask
    
    
    for my_patch_seq_num in expert_patches:
        # get x,y coordinates of this patch in original image
        my_patch_id_x = int(my_patch_seq_num // patches_per_side)          
        my_patch_id_y = int(my_patch_seq_num % patches_per_side)

        # fill in mask at those coordinates
        my_patch_x_s, my_patch_x_e = my_patch_id_x * patch_size, my_patch_id_x * patch_size + patch_size
        my_patch_y_s, my_patch_y_e = my_patch_id_y * patch_size, my_patch_id_y * patch_size + patch_size
        mask[my_patch_x_s:my_patch_x_e, my_patch_y_s:my_patch_y_e] = 1.
    
    return mask


@torch.no_grad()
def make_experts_masks(moe_block: MoEBlock, imsize: int, patch_size: int):
    """
    Create a mask for each expert in moe_block.
    """

    # get routing matrix for this MoEBlock
    routing_matrix =  moe_block.experts_masks[..., 0][:, 1:]
    expert_masks = {}

    for i, exp in enumerate(routing_matrix):
        # indices of non zero tokens
        exp_patches = torch.argwhere(exp).squeeze()
        ar = make_expert_mask(exp_patches, imsize=imsize, patch_size=patch_size)
        expert_masks[i] = ar
    return expert_masks


@torch.no_grad()
def image_through_moe_block(image: np.ndarray, patch_size: int, moe_block: MoEBlock):
    """
    Diaplay how experts in MoEBlock moe_block are processing patches.
    """

    imsize = image.shape[0]
    num_experts = moe_block.num_experts
    cols = 4
    rows =  math.ceil(num_experts/cols)

    # get mask for each expert in this moe block
    expert_masks = make_experts_masks(moe_block, imsize=imsize, patch_size=patch_size)

    fig, ax = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.suptitle(moe_block.__moerepr__())
    ax = ax.flatten()

    for ax_idx, e in enumerate(expert_masks):
        ax[ax_idx].set_title(f'expert {e}')
        ax[ax_idx].set_axis_off()
        ax[ax_idx].imshow(image)
        ax[ax_idx].imshow(expert_masks[e], alpha=0.4, vmin=0, vmax=1)


@torch.no_grad()
def image_through_experts(image: Union[torch.Tensor, np.ndarray], model: nn.Module, patch_size: int):
    
    # check image dimensions
    if isinstance(image, np.ndarray):
        _image = torch.from_numpy(image)
    else:
        _image = image

    if _image.dim() != 3:
        raise ValueError(f"Image with dimension {image.shape} cannot be loaded")
    
    if _image.shape[-1] == 3:
        _image = _image.permute(2, 0, 1)
    
    # go through each block and diplay image
    with evaluating(model):

        # forward image once to update all routing matrices
        model(_image.unsqueeze(0).float())
        
        for block_name, block in model.named_modules():

            if isinstance(block, MoEBlock):  
    
                image_through_moe_block(
                    _image.permute(1, 2, 0).numpy(force=True), 
                    patch_size=patch_size,
                    moe_block=block)
                plt.show()


def display_experts_load(moe_block: Union[MoEBlock, str], expert_loads: torch.Tensor):
    num_experts = expert_loads.shape[0]

    cols = 4
    rows =  math.ceil(num_experts/cols)
    fig, ax = plt.subplots(rows, cols, figsize = (5*cols, 4*rows), sharey=True) 
    fig.suptitle(moe_block.__moerepr__() if isinstance(moe_block, MoEBlock) else moe_block)
    ax = ax.flatten()

    # for each expert, plot the number of tokens that that expert received in each position
    for expert_id in range(num_experts):
        ax[expert_id].bar(x=torch.arange(expert_loads.shape[1]), height=expert_loads[expert_id])
        ax[expert_id].set_xlabel('token position in sequence')
        ax[expert_id].set_ylabel('num tokens')
        ax[expert_id].set_title(f'expert {expert_id}')
        ax[expert_id].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[expert_id].yaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    return fig


def display_abs_experts_load(moe_block: Union[MoEBlock, str], expert_loads: torch.Tensor):
    exp_name = moe_block.__moerepr__() if isinstance(moe_block, MoEBlock) else moe_block
    # create a plot for each expert in this moe block
    experts_abs_load = expert_loads.sum(-1)
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(x=torch.arange(experts_abs_load.shape[0]), height=experts_abs_load)
    ax.set_xlabel('expert')
    ax.set_ylabel('num tokens')
    ax.set_title(f' {exp_name}: tokens per expert')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return fig 
