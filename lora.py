"""
LoRA (Low-Rank Adaptation) for Conv1d layers in diffusion model attention blocks.

Provides utilities to inject, remove, and manage LoRA adapters on the UNet
attention layers (qkv and proj_out) for test-time training.
"""

import torch
import torch.nn as nn
from model.ddpm.unet import AttentionBlock


class LoRAConv1d(nn.Module):
    """Wraps a frozen Conv1d(in, out, 1) with a low-rank adapter."""

    def __init__(self, original_conv, rank=4, alpha=1.0):
        super().__init__()
        self.original_conv = original_conv
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_channels = original_conv.in_channels
        out_channels = original_conv.out_channels

        self.lora_down = nn.Conv1d(in_channels, rank, 1, bias=False)
        self.lora_up = nn.Conv1d(rank, out_channels, 1, bias=False)

        # kaiming init for down, zero init for up (LoRA starts at identity)
        nn.init.kaiming_uniform_(self.lora_down.weight)
        nn.init.zeros_(self.lora_up.weight)

        # freeze the original conv
        for p in self.original_conv.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original_conv(x) + self.scaling * self.lora_up(self.lora_down(x))


def apply_lora(model, rank=4, alpha=1.0):
    """Inject LoRA adapters into all AttentionBlock qkv and proj_out layers.

    Walks ``model.model.model`` (the UNet inside VPPrecond) and replaces
    Conv1d projections with LoRAConv1d wrappers.

    Returns:
        list[LoRAConv1d]: All newly created LoRA modules.
    """
    unet = model.model.model  # DDPM -> VPPrecond -> UNet
    lora_modules = []

    for module in unet.modules():
        if isinstance(module, AttentionBlock):
            for attr_name in ("qkv", "proj_out"):
                orig_conv = getattr(module, attr_name)
                lora_conv = LoRAConv1d(orig_conv, rank=rank, alpha=alpha)
                lora_conv = lora_conv.to(next(orig_conv.parameters()).device)
                setattr(module, attr_name, lora_conv)
                lora_modules.append(lora_conv)

    return lora_modules


def remove_lora(model):
    """Restore original Conv1d layers, undoing apply_lora."""
    unet = model.model.model
    for module in unet.modules():
        if isinstance(module, AttentionBlock):
            for attr_name in ("qkv", "proj_out"):
                wrapper = getattr(module, attr_name)
                if isinstance(wrapper, LoRAConv1d):
                    setattr(module, attr_name, wrapper.original_conv)


def get_lora_params(lora_modules):
    """Return a flat list of all trainable LoRA parameters."""
    params = []
    for m in lora_modules:
        params.extend(m.lora_down.parameters())
        params.extend(m.lora_up.parameters())
    return params


def frozen_tweedie(model, lora_modules, x, sigma):
    """Evaluate Tweedie with LoRA scaling temporarily zeroed.

    This gives the frozen (pre-LoRA) model prediction without needing
    a separate model copy.
    """
    saved = []
    for m in lora_modules:
        saved.append(m.scaling)
        m.scaling = 0.0

    with torch.no_grad():
        out = model.tweedie(x, sigma)

    for m, s in zip(lora_modules, saved):
        m.scaling = s

    return out
