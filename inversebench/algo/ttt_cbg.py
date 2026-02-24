"""
TTT-CBG evaluation plugin for InverseBench.

Loads a pre-trained MeasurementPredictor (CBG classifier) and runs
CBGDPS-style guided sampling: the guidance gradient flows through the
small classifier network (~10M params) instead of the full diffusion
model (~300M params).

Usage via InverseBench:
    python main.py problem=inv-scatter pretrain=inv-scatter algorithm=ttt_cbg
"""

import sys
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from .base import Algo
from utils.scheduler import Scheduler

# classifier.py lives in the repo root (copied there by setup script)
_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from classifier import load_classifier


class TTTCBG(Algo):
    """Inference algorithm using a pre-trained CBG classifier.

    Runs reverse diffusion (PF-ODE Euler) with classifier-based guidance.
    At each step:
      1. Tweedie denoising (no grad, cheap)
      2. Classifier gradient (grad through ~10M param network)
      3. PF-ODE Euler step + normalized guidance

    This is much faster per-step than DPS because the gradient does NOT
    flow through the full diffusion model.
    """

    def __init__(self, net, forward_op, classifier_path,
                 diffusion_scheduler_config, guidance_scale=1.0, sde=False):
        super().__init__(net, forward_op)
        self.classifier = load_classifier(
            classifier_path,
            device=next(net.parameters()).device)
        self.classifier.eval()
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.guidance_scale = guidance_scale
        self.sde = sde

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device

        if num_samples > 1:
            obs = observation.repeat(
                num_samples, *([1] * (observation.ndim - 1)))
        else:
            obs = observation

        # Initial noise
        x = torch.randn(
            num_samples, self.net.img_channels,
            self.net.img_resolution, self.net.img_resolution,
            device=device
        ) * self.scheduler.sigma_max

        pbar = tqdm(range(self.scheduler.num_steps), desc="TTT-CBG sampling")
        for i in pbar:
            sigma = self.scheduler.sigma_steps[i]
            scaling = self.scheduler.scaling_steps[i]
            factor = self.scheduler.factor_steps[i]
            scaling_factor = self.scheduler.scaling_factor[i]

            # 1. Tweedie denoising (no grad — huge memory saving)
            with torch.no_grad():
                denoised = self.net(
                    x / scaling,
                    torch.as_tensor(sigma).to(device))

            # 2. Classifier gradient (grad only through small network)
            x_in = x.detach().requires_grad_(True)
            pred_residual = self.classifier(
                x_in / scaling,
                torch.as_tensor(sigma).to(device),
                obs)
            loss_per_sample = pred_residual.pow(2).flatten(1).sum(-1)
            grad_x = torch.autograd.grad(loss_per_sample.sum(), x_in)[0]

            # 3. Per-sample normalization
            with torch.no_grad():
                norm_factor = loss_per_sample.sqrt()
                norm_factor = norm_factor.view(
                    -1, *([1] * (grad_x.ndim - 1)))
                norm_factor = norm_factor.clamp(min=1e-8)
                normalized_grad = grad_x / norm_factor

            # 4. PF-ODE Euler step
            with torch.no_grad():
                score = (denoised - x / scaling) / sigma ** 2 / scaling

                if self.sde:
                    epsilon = torch.randn_like(x)
                    x = (x * scaling_factor + factor * score
                         + np.sqrt(factor) * epsilon)
                else:
                    x = x * scaling_factor + factor * score * 0.5

                # Apply guidance
                x = x - self.guidance_scale * normalized_grad

                # NaN guard
                if torch.isnan(x).any():
                    break

        return x
