import tqdm
import torch
import numpy as np
import torch.nn as nn
from cores.trajectory import Trajectory
from cores.scheduler import get_diffusion_scheduler, DiffusionPFODE
from cores.mcmc import MCMCSampler
from forward_operator import LatentWrapper


def get_sampler(**kwargs):
    latent = kwargs.pop('latent')
    sampler_type = kwargs.pop('type', 'daps')

    if sampler_type == 'dps':
        # DPS doesn't use mcmc_sampler_config
        kwargs.pop('mcmc_sampler_config', None)
        if latent:
            return LatentDPS(**kwargs)
        return DPS(**kwargs)

    if sampler_type == 'cbg_dps':
        kwargs.pop('mcmc_sampler_config', None)
        classifier_path = kwargs.pop('classifier_path', None)
        classifier = None
        if classifier_path:
            from classifier import load_classifier
            classifier = load_classifier(classifier_path)
        return CBGDPS(classifier=classifier, **kwargs)

    # Default: DAPS
    if latent:
        return LatentDAPS(**kwargs)
    return DAPS(**kwargs)


class DAPS(nn.Module):
    """
    Decoupled Annealing Posterior Sampling (DAPS) implementation.

    Combines diffusion models and MCMC updates for posterior sampling from noisy measurements.
    """

    def __init__(self, annealing_scheduler_config, diffusion_scheduler_config, mcmc_sampler_config):
        """
        Initializes the DAPS sampler with the provided scheduler and sampler configurations.

        Args:
            annealing_scheduler_config (dict): Configuration for annealing scheduler.
            diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
            mcmc_sampler_config (dict): Configuration for MCMC sampler.
        """
        super().__init__()
        annealing_scheduler_config, diffusion_scheduler_config = self._check(annealing_scheduler_config,
                                                                             diffusion_scheduler_config)
        self.annealing_scheduler = get_diffusion_scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.mcmc_sampler = MCMCSampler(**mcmc_sampler_config)

    def sample(self, model, x_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        """
        Performs sampling using the DAPS method.

        Args:
            model (nn.Module): Diffusion model.
            x_start (torch.Tensor): Initial tensor/state.
            operator (nn.Module): Measurement operator.
            measurement (torch.Tensor): Observed measurement tensor.
            evaluator (Evaluator, optional): Evaluator for performance metrics.
            record (bool, optional): If True, records the sampling trajectory.
            verbose (bool, optional): Enables progress bar and logs.
            **kwargs:
                gt (torch.Tensor, optional): Ground truth data for evaluation.

        Returns:
            torch.Tensor: Final sampled tensor/state.
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1) if verbose else range(self.annealing_scheduler.num_steps - 1)
        xt = x_start
        for step in pbar:
            sigma = self.annealing_scheduler.sigma_steps[step]
            # 1. reverse diffusion
            with torch.no_grad():
                diffusion_scheduler = get_diffusion_scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
                sampler = DiffusionPFODE(model, diffusion_scheduler, solver='euler')
                x0hat = sampler.sample(xt)

            # 2. MCMC update
            x0y = self.mcmc_sampler.sample(xt, model, x0hat, operator, measurement, sigma, step / self.annealing_scheduler.num_steps)

            # 3. forward diffusion
            if step != self.annealing_scheduler.num_steps - 1:
                xt = x0y + torch.randn_like(x0y) * self.annealing_scheduler.sigma_steps[step + 1]
            else:
                xt = x0y

            # 4. evaluation
            x0hat_results = x0y_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x0hat_results = evaluator(gt, measurement, x0hat)
                    x0y_results = evaluator(gt, measurement, x0y)

                # record
                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat' + '_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                        'x0y' + '_' + main_eval_fn_name: f"{x0y_results[main_eval_fn_name].item():.2f}",
                    })
            if record:
                self._record(xt, x0y, x0hat, sigma, x0hat_results, x0y_results)
        return xt

    def _record(self, xt, x0y, x0hat, sigma, x0hat_results, x0y_results):
        """Records the intermediate states during sampling."""

        self.trajectory.add_tensor(f'xt', xt)
        self.trajectory.add_tensor(f'x0y', x0y)
        self.trajectory.add_tensor(f'x0hat', x0hat)
        self.trajectory.add_value(f'sigma', sigma)
        for name in x0hat_results.keys():
            self.trajectory.add_value(f'x0hat_{name}', x0hat_results[name])
        for name in x0y_results.keys():
            self.trajectory.add_value(f'x0y_{name}', x0y_results[name])

    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        """Checks and updates the configurations for the schedulers."""

        # sigma_max of diffusion scheduler change each step
        if 'sigma_max' in diffusion_scheduler_config:
            diffusion_scheduler_config.pop('sigma_max')

        return annealing_scheduler_config, diffusion_scheduler_config

    def get_start(self, batch_size, model):
        """
        Generates initial random state tensors from the Gaussian prior.

        Args:
            batch_size (int): Number of initial states to generate.
            model (nn.Module): Diffusion or latent diffusion model.

        Returns:
            torch.Tensor: Random initial tensor.
        """
        device = next(model.parameters()).device
        in_shape = model.get_in_shape()
        x_start = torch.randn(batch_size, *in_shape, device=device) * self.annealing_scheduler.get_prior_sigma()
        return x_start


class LatentDAPS(DAPS):
    """
    Latent Decoupled Annealing Posterior Sampling (LatentDAPS).

    Implements posterior sampling using a latent diffusion model combined with MCMC updates
    """
    def sample(self, model, z_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        """
        Performs sampling using LatentDAPS in latent space, decoding intermediate results.

        Args:
            model (LatentDiffusionModel): Latent diffusion model.
            z_start (torch.Tensor): Initial latent state tensor.
            operator (nn.Module): Measurement operator applied in data space.
            measurement (torch.Tensor): Observed measurement tensor.
            evaluator (Evaluator, optional): Evaluator for monitoring performance.
            record (bool, optional): Whether to record intermediate states and metrics.
            verbose (bool, optional): Enables progress bar and evaluation metrics.
            **kwargs:
                gt (torch.Tensor, optional): Ground truth data for evaluation.

        Returns:
            torch.Tensor: Final sampled data decoded from latent space.
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1) if verbose else range(self.annealing_scheduler.num_steps - 1)
        warpped_operator = LatentWrapper(operator, model)

        zt = z_start
        for step in pbar:
            sigma = self.annealing_scheduler.sigma_steps[step]
            # 1. reverse diffusion
            with torch.no_grad():
                diffusion_scheduler = get_diffusion_scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
                sampler = DiffusionPFODE(model, diffusion_scheduler, solver='euler')
                z0hat = sampler.sample(zt)
                x0hat = model.decode(z0hat)

            # 2. MCMC update
            z0y = self.mcmc_sampler.sample(zt, model, z0hat, warpped_operator, measurement, sigma, step / self.annealing_scheduler.num_steps)
            with torch.no_grad():
                x0y = model.decode(z0y)

            # 3. forward diffusion
            if step != self.annealing_scheduler.num_steps - 1:
                zt = z0y + torch.randn_like(z0y) * self.annealing_scheduler.sigma_steps[step + 1]
            else:
                zt = z0y
            with torch.no_grad():
                xt = model.decode(zt)

            # 4. evaluation
            x0hat_results = x0y_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x0hat_results = evaluator(gt, measurement, x0hat)
                    x0y_results = evaluator(gt, measurement, x0y)

                # record
                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat' + '_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                        'x0y' + '_' + main_eval_fn_name: f"{x0y_results[main_eval_fn_name].item():.2f}",
                    })
            if record:
                self._record(xt, x0y, x0hat, sigma, x0hat_results, x0y_results)
        return xt


class DPS(nn.Module):
    """
    Diffusion Posterior Sampling (Chung et al. 2023).

    Runs a single guided reverse diffusion chain using PF-ODE Euler steps
    with gradient-based measurement guidance.
    """

    def __init__(self, scheduler_config, guidance_scale=1.0):
        super().__init__()
        self.scheduler = get_diffusion_scheduler(**scheduler_config)
        self.guidance_scale = guidance_scale

    def sample(self, model, x_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        if record:
            self.trajectory = Trajectory()

        sigma_steps = self.scheduler.sigma_steps
        num_steps = len(sigma_steps) - 1  # last entry is 0
        pbar = tqdm.trange(num_steps) if verbose else range(num_steps)
        xt = x_start

        for step in pbar:
            sigma = sigma_steps[step]
            sigma_next = sigma_steps[step + 1]
            t = self.scheduler.get_sigma_inv(sigma)
            t_next = self.scheduler.get_sigma_inv(sigma_next)
            dt = t_next - t
            st = self.scheduler.get_scaling(t)
            dst = self.scheduler.get_scaling_derivative(t)
            dsigma = self.scheduler.get_sigma_derivative(t)

            # --- guidance gradient (with grad tracking through model) ---
            # Enable requires_grad on model params so checkpointed layers
            # can compute gradients w.r.t. input through the saved params.
            model.requires_grad_(True)
            xt_in = xt.detach().requires_grad_(True)
            x0hat = model.tweedie(xt_in / st, sigma)
            loss_per_sample = operator.loss(x0hat, measurement)  # [B]
            grad_xt = torch.autograd.grad(loss_per_sample.sum(), xt_in)[0]
            model.requires_grad_(False)

            # Per-sample normalization: rho / ||y - A(x0hat)||
            with torch.no_grad():
                norm_factor = loss_per_sample.sqrt()  # [B]
                norm_factor = norm_factor.view(-1, *([1] * (grad_xt.ndim - 1)))
                norm_factor = norm_factor.clamp(min=1e-8)
                normalized_grad = grad_xt / norm_factor

            # --- PF-ODE Euler step (no grad needed) ---
            with torch.no_grad():
                score = (x0hat.detach() - xt / st) / sigma ** 2
                deriv = dst / st * xt - st * dsigma * sigma * score
                xt_next = xt + dt * deriv

                # Apply guidance
                xt = xt_next - self.guidance_scale * normalized_grad

                # NaN guard
                if torch.isnan(xt).any():
                    if verbose:
                        print(f'NaN detected at step {step}, returning early.')
                    break

            # --- evaluation ---
            x0hat_eval = x0hat.detach()
            x0hat_results = x0y_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x0hat_results = evaluator(gt, measurement, x0hat_eval)
                    x0y_results = x0hat_results  # DPS has no separate x0y

                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                    })
            if record:
                self._record(xt, x0hat_eval, x0hat_eval, sigma, x0hat_results, x0y_results)

        return xt

    def _record(self, xt, x0y, x0hat, sigma, x0hat_results, x0y_results):
        self.trajectory.add_tensor('xt', xt)
        self.trajectory.add_tensor('x0y', x0y)
        self.trajectory.add_tensor('x0hat', x0hat)
        self.trajectory.add_value('sigma', sigma)
        for name in x0hat_results.keys():
            self.trajectory.add_value(f'x0hat_{name}', x0hat_results[name])
        for name in x0y_results.keys():
            self.trajectory.add_value(f'x0y_{name}', x0y_results[name])

    def get_start(self, batch_size, model):
        device = next(model.parameters()).device
        in_shape = model.get_in_shape()
        x_start = torch.randn(batch_size, *in_shape, device=device) * self.scheduler.get_prior_sigma()
        return x_start


class LatentDPS(DPS):
    """
    Latent-space Diffusion Posterior Sampling.

    Gradient chain goes through the decoder: z_t -> tweedie -> z0hat -> decode -> x0hat -> operator -> loss.
    """

    def sample(self, model, z_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        if record:
            self.trajectory = Trajectory()

        sigma_steps = self.scheduler.sigma_steps
        num_steps = len(sigma_steps) - 1
        pbar = tqdm.trange(num_steps) if verbose else range(num_steps)
        zt = z_start

        for step in pbar:
            sigma = sigma_steps[step]
            sigma_next = sigma_steps[step + 1]
            t = self.scheduler.get_sigma_inv(sigma)
            t_next = self.scheduler.get_sigma_inv(sigma_next)
            dt = t_next - t
            st = self.scheduler.get_scaling(t)
            dst = self.scheduler.get_scaling_derivative(t)
            dsigma = self.scheduler.get_sigma_derivative(t)

            # --- guidance gradient (grad through model + decoder) ---
            # Enable requires_grad on model params so checkpointed layers
            # can compute gradients w.r.t. input through the saved params.
            model.requires_grad_(True)
            zt_in = zt.detach().requires_grad_(True)
            z0hat = model.tweedie(zt_in / st, sigma)
            x0hat = model.decode(z0hat)
            loss_per_sample = operator.loss(x0hat, measurement)  # [B]
            grad_zt = torch.autograd.grad(loss_per_sample.sum(), zt_in)[0]
            model.requires_grad_(False)

            # Per-sample normalization
            with torch.no_grad():
                norm_factor = loss_per_sample.sqrt().view(-1, *([1] * (grad_zt.ndim - 1)))
                norm_factor = norm_factor.clamp(min=1e-8)
                normalized_grad = grad_zt / norm_factor

            # --- PF-ODE Euler step in latent space ---
            with torch.no_grad():
                score = (z0hat.detach() - zt / st) / sigma ** 2
                deriv = dst / st * zt - st * dsigma * sigma * score
                zt_next = zt + dt * deriv

                # Apply guidance
                zt = zt_next - self.guidance_scale * normalized_grad

                # NaN guard
                if torch.isnan(zt).any():
                    if verbose:
                        print(f'NaN detected at step {step}, returning early.')
                    break

            # --- decode for evaluation/recording ---
            with torch.no_grad():
                x0hat_eval = model.decode(z0hat.detach())
                xt = model.decode(zt)

            # --- evaluation ---
            x0hat_results = x0y_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x0hat_results = evaluator(gt, measurement, x0hat_eval)
                    x0y_results = x0hat_results

                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                    })
            if record:
                self._record(xt, x0hat_eval, x0hat_eval, sigma, x0hat_results, x0y_results)

        # Return data-space result
        with torch.no_grad():
            xt = model.decode(zt)
        return xt


class CBGDPS(DPS):
    """
    Classifier-Based Guidance DPS.

    Replaces the expensive DPS gradient (through the full diffusion model +
    forward operator) with a cheap gradient through a small
    MeasurementPredictor network.

    The classifier M_phi(x_t, sigma, y) predicts the measurement residual
    A(Tweedie(x_t)) - y.  At inference the guidance loss is
    ||M_phi(x_t, sigma, y)||^2 and its gradient only flows through the
    ~5-10M param classifier, not the ~300M param diffusion model.
    """

    def __init__(self, scheduler_config, guidance_scale=1.0, classifier=None):
        super().__init__(scheduler_config, guidance_scale)
        self.classifier = classifier

    def set_classifier(self, classifier):
        """Attach a trained MeasurementPredictor for inference."""
        self.classifier = classifier

    def sample(self, model, x_start, operator, measurement,
               evaluator=None, record=False, verbose=False, **kwargs):
        if self.classifier is None:
            raise RuntimeError("CBGDPS requires a trained classifier. "
                               "Call set_classifier() or pass classifier_path "
                               "in the config.")
        if record:
            self.trajectory = Trajectory()

        sigma_steps = self.scheduler.sigma_steps
        num_steps = len(sigma_steps) - 1
        pbar = tqdm.trange(num_steps) if verbose else range(num_steps)
        xt = x_start

        for step in pbar:
            sigma = sigma_steps[step]
            sigma_next = sigma_steps[step + 1]
            t = self.scheduler.get_sigma_inv(sigma)
            t_next = self.scheduler.get_sigma_inv(sigma_next)
            dt = t_next - t
            st = self.scheduler.get_scaling(t)
            dst = self.scheduler.get_scaling_derivative(t)
            dsigma = self.scheduler.get_sigma_derivative(t)

            # 1. Tweedie for PF-ODE (no grad — huge memory saving)
            with torch.no_grad():
                x0hat = model.tweedie(xt / st, sigma)

            # 2. Classifier gradient (grad only through small network)
            xt_in = xt.detach().requires_grad_(True)
            pred_residual = self.classifier(xt_in / st, sigma, measurement)
            loss_per_sample = pred_residual.pow(2).flatten(1).sum(-1)  # [B]
            grad_xt = torch.autograd.grad(loss_per_sample.sum(), xt_in)[0]

            # 3. Per-sample normalization (same as DPS)
            with torch.no_grad():
                norm_factor = loss_per_sample.sqrt()
                norm_factor = norm_factor.view(-1, *([1] * (grad_xt.ndim - 1)))
                norm_factor = norm_factor.clamp(min=1e-8)
                normalized_grad = grad_xt / norm_factor

            # 4. PF-ODE Euler step (using x0hat from step 1)
            with torch.no_grad():
                score = (x0hat - xt / st) / sigma ** 2
                deriv = dst / st * xt - st * dsigma * sigma * score
                xt_next = xt + dt * deriv

                # Apply guidance
                xt = xt_next - self.guidance_scale * normalized_grad

                # NaN guard
                if torch.isnan(xt).any():
                    if verbose:
                        print(f'NaN detected at step {step}, returning early.')
                    break

            # --- evaluation ---
            x0hat_eval = x0hat.detach()
            x0hat_results = x0y_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x0hat_results = evaluator(gt, measurement, x0hat_eval)
                    x0y_results = x0hat_results

                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat_' + main_eval_fn_name:
                            f"{x0hat_results[main_eval_fn_name].item():.2f}",
                    })
            if record:
                self._record(xt, x0hat_eval, x0hat_eval,
                             sigma, x0hat_results, x0y_results)

        return xt
