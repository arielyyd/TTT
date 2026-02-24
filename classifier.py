"""
Classifier-Based Guidance (CBG) network for inverse problems.

MeasurementPredictor: small UNet-style encoder-decoder following the ADM
(Dhariwal & Nichol 2021) architecture conventions:

  - Pre-activation ResBlocks: GroupNorm -> SiLU -> Conv (x2), with
    residual skip connections and zero-initialized second conv
  - FiLM sigma conditioning (scale_shift_norm) between the two convolutions
  - Self-attention at the bottleneck (16x16)
  - log(sigma) sinusoidal embedding for stable multi-scale conditioning
  - Zero-initialized output head (network starts predicting zero residual)

Takes (x_t, sigma, y) and predicts the measurement residual:

    M_phi(x_t, sigma, y)  ~  A(Tweedie(x_t, sigma)) - y

At inference the guidance loss is ||M_phi(x_t, sigma, y)||^2, whose gradient
w.r.t. x_t flows only through this small network (~10M params), NOT through
the ~300M-param diffusion model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal timestep embeddings (matches model/ddpm/nn.py)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )
    return embedding


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


# ---------------------------------------------------------------------------
# Building blocks (following ADM / model/ddpm/unet.py conventions)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Pre-activation residual block with FiLM sigma conditioning.

    Structure (matches ADM ResBlock with use_scale_shift_norm=True):
        in_layers:  GroupNorm(in_ch) -> SiLU -> Conv(in_ch -> out_ch)
        FiLM:       emb -> SiLU -> Linear -> (scale, shift)
        out_layers: GroupNorm(out_ch) -> FiLM(scale, shift) -> SiLU -> Conv
        skip:       Identity or 1x1 Conv when channels change
        output:     skip(x) + out_layers(in_layers(x))

    The second conv is zero-initialized so the block starts as identity.
    """

    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        # First half: Norm -> Act -> Conv (changes channels)
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        # Embedding projection -> scale & shift (FiLM)
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * out_ch),
        )

        # Second half: Norm -> FiLM -> Act -> zero_Conv
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act2 = nn.SiLU()
        self.conv2 = zero_module(nn.Conv2d(out_ch, out_ch, 3, padding=1))

        # Skip connection
        if in_ch == out_ch:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, emb):
        # First conv
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # FiLM conditioning (scale_shift_norm)
        emb_out = self.emb_proj(emb)
        while emb_out.ndim < h.ndim:
            emb_out = emb_out[..., None]
        scale, shift = emb_out.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift

        # Second conv (zero-initialized -> block starts as identity)
        h = self.act2(h)
        h = self.conv2(h)

        return self.skip(x) + h


class SelfAttention(nn.Module):
    """
    Multi-head self-attention with pre-norm and zero-initialized output
    projection (matches ADM AttentionBlock).

    Applied at the bottleneck (16x16) for global spatial reasoning.
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x_flat = x.reshape(b, c, -1)                    # [B, C, HW]

        h = self.norm(x_flat)
        qkv = self.qkv(h)                               # [B, 3C, HW]
        q, k, v = qkv.reshape(b, 3, self.num_heads, self.head_dim, -1).unbind(1)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhdn,bhdm->bhnm', q * scale, k)
        attn = attn.softmax(dim=-1)
        h = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        h = h.reshape(b, c, -1)

        h = self.proj(h)                                 # zero-init -> starts as identity
        return (x_flat + h).reshape(b, c, *spatial)


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------

class MeasurementPredictor(nn.Module):
    """
    Small UNet that predicts the measurement residual:

        M_phi(x_t, sigma, y)  ~  A(Tweedie(x_t, sigma)) - y

    Architecture (following ADM conventions)
    ----------------------------------------
    * y is bilinearly resized to match x_t and concatenated channel-wise
    * log(sigma) is embedded via sinusoidal encoding + MLP, injected via FiLM
    * Encoder: 4 levels, each with a ResBlock + stride-2 Downsample
    * Bottleneck: ResBlock + SelfAttention + ResBlock at 16x16
    * Decoder: mirrors encoder with skip connections + ResBlocks
    * Output: GroupNorm -> SiLU -> zero_init 1x1 Conv, then resize to out_size

    Channel progression with default channel_mult=(1, 2, 4, 4):
        Encoder: [C, 2C, 4C, 4C]  (C = base_channels = 64 -> [64, 128, 256, 256])
        Bottleneck: 4C
        Decoder mirrors encoder
    """

    def __init__(self, in_channels=3, y_channels=3, out_channels=3,
                 out_size=256, base_channels=64, emb_dim=256,
                 channel_mult=(1, 2, 4, 4), attn_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.y_channels = y_channels
        self.out_channels = out_channels
        self.out_size = (out_size, out_size) if isinstance(out_size, int) else tuple(out_size)
        self.base_channels = base_channels
        self.emb_dim = emb_dim
        self.channel_mult = list(channel_mult)

        C = base_channels
        ch_list = [C * m for m in channel_mult]  # e.g. [64, 128, 256, 256]
        num_levels = len(channel_mult)
        cat_ch = in_channels + y_channels  # input channels (default 6)
        bot_ch = ch_list[-1]  # bottleneck channels

        # -- sigma embedding (log-scale for better multi-scale coverage) -------
        self.sigma_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # -- input convolution -------------------------------------------------
        self.input_conv = nn.Conv2d(cat_ch, C, 3, padding=1)

        # -- encoder -----------------------------------------------------------
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev_ch = C
        for i, cur_ch in enumerate(ch_list):
            self.enc_blocks.append(ResBlock(prev_ch, cur_ch, emb_dim))
            self.downsamples.append(nn.Conv2d(cur_ch, cur_ch, 3, stride=2, padding=1))
            prev_ch = cur_ch

        # -- bottleneck (ResBlock + Attention + ResBlock) ----------------------
        self.bot_res1 = ResBlock(bot_ch, bot_ch, emb_dim)
        self.bot_attn = SelfAttention(bot_ch, num_heads=attn_heads)
        self.bot_res2 = ResBlock(bot_ch, bot_ch, emb_dim)

        # -- decoder -----------------------------------------------------------
        # Decoder level i receives: upsample(prev) concatenated with skip_i
        # and outputs the channel count for the next (higher-res) level.
        self.upsample_layers = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        # Iterate from deepest to shallowest
        dec_prev_ch = bot_ch
        for i in reversed(range(num_levels)):
            skip_ch = ch_list[i]
            # Output channels: ch_list of one level higher, or C for the top
            if i > 0:
                dec_out_ch = ch_list[i - 1]
            else:
                dec_out_ch = C
            self.upsample_layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            self.dec_blocks.append(ResBlock(dec_prev_ch + skip_ch, dec_out_ch, emb_dim))
            dec_prev_ch = dec_out_ch

        # -- output head (zero-init so the network starts predicting zeros) ----
        self.out_norm = nn.GroupNorm(min(32, C), C)
        self.out_act = nn.SiLU()
        self.out_conv = zero_module(nn.Conv2d(C, out_channels, 1))

    # ------------------------------------------------------------------
    def forward(self, x_t, sigma, y):
        """
        Args:
            x_t:   [B, C_in, 256, 256]  noisy image (pre-scaled by 1/s(t))
            sigma: [B] or scalar         noise level
            y:     [B, C_y, H_y, W_y]   measurement

        Returns:
            [B, out_channels, out_H, out_W]  predicted residual
        """
        B = x_t.shape[0]

        # --- sigma embedding (log-scale) ---
        if not isinstance(sigma, torch.Tensor):
            sigma_t = torch.tensor([sigma], dtype=torch.float32,
                                   device=x_t.device).expand(B)
        else:
            sigma_t = sigma.float().view(-1)
            if sigma_t.numel() == 1:
                sigma_t = sigma_t.expand(B)

        emb = timestep_embedding(sigma_t.log(), self.emb_dim)
        emb = self.sigma_mlp(emb)                        # [B, emb_dim]

        # --- resize y to match x_t spatial dims ---
        if y.shape[-2:] != x_t.shape[-2:]:
            y_in = F.interpolate(y, size=x_t.shape[-2:],
                                 mode='bilinear', align_corners=False)
        else:
            y_in = y

        # --- input conv ---
        h = self.input_conv(torch.cat([x_t, y_in], dim=1))  # [B, C, 256, 256]

        # --- encoder ---
        skips = []
        for enc, down in zip(self.enc_blocks, self.downsamples):
            h = enc(h, emb)
            skips.append(h)                               # save for decoder
            h = down(h)

        # --- bottleneck ---
        h = self.bot_res1(h, emb)
        h = self.bot_attn(h)
        h = self.bot_res2(h, emb)

        # --- decoder ---
        for up, dec in zip(self.upsample_layers, self.dec_blocks):
            h = up(h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = dec(h, emb)

        # --- output ---
        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)                              # [B, out_ch, 256, 256]

        # --- resize to target output size ---
        out_h, out_w = self.out_size
        if h.shape[-2:] != (out_h, out_w):
            h = F.interpolate(h, size=(out_h, out_w),
                              mode='bilinear', align_corners=False)
        return h


# ---------------------------------------------------------------------------
# Save / load utilities
# ---------------------------------------------------------------------------

def save_classifier(classifier, path, metadata=None):
    """Save classifier state_dict + architecture config + optional metadata."""
    checkpoint = {
        'state_dict': classifier.state_dict(),
        'config': {
            'in_channels':   classifier.in_channels,
            'y_channels':    classifier.y_channels,
            'out_channels':  classifier.out_channels,
            'out_size':      list(classifier.out_size),
            'base_channels': classifier.base_channels,
            'emb_dim':       classifier.emb_dim,
            'channel_mult':  classifier.channel_mult,
        },
    }
    if metadata:
        checkpoint.update(metadata)
    torch.save(checkpoint, path)


def load_classifier(path, device='cuda'):
    """Reconstruct MeasurementPredictor from a saved checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint['config']
    classifier = MeasurementPredictor(**config).to(device)
    classifier.load_state_dict(checkpoint['state_dict'])
    classifier.eval()
    return classifier
