from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from einops import rearrange

def _temporal_align_4n1_to_latent(x: torch.Tensor, target_f: Optional[int] = None) -> torch.Tensor:

    b, c, t, h, w = x.shape
    first = x[:, :, :1]
    rest = x[:, :, 1:]
    if rest.shape[2] <= 0:
        out = first
    else:
        num_chunks = rest.shape[2] // 4
        if num_chunks <= 0:
            out = first
        else:
            rest = rest[:, :, : num_chunks * 4]
            rest = rest.view(b, c, num_chunks, 4, h, w).mean(dim=3)
            out = torch.cat([first, rest], dim=2)

    if target_f is not None and out.shape[2] != target_f:

        out = F.interpolate(out, size=(target_f, h, w), mode="trilinear", align_corners=False)
    return out

class PixelFrequencyPath(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()

        self.freq_gate = nn.Sequential(
            nn.Conv2d(in_channels * 2, 16, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, in_channels, 1),
            nn.Sigmoid(),
        )

        self.to_latent_space = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(16, 32, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(32, out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.SiLU(),
        )

    def forward(self, x_rgb: torch.Tensor, target_f: Optional[int] = None) -> torch.Tensor:

        b, c, t, h, w = x_rgb.shape

        x_in = x_rgb.float()

        x_fft = torch.fft.rfft2(x_in, dim=(-2, -1), norm="ortho")
        wf = x_fft.shape[-1]

        x_fft_bt = x_fft.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, wf)
        cat_freq = torch.cat([x_fft_bt.real, x_fft_bt.imag], dim=1).contiguous()

        gate_dtype = self.freq_gate[0].weight.dtype
        cat_freq = cat_freq.to(dtype=gate_dtype)
        gate = self.freq_gate(cat_freq)

        gate = gate.to(dtype=x_fft_bt.real.dtype)

        x_fft_gated = x_fft_bt * gate
        x_fft_gated = x_fft_gated.view(b, t, c, h, wf).permute(0, 2, 1, 3, 4)

        x_spatial = torch.fft.irfft2(x_fft_gated, s=(h, w), dim=(-2, -1), norm="ortho")
        x_spatial = x_spatial.to(dtype=x_rgb.dtype)

        feat = self.to_latent_space(x_spatial)

        feat = _temporal_align_4n1_to_latent(feat, target_f=target_f)
        return feat

class AdaptiveTimestepEstimator(nn.Module):

    def __init__(self, latent_channels: int = 16, rgb_channels: int = 3, base_channels: int = 64):
        super().__init__()

        self.latent_branch = nn.Sequential(
            nn.Conv3d(latent_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.GroupNorm(8, base_channels),
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
        )

        self.rgb_branch = PixelFrequencyPath(in_channels=rgb_channels, out_channels=base_channels)

        self.fusion = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels, 1),
            nn.SiLU(),
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv3d(base_channels, 1, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )

        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, x_latent: torch.Tensor, x_rgb: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat_lat = self.latent_branch(x_latent)

        if x_rgb is None:

            feat_rgb = torch.zeros_like(feat_lat)
        else:

            feat_rgb = self.rgb_branch(x_rgb, target_f=feat_lat.shape[2])
            if feat_rgb.shape[-3:] != feat_lat.shape[-3:]:
                feat_rgb = F.interpolate(
                    feat_rgb,
                    size=feat_lat.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                )

        fused = torch.cat([feat_lat, feat_rgb], dim=1)
        out = self.fusion(fused)
        return out

def heuristic_physical_prior(rgb_video: torch.Tensor, target_shape: Optional[tuple[int, int, int]] = None) -> torch.Tensor:

    b, c, t, h, w = rgb_video.shape

    if c == 3:
        x_gray = 0.299 * rgb_video[:, 0:1] + 0.587 * rgb_video[:, 1:2] + 0.114 * rgb_video[:, 2:3]
    else:
        x_gray = rgb_video

    k_size = 7
    pad = k_size // 2
    x_padded = F.pad(x_gray, (pad, pad, pad, pad, 0, 0), mode="replicate")
    mu = F.avg_pool3d(x_padded, kernel_size=(1, k_size, k_size), stride=1, padding=0)
    mu_sq = F.avg_pool3d(x_padded**2, kernel_size=(1, k_size, k_size), stride=1, padding=0)
    sigma_sq = torch.relu(mu_sq - mu**2)
    texture_energy = torch.sqrt(sigma_sq + 1e-6)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x_gray.dtype, device=x_gray.device).view(1, 1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x_gray.dtype, device=x_gray.device).view(1, 1, 1, 3, 3)
    x_sobel_padded = F.pad(x_gray, (1, 1, 1, 1, 0, 0), mode="replicate")
    grad_x = F.conv3d(x_sobel_padded, sobel_x, padding=0)
    grad_y = F.conv3d(x_sobel_padded, sobel_y, padding=0)
    struct_energy = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

    x_fft = torch.fft.rfft2(x_gray.float(), dim=(-2, -1), norm="ortho")
    wf = x_fft.shape[-1]
    fy = torch.fft.fftfreq(h, d=1.0, device=x_gray.device).view(1, 1, 1, h, 1)
    fx = torch.fft.rfftfreq(w, d=1.0, device=x_gray.device).view(1, 1, 1, 1, wf)
    rr = torch.sqrt(fy**2 + fx**2)

    cutoff = 0.35
    mask = torch.sigmoid((rr - cutoff) * 40.0)
    x_fft_hp = x_fft * mask
    x_hp = torch.fft.irfft2(x_fft_hp, s=(h, w), dim=(-2, -1), norm="ortho").to(dtype=x_gray.dtype)
    fft_energy = torch.abs(x_hp)

    raw_energy = texture_energy * 0.8 + struct_energy * 0.2 + fft_energy * 0.6
    raw_energy_padded = F.pad(raw_energy, (4, 4, 4, 4, 0, 0), mode="replicate")
    energy_smooth = F.avg_pool3d(raw_energy_padded, kernel_size=(1, 9, 9), stride=1, padding=0)
    mag = torch.tanh(energy_smooth * 4.0)

    if target_shape is None:
        target_f, target_h, target_w = t, h // 16, w // 16
    else:
        target_f, target_h, target_w = target_shape

    mag_spatial = F.adaptive_avg_pool3d(mag, (t, target_h, target_w))

    mag_aligned = _temporal_align_4n1_to_latent(mag_spatial, target_f=target_f)
    mag_aligned = torch.clamp(mag_aligned, 0.0, 1.0)
    return mag_aligned

def predict_timestep_map(adaptive_timestep_estimator: nn.Module, lq_latent: torch.Tensor, lq_video: torch.Tensor):

    N_map = adaptive_timestep_estimator(lq_latent, lq_video)
    N_map = torch.sigmoid(N_map)
    b, _, f, ht, wt = N_map.shape

    H_map = heuristic_physical_prior(lq_video, target_shape=(f, ht, wt))

    t_map = H_map * 0.25 + N_map * 0.75
    t_map = torch.clamp(t_map, 0.0, 1.0)

    t_map_tokens = rearrange(t_map, "b c f h w -> b (f h w c)")
    return t_map_tokens
