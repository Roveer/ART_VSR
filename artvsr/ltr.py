import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class StructureDetailDecomposer(nn.Module):

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.smooth_filter = nn.Conv3d(
            channels, channels, kernel_size=kernel_size, padding=pad, groups=channels, bias=True
        )

        with torch.no_grad():
            self.smooth_filter.weight.zero_()
            self.smooth_filter.bias.zero_()
            val = 1.0 / float(kernel_size * kernel_size * kernel_size)
            self.smooth_filter.weight.fill_(val)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_structure = self.smooth_filter(x)
        x_detail = x - x_structure
        return x_structure, x_detail

class AdaptiveDetailModulator(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        self.t_embed = nn.Sequential(
            nn.SiLU(),
            nn.Conv3d(1, in_channels * 2, kernel_size=1),
        )
        nn.init.zeros_(self.t_embed[1].weight)
        nn.init.zeros_(self.t_embed[1].bias)

    def forward(self, x: torch.Tensor, t_map: torch.Tensor) -> torch.Tensor:
        emb = self.t_embed(t_map)
        gamma, beta = torch.chunk(emb, 2, dim=1)
        return x * (1.0 + gamma) + beta

class TrajectoryRectifier(nn.Module):

    def __init__(self, channels: int = 16):
        super().__init__()
        self.drift_net = nn.Sequential(
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )
        self.modulator = AdaptiveDetailModulator(channels)

        nn.init.zeros_(self.drift_net[-1].weight)
        nn.init.zeros_(self.drift_net[-1].bias)

    def forward(self, x_detail: torch.Tensor, noise_detail: torch.Tensor, t_map: torch.Tensor) -> torch.Tensor:

        energy = torch.mean(x_detail**2, dim=1, keepdim=True)
        texture_mask = torch.tanh(energy * 5.0)

        adaptive_noise = self.modulator(noise_detail, t_map) * texture_mask
        x_mixed = x_detail + adaptive_noise

        drift = self.drift_net(x_mixed)
        return x_mixed + drift

class LatentTrajectoryRectifier(nn.Module):

    def __init__(self, latent_channels: int = 16):
        super().__init__()
        self.decomposer = StructureDetailDecomposer(latent_channels)
        self.rectifier = TrajectoryRectifier(channels=latent_channels)
        self.fusion = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, x_lq: torch.Tensor, t_map: torch.Tensor) -> torch.Tensor:
        b, c, f, h, w = x_lq.shape
        t_map_vol = self._process_tmap(t_map, f, h, w)

        x_struct, x_detail = self.decomposer(x_lq)

        noise = torch.randn_like(x_lq)
        _, noise_detail = self.decomposer(noise)

        x_detail_aligned = self.rectifier(x_detail, noise_detail, t_map_vol)

        out = x_struct + x_detail_aligned

        return x_lq + self.fusion(out - x_lq)

    def _process_tmap(self, t_map: torch.Tensor, f: int, h: int, w: int) -> torch.Tensor:

        if len(t_map.shape) < 5:
            t_map = rearrange(
                t_map,
                "b (f h_t w_t c_t) -> b c_t f h_t w_t",
                f=f,
                h_t=h // 2,
                w_t=w // 2,
                c_t=1,
            )
            t_map = F.interpolate(t_map, size=(f, h, w), mode="trilinear", align_corners=False)
        return t_map
