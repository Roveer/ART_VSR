from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_gaussian3x3_kernel(dtype, device) -> torch.Tensor:
    values = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    return torch.tensor(values, dtype=dtype, device=device)

def _wavelet_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError("expected tensor shape (N, C, H, W)")
    _, channels, _, _ = x.shape
    base = _make_gaussian3x3_kernel(x.dtype, x.device)
    weight = base.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    x_pad = F.pad(x, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(x_pad, weight, stride=1, padding=0, dilation=radius, groups=channels)

def _wavelet_decompose(x: torch.Tensor, levels: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 4:
        raise ValueError("expected tensor shape (N, C, H, W)")
    high = torch.zeros_like(x)
    low = x
    for level in range(levels):
        radius = 2 ** level
        blurred = _wavelet_blur(low, radius)
        high = high + low - blurred
        low = blurred
    return high, low

def _wavelet_reconstruct(content: torch.Tensor, style: torch.Tensor, levels: int = 5) -> torch.Tensor:
    content_high, _ = _wavelet_decompose(content, levels=levels)
    _, style_low = _wavelet_decompose(style, levels=levels)
    return content_high + style_low

class TorchColorCorrectorWavelet(nn.Module):
    def __init__(self, levels: int = 5):
        super().__init__()
        self.levels = levels

    @staticmethod
    def _flatten_time(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        if x.dim() != 5:
            raise ValueError("expected tensor shape (B, C, F, H, W)")
        batch, channels, frames, height, width = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        return y, batch, frames

    @staticmethod
    def _unflatten_time(y: torch.Tensor, batch: int, frames: int) -> torch.Tensor:
        total, channels, height, width = y.shape
        if total != batch * frames:
            raise ValueError("flattened time dimension does not match batch and frame count")
        return y.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)

    def forward(
        self,
        hq_image: torch.Tensor,
        lq_image: torch.Tensor,
        clip_range: Tuple[float, float] = (-1.0, 1.0),
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        if hq_image.shape != lq_image.shape:
            raise ValueError("hq_image and lq_image must have the same shape")
        if hq_image.dim() != 5 or hq_image.shape[1] != 3:
            raise ValueError("expected tensor shape (B, 3, F, H, W)")

        _, _, frames, _, _ = hq_image.shape
        if chunk_size is None or chunk_size >= frames:
            hq4, batch, frames = self._flatten_time(hq_image)
            lq4, _, _ = self._flatten_time(lq_image)
            out4 = _wavelet_reconstruct(hq4, lq4, levels=self.levels)
            out4 = torch.clamp(out4, *clip_range)
            return self._unflatten_time(out4, batch, frames)

        outputs = []
        for start in range(0, frames, chunk_size):
            end = min(start + chunk_size, frames)
            hq4, batch, chunk_frames = self._flatten_time(hq_image[:, :, start:end])
            lq4, _, _ = self._flatten_time(lq_image[:, :, start:end])
            out4 = _wavelet_reconstruct(hq4, lq4, levels=self.levels)
            out4 = torch.clamp(out4, *clip_range)
            outputs.append(self._unflatten_time(out4, batch, chunk_frames))
        return torch.cat(outputs, dim=2)
