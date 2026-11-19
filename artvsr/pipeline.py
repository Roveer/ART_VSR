from pathlib import Path
from typing import Optional, Union

import torch
from diffsynth.utils import BasePipeline
from diffsynth.vram_management import AutoWrappedLinear, AutoWrappedModule, WanAutoCastLayerNorm, enable_vram_management
from PIL import Image
from tqdm import tqdm

from artvsr.checkpoint import DEFAULT_CHECKPOINT, load_artvsr_checkpoint
from artvsr.postprocess import TorchColorCorrectorWavelet
from artvsr.spatial_flow import tile_segment_infer
from artvsr.wan.wan_video_dit import RMSNorm, WanModel
from artvsr.wan.wan_video_vae import CausalConv3d, RMS_norm, Upsample, WanVideoVAE


class ARTVSRPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.ate = None
        self.ltr = None
        self.prompt_emb_read = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
    ):
        pipe = cls(device=device, torch_dtype=torch_dtype)
        load_artvsr_checkpoint(pipe, checkpoint_path)
        pipe.ate.eval()
        pipe.ltr.eval()
        pipe.dit.eval()
        pipe.vae.eval()
        return pipe

    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                    torch.nn.Embedding: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )

    @torch.no_grad()
    def run_video(
        self,
        lq_video: Optional[list[Image.Image]] = None,
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (56, 56),
        tile_stride: Optional[tuple[int, int]] = (32, 32),
        sliding_window_size: Optional[tuple[int, int]] = (56, 64),
        sliding_window_overlap: Optional[int] = 8,
        chunk_size=9,
        overlap=1,
        color_fix=True,
    ):
        self.load_models_to_device(["dit"])
        self.load_models_to_device(["vae"])
        orig_width, orig_height = lq_video[0].size
        orig_frame_count = len(lq_video)
        target_h = ((orig_height + 15) // 16) * 16
        target_w = ((orig_width + 15) // 16) * 16
        remainder = (orig_frame_count - 1) % 4
        pad_frames = (4 - remainder) % 4 if remainder != 0 else 0
        print(f"[Pre-process] Input: {orig_width}x{orig_height}, {orig_frame_count} frames.")
        print(f"[Pre-process] Target: {target_w}x{target_h}, {orig_frame_count + pad_frames} frames (Padding: {pad_frames}).")
        processed_input = []
        for img in lq_video:
            if (img.width, img.height) != (target_w, target_h):
                new_img = Image.new(img.mode, (target_w, target_h), (0, 0, 0))
                new_img.paste(img, (0, 0))
                processed_input.append(new_img)
            else:
                processed_input.append(img)
        if pad_frames > 0:
            processed_input.extend([processed_input[-1]] * pad_frames)
        lq_video = self.preprocess_video(processed_input)
        b, c, t, h, w = lq_video.shape
        stride = chunk_size - overlap
        chunk_starts = []
        start = 0
        while start < t:
            chunk_starts.append(start)
            if start + chunk_size >= t:
                break
            start += stride
        print(f"[Chunk Info] Total frames: {t}, Chunk size: {chunk_size}, Overlap: {overlap}, Stride: {stride}")
        print(f"[Chunk Info] Chunk starts: {chunk_starts}")
        final_results = []
        current_idx = 0
        for i, start_idx in enumerate(tqdm(chunk_starts, desc="Processing Video Chunks")):
            end_idx = min(start_idx + chunk_size, t)
            chunk_lq_video = lq_video[:, :, start_idx:end_idx, :, :]
            actual_chunk_size = end_idx - start_idx
            if actual_chunk_size < chunk_size:
                print(f"[Warning] Chunk {i} has only {actual_chunk_size} frames (expected {chunk_size})")
            chunk_latents = self.vae.encode(
                chunk_lq_video.to(torch.bfloat16),
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=self.torch_dtype, device=self.device)
            self.ate.to(device=chunk_latents.device, dtype=chunk_latents.dtype)
            self.ltr.to(device=chunk_latents.device, dtype=chunk_latents.dtype)
            chunk_output_latent = tile_segment_infer(
                self,
                chunk_latents,
                chunk_lq_video,
                self.prompt_emb_read,
                sliding_window_size,
                sliding_window_overlap,
            )
            self.load_models_to_device(["vae"])
            video_output_chunk = self.vae.decode(
                chunk_output_latent,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            if color_fix:
                color_corrector = TorchColorCorrectorWavelet(levels=5)
                video_output_chunk = color_corrector(
                    video_output_chunk.to(device=chunk_lq_video.device),
                    chunk_lq_video,
                    clip_range=(-1, 1),
                    chunk_size=16,
                )
            skip = current_idx - start_idx
            if skip < 0:
                print(f"[Error] Gap detected! current_idx {current_idx} < start_idx {start_idx}")
                skip = 0
            if i == len(chunk_starts) - 1:
                valid_frames = video_output_chunk[:, :, skip:, :, :]
            else:
                next_start = chunk_starts[i + 1]
                frames_needed = next_start - current_idx
                if frames_needed > 0:
                    valid_frames = video_output_chunk[:, :, skip:skip + frames_needed, :, :]
                else:
                    valid_frames = video_output_chunk[:, :, :0, :, :]
            current_idx += valid_frames.shape[2]
            print(f"[Chunk {i}] Input: [{start_idx}:{end_idx}], Output len: {video_output_chunk.shape[2]}, Valid len: {valid_frames.shape[2]}, Skip: {skip}")
            final_results.append(valid_frames)
        output_video = torch.cat(final_results, dim=2)
        print(f"[Chunk Merge] Total output frames before post-process: {output_video.shape[2]}")
        output_video = self.vae_output_to_video(output_video)
        print(f"[Chunk Merge] Total output frames after vae_output_to_video: {len(output_video)}")
        output_frame_count = len(output_video)
        if output_frame_count != orig_frame_count:
            print(f"[Warning] Output frame count ({output_frame_count}) != original ({orig_frame_count})")
            if output_frame_count > orig_frame_count:
                output_video = output_video[:orig_frame_count]
            elif output_frame_count < orig_frame_count:
                output_video.extend([output_video[-1]] * (orig_frame_count - output_frame_count))
        restored_video = []
        for img in output_video:
            if (img.width, img.height) != (orig_width, orig_height):
                restored_video.append(img.crop((0, 0, orig_width, orig_height)))
            else:
                restored_video.append(img)
        print(f"[Post-process] Restored to: {orig_width}x{orig_height}, {len(restored_video)} frames.")
        return restored_video


def build_artvsr_pipeline(checkpoint=DEFAULT_CHECKPOINT, device="cuda", torch_dtype=torch.bfloat16, enable_vram=True):
    pipe = ARTVSRPipeline.from_checkpoint(checkpoint_path=checkpoint, torch_dtype=torch_dtype, device=device)
    if enable_vram:
        pipe.enable_vram_management()
    return pipe


def run_artvsr_video(pipe, video_input, chunk_size=17, overlap=1, color_fix=True):
    return pipe.run_video(lq_video=video_input, chunk_size=chunk_size, overlap=overlap, color_fix=color_fix)
