import torch
from einops import rearrange
from artvsr.ate import predict_timestep_map
from artvsr.ltr import LatentTrajectoryRectifier
from artvsr.wan.wan_video_dit import WanModel, sinusoidal_embedding_1d

def tile_segment_infer(
    pipe, seg_latent, seg_lq_video, context,
    tile_size, tile_overlap,
):
    B, C, tT, H, W = seg_latent.shape
    device = seg_latent.device

    tile_h, tile_w = tile_size
    assert tile_overlap >= 0, "tile_overlap must be >= 0"
    assert tile_h > tile_overlap, "tile_h must be > tile_overlap"
    assert tile_w > tile_overlap, "tile_w must be > tile_overlap"

    if H <= tile_h and W <= tile_w:
        out = one_step_infer(pipe, seg_latent, seg_lq_video, context)
        return out

    stride_h = tile_h - tile_overlap
    stride_w = tile_w - tile_overlap
    assert stride_h > 0 and stride_w > 0, "stride must be > 0"

    out = torch.zeros_like(seg_latent)
    weight = torch.zeros_like(seg_latent)

    def build_feather_mask(tile_h, tile_w, overlap):
        mask = torch.ones((1, 1, 1, tile_h, tile_w), device=device)
        if overlap > 0:
            def feather_1d(L, o):
                if o <= 0:
                    return torch.ones(L, device=device)
                o = min(o, L // 2)
                base = torch.ones(L, device=device)
                ramp = (torch.arange(o, device=device) + 0.5) / o
                base[:o] = ramp
                base[-o:] = torch.flip(ramp, dims=[0])
                return base

            mh = feather_1d(tile_h, overlap).view(1,1,1,tile_h,1)
            mw = feather_1d(tile_w, overlap).view(1,1,1,1,tile_w)
            mask = mh * mw
        return mask

    feather_mask = build_feather_mask(tile_h, tile_w, tile_overlap)

    if H < tile_h:
        h0_list = [0]
    else:
        h0_list = list(range(0, H - tile_h + 1, stride_h))

        if not h0_list or h0_list[-1] != H - tile_h:
            h0_list.append(H - tile_h)

    if W < tile_w:
        w0_list = [0]
    else:
        w0_list = list(range(0, W - tile_w + 1, stride_w))
        if not w0_list or w0_list[-1] != W - tile_w:
            w0_list.append(W - tile_w)

    h0_list = sorted(set(h0_list))
    w0_list = sorted(set(w0_list))

    for h0 in h0_list:
        for w0 in w0_list:

            h1 = min(h0 + tile_h, H)
            w1 = min(w0 + tile_w, W)

            tile_latent = seg_latent[:, :, :, h0:h1, w0:w1]
            tile_lq_video = seg_lq_video[:, :, :, h0*8:h1*8, w0*8:w1*8]

            tile_out = one_step_infer(pipe, tile_latent, tile_lq_video, context)

            current_tile_h = h1 - h0
            current_tile_w = w1 - w0

            if current_tile_h != tile_h or current_tile_w != tile_w:
                fm = build_feather_mask(current_tile_h, current_tile_w, tile_overlap)
            else:
                fm = feather_mask

            out[:, :, :, h0:h1, w0:w1] += tile_out * fm
            weight[:, :, :, h0:h1, w0:w1] += fm

    weight = torch.clamp(weight, min=1e-8)
    out = out / weight

    return out

def one_step_infer(pipe, latent, lq_video, context):

    t_map = predict_timestep_map(pipe.ate, latent, lq_video)

    latents_pred = spatial_modulated_flow_step(
        dit=pipe.dit,
        ltr=pipe.ltr,
        latent=latent,
        context=context,
        t_map=t_map,
    )
    return latents_pred

def spatial_modulated_flow_step(
    dit: WanModel,
    ltr: LatentTrajectoryRectifier = None,
    latent: torch.Tensor = None,
    context: torch.Tensor = None,
    t_map = None,
):

    if ltr is not None:
        latent = ltr(latent, t_map)

    B, N = t_map.shape
    t_flat = t_map.reshape(B * N) * 1000.0

    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, t_flat).view(B, N, -1))

    t_mod = dit.time_projection(t).view(B, N, 6, 1536)

    context = dit.text_embedding(context.to(device=t_mod.device, dtype=latent.dtype))

    x = latent

    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if t_map.shape[0] != context.shape[0]:
        t_map = torch.concat([t_map] * context.shape[0], dim=0)

    x = dit.patchify(x)

    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    freqs = torch.concat([freqs], dim=0)

    for block in dit.blocks:

        x = block(x, context, t_mod, freqs)

    x = dit.head(x, t)

    x = x * t_map.unsqueeze(2)

    x = dit.unpatchify(x, (f, h, w))
    return latent - x
