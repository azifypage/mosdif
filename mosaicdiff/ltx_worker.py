"""Run the LTX-2.5 Uncensored Turbo inpainting & restoration graph with ComfyUI's Python.

Supports LTX-2.5 DiT (including Q4_K_M GGUF and safetensors), Gemma-4 text encoder,
and LTX Video VAE.
Pre-baked with Eros10 NSFW LoRA and DMD Distilled LoRA for 4-8 step fast sampling.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


def _boot() -> None:
    import logging

    import comfy.options

    comfy.options.args_parsing = False
    import comfy.cli_args as cli_args

    cli_args.args.use_sage_attention = True
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    if os.name == "nt" and cli_args.args.cuda_device is None and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if os.name == "nt":
        os.environ["MIMALLOC_PURGE_DELAY"] = "0"
    try:
        import cuda_malloc  # noqa: F401
    except ImportError:
        pass

    try:
        import server
        if hasattr(server, "PromptServer") and getattr(server.PromptServer, "instance", None) is None:
            class DummyPromptServer:
                def __init__(self):
                    self.routes = None
                    self.app = None
                    self.loop = None
                    self.sockets = {}
                    self.supports = []
                def send_sync(self, *a, **k): pass
                def add_routes(self, *a, **k): pass
                def add_on_prompt_handler(self, *a, **k): pass
            server.PromptServer.instance = DummyPromptServer()
    except Exception:
        pass

    return


def _enable_dynamic_vram() -> None:
    """Native ComfyUI model management handles GPU/CPU offloading automatically."""
    pass


def _load_unet(job: dict):
    model_path = str(job["unet"])
    if model_path.endswith(".gguf"):
        print(f"Loading GGUF Diffusion Model: {Path(model_path).name}", flush=True)

        # Register model folder in folder_paths so get_full_path can locate it
        try:
            import folder_paths
            model_parent = str(Path(model_path).parent)
            for folder_key in ("unet", "diffusion_models", "unet_gguf"):
                try:
                    folder_paths.add_model_folder_path(folder_key, model_parent)
                except Exception:
                    pass
        except Exception:
            pass

        # 1. Direct package import of ComfyUI-GGUF
        import importlib.util
        comfy_root = Path(job.get("comfy_root", ""))
        candidate_dirs = [
            comfy_root / "custom_nodes" / "ComfyUI-GGUF",
            comfy_root / "custom_nodes" / "ComfyUI_GGUF",
            comfy_root / "custom_nodes" / "comfyui-gguf",
            comfy_root / "custom_nodes" / "comfyui_gguf",
            comfy_root / "custom_nodes" / "comfyui-gguf-loader",
            Path(__file__).resolve().parents[1] / "comfy_nodes" / "ComfyUI-GGUF",
        ]
        for c_dir in candidate_dirs:
            init_file = c_dir / "__init__.py"
            if init_file.is_file():
                try:
                    pkg_name = "ComfyUI_GGUF_ext"
                    if pkg_name not in sys.modules:
                        spec = importlib.util.spec_from_file_location(
                            pkg_name,
                            init_file,
                            submodule_search_locations=[str(c_dir)],
                        )
                        if spec and spec.loader:
                            mod = importlib.util.module_from_spec(spec)
                            sys.modules[pkg_name] = mod
                            spec.loader.exec_module(mod)
                    else:
                        mod = sys.modules[pkg_name]

                    # A. Try UnetLoaderGGUF node class
                    if hasattr(mod, "NODE_CLASS_MAPPINGS") and "UnetLoaderGGUF" in mod.NODE_CLASS_MAPPINGS:
                        loader_cls = mod.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]
                        try:
                            return loader_cls().load_unet(Path(model_path).name)[0]
                        except Exception:
                            pass
                        try:
                            return loader_cls().load_unet(model_path)[0]
                        except Exception:
                            pass

                    # B. Direct state dict load via nodes submodule
                    nodes_mod = sys.modules.get(f"{pkg_name}.nodes")
                    if nodes_mod and hasattr(nodes_mod, "GGMLOps") and hasattr(nodes_mod, "gguf_sd_loader"):
                        ops = nodes_mod.GGMLOps()
                        sd = nodes_mod.gguf_sd_loader(model_path)
                        import comfy.sd
                        return comfy.sd.load_diffusion_model_state_dict(sd, model_options={"custom_operations": ops})
                except Exception as exc:
                    print(f"Notice: loading {c_dir.name} directly failed: {exc}", flush=True)

        # 2. Try direct import from custom_nodes if available
        try:
            from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF
            try:
                return UnetLoaderGGUF().load_unet(Path(model_path).name)[0]
            except Exception:
                return UnetLoaderGGUF().load_unet(model_path)[0]
        except Exception:
            pass

        # 3. Check registered nodes.NODE_CLASS_MAPPINGS as fallback
        import nodes
        if hasattr(nodes, "NODE_CLASS_MAPPINGS") and "UnetLoaderGGUF" in nodes.NODE_CLASS_MAPPINGS:
            loader_cls = nodes.NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]
            try:
                return loader_cls().load_unet(Path(model_path).name)[0]
            except Exception:
                return loader_cls().load_unet(model_path)[0]

        err_msg = (
            f"GGUF loader (ComfyUI-GGUF) tidak dapat dimuat untuk file '{Path(model_path).name}'. "
            "Pastikan node 'ComfyUI-GGUF' telah terpasang di folder custom_nodes ComfyUI dan pustaka 'gguf' sudah terinstall."
        )
        raise RuntimeError(err_msg)
    else:
        import comfy.sd
        print(f"Loading Diffusion Model: {Path(model_path).name}", flush=True)
        return comfy.sd.load_diffusion_model(model_path, model_options={})
def _load_clip(job: dict):
    import comfy.sd
    clip_path = str(job["clip"])
    print(f"Loading Text Encoder: {Path(clip_path).name}", flush=True)
    clip_type = getattr(comfy.sd.CLIPType, "LTXV", None)
    if clip_type is None:
        clip_type = getattr(comfy.sd.CLIPType, "GEMMA", None)
    try:
        return comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=None,
            clip_type=clip_type,
        )
    except Exception as exc:
        print(f"Notice: load_clip with {clip_type} failed ({exc}), falling back to auto-detect", flush=True)
        return comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=None,
        )


def _load_vae(job: dict):
    import comfy.sd
    import comfy.utils
    vae_path = str(job["video_vae"])
    print(f"Loading Video VAE: {Path(vae_path).name}", flush=True)
    vae_sd, vae_metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
    vae = comfy.sd.VAE(sd=vae_sd, metadata=vae_metadata)
    vae.throw_exception_if_invalid()
    return vae


def _load(job: dict):
    _enable_dynamic_vram()
    import torch
    import comfy.sd
    import comfy.utils
    from comfy_extras.nodes_custom_sampler import (
        BasicGuider,
        BasicScheduler,
        KSamplerSelect,
        RandomNoise,
        SamplerCustomAdvanced,
    )
    from nodes import VAEDecode
    from PIL import Image

    clip = _load_clip(job)
    vae = _load_vae(job)
    model = _load_unet(job)

    lora_path = job.get("lora")
    if lora_path and Path(lora_path).is_file():
        print(f"Loading optional LoRA: {Path(lora_path).name}", flush=True)
        lora, lora_metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
        model, _clip = comfy.sd.load_lora_for_models(model, None, lora, 1.0, 0, lora_metadata=lora_metadata)
    else:
        print("Base model has baked-in LoRAs (Eros10 NSFW & DMD Distilled). Running directly.", flush=True)

    _watch_progress()
    _watch_vae(vae)
    _watch_text(clip)

    return {
        "torch": torch,
        "Image": Image,
        "clip": clip,
        "vae": vae,
        "base_model": model,
        "decode": VAEDecode(),
        "guider": BasicGuider,
        "scheduler": BasicScheduler,
        "sampler": KSamplerSelect,
        "noise": RandomNoise,
        "sample": SamplerCustomAdvanced,
    }


def _bundled_node(job: dict, folder: str, filename: str) -> Path:
    candidates = []
    nodes = str(job.get("nodes_dir") or "").strip()
    if nodes:
        candidates.append(Path(nodes) / folder / filename)
    candidates.append(Path(__file__).resolve().parents[1] / "comfy_nodes" / folder / filename)
    candidates.append(Path("/content/mosdif/comfy_nodes") / folder / filename)
    candidates.append(Path("/content/MosaicDiff/comfy_nodes") / folder / filename)
    candidates.append(Path(job["comfy_root"]) / "custom_nodes" / folder / filename)
    for c in candidates:
        if c.is_file():
            return c
    return Path(job["comfy_root"]) / "custom_nodes" / folder / filename


def _gpu_line() -> str:
    import torch

    if not torch.cuda.is_available():
        return "GPU n/a"
    free, total = torch.cuda.mem_get_info()
    return f"{(total - free) / 1024 ** 3:.1f} GiB used, {free / 1024 ** 3:.1f} GiB free"


def _free_other_models() -> None:
    import comfy.model_management as model_management

    model_management.unload_all_models()
    model_management.soft_empty_cache()


def _watch_progress() -> None:
    import comfy.utils

    last = {"step": None}

    def hook(current, total, preview=None, node_id=None):
        step = int(current)
        if step == last["step"]:
            return
        last["step"] = step
        print(f"Sampling step {step}/{int(total)}", flush=True)

    comfy.utils.set_progress_bar_global_hook(hook)


def _ensure_5d_latent(tensor, torch=None):
    """Normalize a latent tensor to strictly 5D [Batch=1, Channels=128, Time, Height, Width] for LTX Video VAE."""
    if hasattr(tensor, "unbind") and not (torch is not None and isinstance(tensor, torch.Tensor)):
        try:
            tensor = tensor.unbind()[0]
        except Exception:
            pass

    if not hasattr(tensor, "dim"):
        return tensor

    if tensor.dim() == 5:
        # If batch dim is not 1 and first dim is 128 (channels), unsqueeze batch dim 0
        if tensor.shape[0] == 128 and tensor.shape[1] != 128:
            return tensor.unsqueeze(0)
        return tensor

    if tensor.dim() == 4:
        # Case 1: [Channels=128, T, H, W] -> unsqueeze batch dim 0 -> [1, 128, T, H, W]
        if tensor.shape[0] == 128:
            return tensor.unsqueeze(0)
        # Case 2: [Batch=1, Channels=128, H, W] -> single-frame video: [1, 128, 1, H, W]
        elif tensor.shape[1] == 128 and tensor.shape[0] == 1:
            return tensor.unsqueeze(2)
        # Case 3: [T, Channels=128, H, W] -> [1, 128, T, H, W]
        elif tensor.shape[1] == 128:
            return tensor.permute(1, 0, 2, 3).unsqueeze(0)
        else:
            return tensor.unsqueeze(0)

    if tensor.dim() == 3:
        # [Channels=128, H, W] -> [1, 128, 1, H, W]
        return tensor.unsqueeze(0).unsqueeze(2)

    return tensor


def _watch_vae(vae) -> None:
    encode = vae.encode
    decode = vae.decode

    def encode_logged(pixel_samples, *args, **kwargs):
        shape_str = f" shape {list(pixel_samples.shape)}" if hasattr(pixel_samples, "shape") else ""
        print(f"VAE encode ({_gpu_line()}){shape_str}", flush=True)
        _free_other_models()
        started = time.perf_counter()
        out = encode(pixel_samples, *args, **kwargs)
        print(f"VAE encode finished in {time.perf_counter() - started:.1f}s", flush=True)
        return out

    def decode_logged(samples, *args, **kwargs):
        import torch
        samples = _ensure_5d_latent(samples, torch)
        shape_str = f" shape {list(samples.shape)}" if hasattr(samples, "shape") else ""
        print(f"VAE decode ({_gpu_line()}){shape_str}", flush=True)
        _free_other_models()
        started = time.perf_counter()
        out = decode(samples, *args, **kwargs)
        print(f"VAE decode finished in {time.perf_counter() - started:.1f}s", flush=True)
        return out

    vae.encode = encode_logged
    vae.decode = decode_logged


def _watch_text(clip) -> None:
    encode = clip.encode_from_tokens_scheduled

    def encode_logged(*args, **kwargs):
        print(f"Text encoder ({_gpu_line()})", flush=True)
        _free_other_models()
        started = time.perf_counter()
        out = encode(*args, **kwargs)
        print(f"Text encoder finished in {time.perf_counter() - started:.1f}s", flush=True)
        return out

    clip.encode_from_tokens_scheduled = encode_logged


def _read_frames(torch, image_cls, paths: list[str]):
    import numpy as np

    frames = []
    for path in paths:
        image = image_cls.open(path).convert("RGB")
        frames.append(torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).float().div_(255.0))
    return torch.stack(frames, dim=0)


def _pin_reference_border(loaded, frames, width: int, height: int, mask_path: str | None) -> dict:
    """Encode the reference video into LTX latents and attach a noise mask for inpainting."""
    torch = loaded["torch"]
    import comfy.utils

    # batch: [T, C, H, W] -> [1, T, H, W, C]
    batch = comfy.utils.common_upscale(frames.movedim(-1, 1), width, height, "lanczos", "disabled")
    pixel_samples = batch.movedim(1, -1)
    # LTX VAE encodes [B, T, H, W, C] or [T, H, W, C]
    try:
        encoded = loaded["vae"].encode(pixel_samples.unsqueeze(0))
    except Exception:
        encoded = loaded["vae"].encode(pixel_samples)

    if hasattr(encoded, "samples"):
        encoded = encoded.samples

    encoded = _ensure_5d_latent(encoded, torch)
    lat_t = encoded.shape[2]
    lat_h = encoded.shape[3]
    lat_w = encoded.shape[4]

    mask = _denoise_mask(torch, mask_path, lat_t, lat_h, lat_w)
    mask = mask.to(device=encoded.device, dtype=torch.float32)

    latent = {
        "samples": encoded,
        "noise_mask": mask,
    }

    held = float((mask <= 0).float().mean())
    print(f"LTX noise mask holds {held:.0%} of area (preserves border & unmasked region)", flush=True)
    return latent


def _denoise_mask(torch, mask_path: str | None, frames: int, height: int, width: int):
    import cv2

    if mask_path and Path(mask_path).is_file():
        image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = None
    if image is None:
        border = max(1, min(height, width) // 10)
        mask = torch.ones((1, 1, frames, height, width), dtype=torch.float32)
        mask[:, :, :, :border, :] = 0
        mask[:, :, :, -border:, :] = 0
        mask[:, :, :, :, :border] = 0
        mask[:, :, :, :, -border:] = 0
        return mask
    small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    plane = torch.from_numpy(small.astype("float32") / 255.0).view(1, 1, 1, height, width)
    return plane.expand(1, 1, frames, height, width).contiguous()


def _rtx_node(job: dict):
    path = _bundled_node(job, "comfyui_nvidia_rtx_nodes", "__init__.py")
    if not path.is_file():
        raise RuntimeError(f"RTX Video Super Resolution node not found: {path}")
    spec = importlib.util.spec_from_file_location("mosaicdiff_rtx_video_superres", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"RTX Video Super Resolution node could not be loaded: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _rtx_fit(job: dict, images, gen_w: int, gen_h: int, crop_w: int, crop_h: int):
    """Upscale frames with RTX Video Super Resolution if available, or direct Lanczos."""
    import comfy.utils

    _free_other_models()
    if importlib.util.find_spec("nvvfx") is not None:
        try:
            cover = max(crop_w / gen_w, crop_h / gen_h, 2.0)
            scale = min(4.0, cover)
            print(
                f"RTX upscale {gen_w}x{gen_h} by {scale:.2f}, then fit {crop_w}x{crop_h}",
                flush=True,
            )
            module = _rtx_node(job)
            upscaled = module.RTXVideoSuperResolution.execute(
                images,
                {"resize_type": module.UpscaleType.SCALE_BY, "scale": scale, "width": 0, "height": 0},
                "ULTRA",
            )[0]
            fitted = comfy.utils.common_upscale(
                upscaled.movedim(-1, 1),
                crop_w,
                crop_h,
                "lanczos",
                "disabled",
            ).movedim(1, -1)
            del upscaled
            return fitted
        except Exception as exc:
            print(f"Notice: RTX Super Resolution fallback to Lanczos ({exc})", flush=True)

    return comfy.utils.common_upscale(
        images.movedim(-1, 1),
        crop_w,
        crop_h,
        "lanczos",
        "disabled",
    ).movedim(1, -1)


def _read_crop(frame, crop: list[int]):
    x1, y1, x2, y2 = (int(value) for value in crop)
    image = frame[y1:y2, x1:x2]
    if image.size == 0:
        raise RuntimeError(f"Empty crop {x1},{y1}-{x2},{y2}")
    return image.copy()


def _stream_windows(loaded, job: dict) -> None:
    import cv2

    windows = list(job["windows"])
    pending = [
        {"window": window, "need": set(int(frame) for frame in window["frame_indices"]), "got": {}}
        for window in windows
    ]
    capture = cv2.VideoCapture(job["video"])
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {job['video']}")
    frame_idx = 0
    try:
        while pending:
            ok, frame = capture.read()
            if not ok:
                break
            ready = []
            for item in pending:
                if frame_idx not in item["need"]:
                    continue
                item["got"][frame_idx] = _read_crop(frame, item["window"]["crop"])
                if len(item["got"]) == len(item["need"]):
                    ready.append(item)
            for item in ready:
                number = windows.index(item["window"]) + 1
                print(f"Sample {number}/{len(windows)}", flush=True)
                _restore_window(loaded, job, item["window"], item["got"])
                item["got"].clear()
                pending.remove(item)
            frame_idx += 1
    finally:
        capture.release()
    if pending:
        missing = sorted(pending[0]["need"] - set(pending[0]["got"]))[:8]
        raise RuntimeError(f"VSR video ended before frame(s) {missing}")


def _restore_window(loaded, job: dict, window: dict, crops: dict | None = None) -> None:
    torch = loaded["torch"]
    started = time.perf_counter()
    if crops is None:
        frames = _read_frames(torch, loaded["Image"], window["frames"])
    else:
        import numpy as np

        ordered = []
        for frame_idx in window["frame_indices"]:
            bgr = crops[int(frame_idx)]
            rgb = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).float().div_(255.0)
            ordered.append(rgb)
        frames = torch.stack(ordered, dim=0)

    length = int(frames.shape[0])
    width = int(window["width"])
    height = int(window["height"])

    print(
        f"Conditioning {length} frames, generation {width}x{height}, reference {frames.shape[2]}x{frames.shape[1]}",
        flush=True,
    )

    # Encode prompt for LTX
    tokens = loaded["clip"].tokenize(job["prompt"])
    cond = loaded["clip"].encode_from_tokens_scheduled(tokens)
    # Inject frame_rate into conditioning
    positive = []
    for item in cond:
        c = item[1].copy()
        c["frame_rate"] = 24.0
        positive.append([item[0], c])

    latent = _pin_reference_border(loaded, frames, width, height, window.get("mask"))

    steps = int(job.get("steps", 4))
    seed = int(job.get("seed", 42))
    print(f"Sampling {steps} steps (LTX-2.5 Turbo, seed {seed})", flush=True)

    guider = loaded["guider"].execute(loaded["base_model"], positive)[0]
    sigmas = loaded["scheduler"].execute(loaded["base_model"], "simple", steps, 1.0)[0]
    sampler = loaded["sampler"].execute("euler")[0]
    noise = loaded["noise"].execute(seed)[0]
    sampled = loaded["sample"].execute(noise, guider, sampler, sigmas, latent)[0]

    # Handle NestedTensor or pure tensor output
    samples_tensor = sampled["samples"]
    samples_tensor = _ensure_5d_latent(samples_tensor, torch)
    print(f"Samples tensor shape for decode: {list(samples_tensor.shape)}", flush=True)

    try:
        images = loaded["decode"].decode(loaded["vae"], {"samples": samples_tensor})[0]
    except Exception as exc:
        print(f"Notice: nodes.VAEDecode failed ({exc}), falling back to direct vae.decode", flush=True)
        images = loaded["vae"].decode(samples_tensor)

    # If decoded video is 5D [1, T, H, W, C], remove batch dim to get [T, H, W, C]
    if hasattr(images, "dim"):
        if images.dim() == 5:
            images = images.squeeze(0)
        elif images.dim() == 3:
            images = images.unsqueeze(0)

    if images.shape[0] != length:
        # Trim or pad if subtle rounding mismatch occurred
        if images.shape[0] > length:
            images = images[:length]
        else:
            last_img = images[-1:]
            needed = length - images.shape[0]
            images = torch.cat([images, last_img.repeat(needed, 1, 1, 1)], dim=0)

    crop = window.get("crop")
    if crop:
        crop_w = int(crop[2]) - int(crop[0])
        crop_h = int(crop[3]) - int(crop[1])
    else:
        crop_w, crop_h = width, height

    images = _rtx_fit(job, images, width, height, crop_w, crop_h)
    out_dir = Path(window["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing {length} frames", flush=True)
    for index in range(length):
        array = images[index].clamp(0, 1).mul(255).round().to(dtype=torch.uint8).cpu().numpy()
        loaded["Image"].fromarray(array).save(out_dir / f"{index:06d}.png")
    print(f"Window finished in {time.perf_counter() - started:.1f}s", flush=True)
    del frames, positive, latent, sampled, images
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    os.chdir(job["comfy_root"])
    sys.path.insert(0, job["comfy_root"])
    _boot()
    print(f"Comfy process started ({_gpu_line()})", flush=True)
    loaded = _load(job)
    windows = job["windows"]
    with loaded["torch"].no_grad():
        if job.get("video"):
            _stream_windows(loaded, job)
        else:
            for index, window in enumerate(windows, start=1):
                print(f"Sample {index}/{len(windows)}", flush=True)
                _restore_window(loaded, job, window)
    print("LTX-2.5 pass finished", flush=True)


if __name__ == "__main__":
    main()
