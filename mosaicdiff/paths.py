"""Where MosaicDiff keeps settings, and where it looks for model files."""

from __future__ import annotations

import sys
from pathlib import Path

APP_DIR_NAME = "MosaicDiff"

# Names used when someone drops the weights next to the installed program.
BUNDLED_NAMES = {
    "vsr": "basicvsr.pth",
    "detector": "rfdetr.onnx",
    "unet": "ltx2.5-Stubelius_remix_v1_Q4_K_S.gguf",
    "lora": "lora.safetensors",
    "clip": "gemma4_12b_ltx25_uncensored-int8.safetensors",
    "vae": "ltx25_uncensored_video_vae.safetensors",
}

# ComfyUI stays an install. Weight files are resolved from models\ next to the program.
LOCAL_DEFAULTS = {
    "comfy_python": Path(sys.executable) if sys.platform != "win32" else Path(r"E:\Workspace\ume52\scripts\venv\Scripts\python.exe"),
    "comfy_root": Path("/content/ComfyUI") if sys.platform != "win32" else Path(r"E:\Workspace\ume52\ComfyUI"),
}

MODEL_LABELS = {
    "vsr": "BasicVSR++ checkpoint",
    "detector": "Mosaic detector",
    "unet": "LTX-2.5 Stubelius Remix v1 DiT",
    "lora": "Optional LoRA (Baked into GGUF by default)",
    "clip": "Gemma-4 LTX-2.5 Text Encoder",
    "vae": "LTX-2.5 Video VAE",
    "comfy_python": "ComfyUI Python",
    "comfy_root": "ComfyUI folder",
}


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def install_dir() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def worker_script() -> Path:
    if is_frozen():
        for cand in ("ltx_worker.py", "h3_worker.py"):
            p = Path(getattr(sys, "_MEIPASS")) / "mosaicdiff" / cand
            if p.is_file():
                return p
        return Path(getattr(sys, "_MEIPASS")) / "mosaicdiff" / "ltx_worker.py"
    for cand in ("ltx_worker.py", "h3_worker.py"):
        p = Path(__file__).resolve().parent / cand
        if p.is_file():
            return p
    return Path(__file__).resolve().parent / "ltx_worker.py"


def settings_path() -> Path:
    """The shipped program keeps its own settings beside the exe.

    A shared AppData file would point it at the development folders above Shipped.
    """
    if is_frozen():
        return install_dir() / "settings.json"
    if sys.platform == "win32":
        root = Path.home() / "AppData" / "Roaming" / APP_DIR_NAME
    else:
        root = Path.home() / ".config" / APP_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root / "settings.json"


def models_dir() -> Path:
    return install_dir() / "models"


ALTERNATIVE_FILENAMES = {
    "vsr": ["basicvsr.pth", "lada_mosaic_restoration_model_generic_v1.2.pth"],
    "detector": ["rfdetr.onnx", "rfdetr-v6.onnx", "rfdetr-v6-large.onnx"],
    "unet": [
        "ltx2.5-Stubelius_remix_v1_Q4_K_S.gguf",
        "ltx2.5-Stubelius_remix_v1_int8_convrot.safetensors",
        "ltx2.5-Stubelius_remix_beta2_int8_convrot.safetensors",
        "ltx25StubeliusRemix_beta2Int8.safetensors",
        "ltx2.5-Stubelius_remix_v1_bf16.safetensors",
        "ltx2.5-Stubelius_remix_beta1.safetensors",
        "ltx25_uncensored_v1.1-Q4_K_M.gguf",
        "ltx25_uncensored_v1.1-Q6_K.gguf",
        "ltx25_uncensored_v1.1-Q8_0.gguf",
        "ltx25_uncensored_v1.1-fp8_scaled.safetensors",
        "ltx25_uncensored_v1.1-fp8.safetensors",
        "ltx25_uncensored_v1.1-int8.safetensors",
        "unet.gguf",
        "unet.safetensors",
        "10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors",
    ],
    "lora": ["lora.safetensors"],
    "clip": [
        "gemma4_12b_ltx25_uncensored-int8.safetensors",
        "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        "gemma4_12b_ltx25_uncensored-Q4_K_M.gguf",
        "gemma4_12b_ltx25_uncensored-Q6_K.gguf",
        "gemma4_12b_ltx25_uncensored-Q8_0.gguf",
        "gemma4-12b_with-proj-ltx-uncensored-int8.safetensors",
        "clip.safetensors",
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    ],
    "vae": [
        "ltx25_uncensored_video_vae.safetensors",
        "ltx-2.5-video-vae-bf16.safetensors",
        "vae.safetensors",
        "minimax_h3_video_vae_fp16.safetensors",
    ],
}


def find_model_file(key: str, comfy_root: Path | None = None) -> Path | None:
    """Search for model file across mosdif/models, MosaicDiff/models, and ComfyUI folders."""
    candidate_dirs: list[Path] = [
        models_dir(),
        Path("/content/mosdif/models"),
        Path("/content/MosaicDiff/models"),
    ]
    if comfy_root and comfy_root.is_dir():
        if key == "unet":
            candidate_dirs.extend([
                comfy_root / "models" / "diffusion_models",
                comfy_root / "models" / "unet",
            ])
        elif key == "clip":
            candidate_dirs.extend([
                comfy_root / "models" / "text_encoders",
                comfy_root / "models" / "clip",
            ])
        elif key == "vae":
            candidate_dirs.append(comfy_root / "models" / "vae")
        elif key == "lora":
            candidate_dirs.append(comfy_root / "models" / "loras")

    names = ALTERNATIVE_FILENAMES.get(key, [BUNDLED_NAMES.get(key, "")])
    for folder in candidate_dirs:
        if not folder.is_dir():
            continue
        for name in names:
            if not name:
                continue
            candidate = folder / name
            if candidate.is_file():
                return candidate

    # Dynamic fallback discovery if exact filename differs
    for folder in candidate_dirs:
        if not folder.is_dir():
            continue
        try:
            files = sorted(folder.iterdir())
        except Exception:
            continue
        if key == "unet":
            for f in files:
                if f.is_file() and f.suffix.lower() in [".gguf", ".safetensors"]:
                    l = f.name.lower()
                    if any(w in l for w in ["stubelius", "remix", "ltx25", "ltx-2.5", "uncensored"]):
                        return f
        elif key == "clip":
            for f in files:
                if f.is_file() and f.suffix.lower() in [".gguf", ".safetensors"]:
                    l = f.name.lower()
                    if any(w in l for w in ["gemma", "text_encoder", "clip"]):
                        return f
        elif key == "vae":
            for f in files:
                if f.is_file() and f.suffix.lower() in [".safetensors", ".pt"]:
                    if "vae" in f.name.lower():
                        return f

    return None


def nodes_dir() -> Path:
    """Custom Comfy nodes shipped with MosaicDiff, not taken from the Comfy install."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS")) / "comfy_nodes"
    return Path(__file__).resolve().parents[1] / "comfy_nodes"


def default_output_dir() -> Path:
    return install_dir() / "Output"


def bundled_model(key: str) -> Path | None:
    found = find_model_file(key)
    if found is not None:
        return found
    name = BUNDLED_NAMES.get(key)
    if name is None:
        return None
    path = models_dir() / name
    return path if path.is_file() else None
