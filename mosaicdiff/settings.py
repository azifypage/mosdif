"""Saved window settings."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mosaicdiff.paths import (
    BUNDLED_NAMES,
    LOCAL_DEFAULTS,
    MODEL_LABELS,
    default_output_dir,
    find_model_file,
    is_frozen,
    models_dir,
    settings_path,
)

DEFAULT_PROMPT = "head_swap:\nFACE:\nPhotorealistic human face, natural skin texture, sharp detailed eyes."
DEFAULT_NEGATIVE = "mouth movement, cartoon, anime, illustration, 3d render, plastic, painting, blurry, deformed"

PATH_KEYS = (
    "unet",
    "lora",
    "clip",
    "vae",
    "comfy_python",
    "comfy_root",
)


@dataclass
class Settings:
    output_dir: str = ""
    head_image: str = ""
    h3_seconds: int = 3
    h3_resolution: int = 512
    h3_steps: int = 6
    compare: bool = False
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = 42
    denoise: float = 0.40
    lora_strength: float = 0.80
    paths: dict[str, str] = field(default_factory=dict)

    @property
    def ltx_seconds(self) -> int:
        return self.h3_seconds

    @ltx_seconds.setter
    def ltx_seconds(self, value: int):
        self.h3_seconds = value

    @property
    def ltx_resolution(self) -> int:
        return self.h3_resolution

    @ltx_resolution.setter
    def ltx_resolution(self, value: int):
        self.h3_resolution = value

    @property
    def ltx_steps(self) -> int:
        return self.h3_steps

    @ltx_steps.setter
    def ltx_steps(self, value: int):
        self.h3_steps = value

    def resolved(self, key: str) -> Path:
        # The shipped exe only reads weights from the models folder beside it.
        if is_frozen() and key in BUNDLED_NAMES:
            return models_dir() / BUNDLED_NAMES[key]
        chosen = (self.paths.get(key) or "").strip()
        if chosen and Path(chosen).exists():
            return Path(chosen)

        if key in BUNDLED_NAMES:
            c_root = None
            if "comfy_root" in self.paths:
                c_root = Path(self.paths["comfy_root"])
            elif "comfy_root" in LOCAL_DEFAULTS:
                c_root = LOCAL_DEFAULTS["comfy_root"]
            found = find_model_file(key, c_root)
            if found is not None:
                return found
            name = BUNDLED_NAMES.get(key)
            if name is not None:
                return models_dir() / name

        if chosen:
            return Path(chosen)
        return LOCAL_DEFAULTS.get(key, models_dir() / f"{key}.safetensors")

    def missing(self) -> list[tuple[str, Path]]:
        gone = []
        for key in ("unet", "lora", "clip", "vae"):
            path = self.resolved(key)
            if not path.exists():
                gone.append((MODEL_LABELS.get(key, key), path))
        return gone

    def save(self) -> None:
        payload = {
            "output_dir": self.output_dir,
            "head_image": self.head_image,
            "h3_seconds": self.h3_seconds,
            "h3_resolution": self.h3_resolution,
            "h3_steps": self.h3_steps,
            "compare": self.compare,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "denoise": self.denoise,
            "lora_strength": self.lora_strength,
            "paths": self.paths,
        }
        settings_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> "Settings":
        path = settings_path()
        if not path.is_file():
            settings = cls()
            if is_frozen():
                settings.output_dir = str(default_output_dir())
            return settings
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            settings = cls()
            if is_frozen():
                settings.output_dir = str(default_output_dir())
            return settings
        settings = cls(
            output_dir=str(data.get("output_dir") or ""),
            head_image=str(data.get("head_image") or ""),
            h3_seconds=int(data.get("h3_seconds") or 3),
            h3_resolution=int(data.get("h3_resolution") or 512),
            h3_steps=int(data.get("h3_steps") or 6),
            compare=False,
            prompt=str(data.get("prompt") or ""),
            negative_prompt=str(data.get("negative_prompt") or ""),
            seed=int(data.get("seed") or 42),
            denoise=float(data.get("denoise") or 0.40),
            lora_strength=float(data.get("lora_strength") or 0.80),
            paths={key: str(value) for key, value in dict(data.get("paths") or {}).items()},
        )
        settings.h3_seconds = min(15, max(1, settings.h3_seconds))
        settings.h3_resolution = min(1280, max(384, settings.h3_resolution - settings.h3_resolution % 32))
        settings.h3_steps = min(16, max(2, settings.h3_steps))
        if is_frozen():
            settings.output_dir = str(default_output_dir())
            for key in BUNDLED_NAMES:
                settings.paths.pop(key, None)
        return settings
