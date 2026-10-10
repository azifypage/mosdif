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

PATH_KEYS = (
    "vsr",
    "detector",
    "unet",
    "clip",
    "vae",
    "comfy_python",
    "comfy_root",
)


@dataclass
class Settings:
    output_dir: str = ""
    h3_seconds: int = 4
    h3_resolution: int = 512
    h3_steps: int = 4
    compare: bool = False
    prompt: str = ""
    seed: int = 42
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
        return LOCAL_DEFAULTS[key]

    def missing(self) -> list[tuple[str, Path]]:
        gone = []
        for key in PATH_KEYS:
            path = self.resolved(key)
            if not path.exists():
                gone.append((MODEL_LABELS[key], path))
        return gone

    def save(self) -> None:
        payload = {
            "output_dir": self.output_dir,
            "h3_seconds": self.h3_seconds,
            "h3_resolution": self.h3_resolution,
            "h3_steps": self.h3_steps,
            "compare": self.compare,
            "prompt": self.prompt,
            "seed": self.seed,
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
            h3_seconds=int(data.get("h3_seconds") or 4),
            h3_resolution=int(data.get("h3_resolution") or 512),
            h3_steps=int(data.get("h3_steps") or 4),
            compare=False,
            prompt=str(data.get("prompt") or ""),
            seed=int(data.get("seed") or 42),
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
