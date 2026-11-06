"""Map the configured checkpoint onto each backend's on-disk format.

Breeze-ASR-25 is a Transformers Whisper checkpoint. MLX, GGML, and OpenVINO
each need a converted copy. Known public conversions are used when they exist.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from meetscribe.errors import ModelMissing

BREEZE = "MediaTek-Research/Breeze-ASR-25"
TURBO = "openai/whisper-large-v3-turbo"

# Community MLX conversions that mlx-whisper can download directly.
MLX_REPOS = {
    BREEZE: "schsu/breeze-asr-25-mlx",
    TURBO: "mlx-community/whisper-large-v3-turbo",
}

GGML_FILES = {
    BREEZE: "ggml-breeze-asr-25.bin",
    TURBO: "ggml-large-v3-turbo.bin",
}

OPENVINO_DIRS = {
    BREEZE: "breeze-asr-25-ov",
    TURBO: "whisper-large-v3-turbo-ov",
}

_OPENVINO_MARKERS = (
    "openvino_encoder_model.xml",
    "openvino_decoder_model.xml",
    "openvino_model.xml",
    "generation_config.json",
)


def default_model_dir(environ: Mapping[str, str] | None = None, home: Path | None = None) -> Path:
    env = os.environ if environ is None else environ
    configured = env.get("MEETSCRIBE_MODEL_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    home_dir = Path.home() if home is None else home
    return home_dir / ".cache" / "meetscribe" / "models"


def resolve_mlx_repo(model: str, fallback_model: str) -> tuple[str, str | None]:
    """Return an MLX repo id or a local directory, plus an optional warning."""

    selected = model.strip() or fallback_model.strip()
    local = Path(selected).expanduser()
    if selected and local.exists():
        return str(local), None
    if selected in MLX_REPOS:
        return MLX_REPOS[selected], None
    lowered = selected.lower()
    if "mlx" in lowered or selected.startswith("mlx-community/"):
        return selected, None
    fallback = MLX_REPOS.get(fallback_model.strip(), MLX_REPOS[TURBO])
    warning = (
        f"{selected} has no pinned MLX conversion. mlx-whisper will load {fallback}. "
        "Convert it with scripts/convert_mlx.py on Apple Silicon."
    )
    return fallback, warning


def resolve_ggml_file(model: str, fallback_model: str, model_dir: Path) -> tuple[Path, str | None]:
    """Return a GGML file. Breeze falls back to large-v3-turbo when only that file exists."""

    selected = model.strip() or fallback_model.strip()
    direct = Path(selected).expanduser()
    if direct.is_file():
        return direct, None

    filename = GGML_FILES.get(selected, f"{_slug(selected)}.bin")
    wanted = model_dir / filename
    if wanted.is_file():
        return wanted, None

    if selected == BREEZE:
        turbo = model_dir / GGML_FILES[TURBO]
        warning = (
            f"GGML weights for {BREEZE} were not found at {wanted}. "
            "Convert them with scripts/convert_ggml.py."
        )
        if turbo.is_file():
            return turbo, warning + f" Using {turbo.name} until then."
        raise ModelMissing(_ggml_missing(model_dir, wanted, turbo))

    raise ModelMissing(_ggml_missing(model_dir, wanted, model_dir / GGML_FILES[TURBO]))


def resolve_openvino_dir(
    model: str,
    fallback_model: str,
    model_dir: Path,
) -> tuple[Path, str | None]:
    """Return an OpenVINO IR directory exported by optimum-cli."""

    selected = model.strip() or fallback_model.strip()
    direct = Path(selected).expanduser()
    if direct.is_dir() and _is_openvino_dir(direct):
        return direct, None

    wanted_name = OPENVINO_DIRS.get(selected, OPENVINO_DIRS[TURBO])
    wanted = model_dir / wanted_name
    if _is_openvino_dir(wanted):
        return wanted, None

    if selected == BREEZE:
        turbo = model_dir / OPENVINO_DIRS[TURBO]
        warning = (
            f"OpenVINO IR for {BREEZE} was not found at {wanted}. "
            "Export it with scripts/export_openvino.py."
        )
        if _is_openvino_dir(turbo):
            return turbo, warning + f" Using {turbo.name} until then."
        raise ModelMissing(_openvino_missing(model_dir, wanted, turbo))
    raise ModelMissing(_openvino_missing(model_dir, wanted, model_dir / OPENVINO_DIRS[TURBO]))


def _is_openvino_dir(path: Path) -> bool:
    return path.is_dir() and any((path / name).is_file() for name in _OPENVINO_MARKERS)


def _slug(value: str) -> str:
    return value.replace("/", "-")


def _ggml_missing(model_dir: Path, wanted: Path, turbo: Path) -> str:
    return (
        "GGML model not found. whisper.cpp / Vulkan needs a .bin file.\n"
        f"  expected: {wanted}\n"
        f"  or the fallback: {turbo}\n"
        "Download the turbo weights from https://huggingface.co/ggerganov/whisper.cpp "
        f"(file {GGML_FILES[TURBO]}) into {model_dir}, or run scripts/convert_ggml.py "
        f"for {BREEZE}."
    )


def _openvino_missing(model_dir: Path, wanted: Path, turbo: Path) -> str:
    return (
        "OpenVINO IR directory not found.\n"
        f"  expected: {wanted}\n"
        f"  or the fallback: {turbo}\n"
        "Export with scripts/export_openvino.py. "
        "optimum-cli export openvino --model <id> --trust-remote-code <output>."
    )
