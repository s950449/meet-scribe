#!/usr/bin/env python3
"""Convert MediaTek Breeze-ASR-25 (or another HF Whisper model) to GGML.

whisper.cpp ships models/convert-h5-to-ggml.py. This wrapper downloads the
Hugging Face snapshot when huggingface_hub is installed and runs that script.
There is no maintained public GGML build of Breeze-ASR-25. large-v3-turbo
GGML weights are published by ggerganov/whisper.cpp as ggml-large-v3-turbo.bin
and can be dropped in ~/.cache/meetscribe/models/ without conversion.

    git clone https://github.com/ggml-org/whisper.cpp.git
    uv run --with huggingface_hub --with torch --with transformers \
        python scripts/convert_ggml.py --convert --whisper-cpp whisper.cpp

Vulkan runtime (separate from conversion):

    cmake -B whisper.cpp/build -S whisper.cpp -DGGML_VULKAN=1
    cmake --build whisper.cpp/build --target whisper-cli -j
    export MEETSCRIBE_WHISPER_CPP=$PWD/whisper.cpp/build/bin/whisper-cli

Or build the Python binding against that tree:

    GGML_VULKAN=1 uv pip install --no-binary pywhispercpp pywhispercpp
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="MediaTek-Research/Breeze-ASR-25")
    parser.add_argument(
        "--whisper-cpp",
        default="",
        help="Checkout that contains models/convert-h5-to-ggml.py",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output .bin path. Default: ~/.cache/meetscribe/models/ggml-breeze-asr-25.bin",
    )
    parser.add_argument("--convert", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).expanduser() if args.output else _default_output(args.model)
    if not args.convert:
        print(__doc__)
        print(f"Default output path: {output}")
        return 0
    checkout = Path(args.whisper_cpp).expanduser() if args.whisper_cpp else None
    if checkout is None or not (checkout / "models" / "convert-h5-to-ggml.py").is_file():
        print("Pass --whisper-cpp pointing at a whisper.cpp checkout.", file=sys.stderr)
        return 1
    snapshot = _download(args.model)
    command = [
        sys.executable,
        str(checkout / "models" / "convert-h5-to-ggml.py"),
        str(snapshot),
        str(checkout / "models"),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode
    produced = _newest_bin(snapshot, checkout)
    if produced is None:
        print("Conversion finished but no .bin file was found.", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, output)
    print(f"GGML model: {output}")
    return 0


def _default_output(model: str) -> Path:
    name = "ggml-breeze-asr-25.bin" if "Breeze-ASR-25" in model else "ggml-model.bin"
    return Path.home() / ".cache" / "meetscribe" / "models" / name


def _download(model: str) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "uv run --with huggingface_hub python scripts/convert_ggml.py --convert"
        ) from exc
    path = snapshot_download(model)
    return Path(path)


def _newest_bin(*roots: Path) -> Path | None:
    found = [path for root in roots for path in root.rglob("*.bin")]
    if not found:
        return None
    return max(found, key=lambda path: path.stat().st_mtime)


if __name__ == "__main__":
    raise SystemExit(main())
