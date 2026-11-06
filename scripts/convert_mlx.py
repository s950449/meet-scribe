#!/usr/bin/env python3
"""Point mlx-whisper at Breeze-ASR-25, or convert a Whisper checkpoint to MLX.

Preconverted weights already exist and are the default in meetscribe:

  schsu/breeze-asr-25-mlx
  mlx-community/whisper-large-v3-turbo

Conversion itself has to run on Apple Silicon. The official converter lives in
ml-explore/mlx-examples (whisper/convert.py). Breeze is a Hugging Face
Transformers Whisper checkpoint, so convert from that repo and rename the
weight file to weights.safetensors, which is the name mlx-whisper loads.

    git clone https://github.com/ml-explore/mlx-examples.git
    python mlx-examples/whisper/convert.py \\
        --torch-name-or-path MediaTek-Research/Breeze-ASR-25 \\
        --mlx-path ~/.cache/meetscribe/models/breeze-asr-25-mlx \\
        --dtype float16
    # If the script wrote model.safetensors, rename it:
    mv model.safetensors weights.safetensors

Then set asr.model to that directory, or pass it as --asr-model.
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
    parser.add_argument("--torch-name-or-path", default="MediaTek-Research/Breeze-ASR-25")
    parser.add_argument(
        "--mlx-path",
        default=str(Path.home() / ".cache/meetscribe/models/breeze-asr-25-mlx"),
    )
    parser.add_argument(
        "--convert-script",
        default="",
        help="Path to mlx-examples whisper/convert.py",
    )
    parser.add_argument("--dtype", default="float16")
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Run the converter instead of printing instructions.",
    )
    args = parser.parse_args()
    if not args.convert:
        print(__doc__)
        return 0
    script = Path(args.convert_script).expanduser() if args.convert_script else None
    if script is None or not script.is_file():
        print("Pass --convert-script pointing at mlx-examples/whisper/convert.py", file=sys.stderr)
        return 1
    destination = Path(args.mlx_path).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(script),
        "--torch-name-or-path",
        args.torch_name_or_path,
        "--mlx-path",
        str(destination),
        "--dtype",
        args.dtype,
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode
    misnamed = destination / "model.safetensors"
    expected = destination / "weights.safetensors"
    if misnamed.is_file() and not expected.exists():
        shutil.move(misnamed, expected)
        print(f"Renamed {misnamed.name} to {expected.name}")
    print(f"MLX checkpoint: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
