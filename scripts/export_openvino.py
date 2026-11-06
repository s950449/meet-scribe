#!/usr/bin/env python3
"""Export a Whisper checkpoint to OpenVINO IR for WhisperPipeline.

Requires optimum-intel. Breeze-ASR-25 is a Whisper Large checkpoint, so the
same automatic-speech-recognition export used for large-v3-turbo applies.
If the export rejects Breeze, export openai/whisper-large-v3-turbo and point
asr.model at that directory. meetscribe will also do that fallback when the
turbo IR exists and the Breeze IR does not.

    pip install -U "optimum-intel[openvino]"
    python scripts/export_openvino.py --convert
    python scripts/export_openvino.py --model openai/whisper-large-v3-turbo --convert

GPU inference then uses:

    meetscribe transcribe meeting.wav --asr-backend openvino
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="MediaTek-Research/Breeze-ASR-25")
    parser.add_argument("--output", default="")
    parser.add_argument("--convert", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).expanduser() if args.output else _default_output(args.model)
    if not args.convert:
        print(__doc__)
        print(f"Default output directory: {output}")
        return 0
    cli = shutil.which("optimum-cli")
    if cli is None:
        print("optimum-cli is not on PATH. pip install -U 'optimum-intel[openvino]'", file=sys.stderr)
        return 1
    output.mkdir(parents=True, exist_ok=True)
    command = [
        cli,
        "export",
        "openvino",
        "--model",
        args.model,
        "--task",
        "automatic-speech-recognition-with-past",
        "--trust-remote-code",
        str(output),
    ]
    print(" ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode
    print(f"OpenVINO model: {output}")
    return 0


def _default_output(model: str) -> Path:
    if "Breeze-ASR-25" in model:
        name = "breeze-asr-25-ov"
    elif "large-v3-turbo" in model:
        name = "whisper-large-v3-turbo-ov"
    else:
        name = "whisper-ov"
    return Path.home() / ".cache" / "meetscribe" / "models" / name


if __name__ == "__main__":
    raise SystemExit(main())
