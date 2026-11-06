"""Command line interface. Library entry points live in ``meetscribe.pipeline``."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from meetscribe import __version__
from meetscribe.audio.devices import parse_pactl_short
from meetscribe.config import AppConfig, load_config, parse_speaker_map
from meetscribe.errors import MeetscribeError
from meetscribe.events import ProgressEvent, StdioSink
from meetscribe.hardware import detect_hardware, recommended_asr, recommended_capture
from meetscribe.llm.profiles import select_active_llm
from meetscribe.pipeline import record_audio, run_meeting, summarize_file
from meetscribe.probe import describe_backend
from meetscribe.registry import build_registry

app = typer.Typer(
    help="Transcribe a meeting locally and write a Traditional Chinese summary.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="TOML config file. Secrets stay in the environment."),
    ] = None,
    json_events: Annotated[
        bool,
        typer.Option(
            "--json-events",
            help="Write JSONL progress events to stdout for a future GUI.",
        ),
    ] = False,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Print the version and exit.",
        ),
    ] = None,
) -> None:
    """Local meeting capture, transcription, and summary."""

    del version
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["json_events"] = json_events


@app.command()
def record(
    ctx: typer.Context,
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory for mic.wav and system.wav."),
    ] = Path("recording"),
    duration: Annotated[
        float | None,
        typer.Option(
            "--duration", "-d", help="Stop after this many recorded seconds, excluding pauses."
        ),
    ] = None,
    backend: Annotated[
        str | None,
        typer.Option("--backend", help="Capture backend: auto, macos, windows, linux."),
    ] = None,
    mic_device: Annotated[
        str | None, typer.Option("--mic-device", help="Microphone name or index.")
    ] = None,
    system_device: Annotated[
        str | None,
        typer.Option("--system-device", help="Monitor source or WASAPI loopback name."),
    ] = None,
    live: Annotated[
        bool,
        typer.Option(
            "--live",
            help="Transcribe chunks while recording. The final note still uses the full files.",
        ),
    ] = False,
    chunk_seconds: Annotated[
        float | None,
        typer.Option("--chunk-seconds", help="Live transcription chunk length. Default 5."),
    ] = None,
    list_devices: Annotated[
        bool, typer.Option("--list-devices", help="Print capture devices and exit.")
    ] = False,
) -> None:
    """Record mic and system audio. In a terminal, p pauses/resumes; q or Ctrl+C stops."""

    if list_devices:
        typer.echo(_device_report())
        return

    def action(sink: StdioSink) -> None:
        config = _config(
            ctx,
            capture_backend=backend,
            mic_device=mic_device,
            system_device=system_device,
            chunk_seconds=chunk_seconds,
        )
        record_audio(config, output_dir, duration_s=duration, live=live, sink=sink)

    _guard(ctx, action)


@app.command()
def transcribe(
    ctx: typer.Context,
    source: Annotated[Path, typer.Argument(help="Mic audio file, or a directory with mic.wav.")],
    system: Annotated[
        Path | None,
        typer.Option("--system", help="System-audio recording that starts with the mic file."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Markdown path. A JSON sidecar is written beside it."),
    ] = None,
    title: Annotated[str | None, typer.Option("--title", help="Meeting title.")] = None,
    asr_backend: Annotated[
        str | None, typer.Option("--asr-backend", help="ASR backend name or auto.")
    ] = None,
    asr_model: Annotated[str | None, typer.Option("--asr-model", help="ASR checkpoint id.")] = None,
    diarization: Annotated[
        str | None,
        typer.Option(
            "--diarization",
            help="Diarization backend: auto, track, single-speaker, pyannote, sherpa-onnx.",
        ),
    ] = None,
    language: Annotated[
        str | None, typer.Option("--language", help="Whisper language code. Default zh.")
    ] = None,
    speaker: Annotated[
        list[str] | None,
        typer.Option(
            "--speaker", "-s", help="Map a speaker id to a name, e.g. Me=林可安. Repeatable."
        ),
    ] = None,
    base_url: Annotated[
        str | None, typer.Option("--base-url", help="OpenAI-compatible API base URL.")
    ] = None,
    llm_provider: Annotated[
        str | None,
        typer.Option(
            "--llm-provider", help="omlx or llama-server. Apple Silicon defaults to omlx."
        ),
    ] = None,
    llm_model: Annotated[
        str | None,
        typer.Option(
            "--llm-model",
            help="Model alias. Overrides /v1/models and selects the sampling profile.",
        ),
    ] = None,
    no_polish: Annotated[bool, typer.Option("--no-polish", help="Keep the raw ASR text.")] = False,
    keep_audio: Annotated[
        bool, typer.Option("--keep-audio", help="Keep the 16 kHz wavs next to the note.")
    ] = False,
    work_dir: Annotated[
        Path | None, typer.Option("--work-dir", help="Directory for persistent checkpoints.")
    ] = None,
    resume: Annotated[
        bool, typer.Option("--resume", help="Continue from compatible completed stages.")
    ] = False,
) -> None:
    """Transcribe audio, assign speakers, and optionally polish. Does not summarize."""

    def action(sink: StdioSink) -> None:
        config = _config(
            ctx,
            asr_backend=asr_backend,
            asr_model=asr_model,
            diarization_backend=diarization,
            language=language,
            llm_base_url=base_url,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_polish=False if no_polish else None,
            speaker_map=parse_speaker_map(speaker) if speaker else None,
        )
        result = run_meeting(
            config,
            mic=source,
            system=system,
            output=output,
            title=title,
            sink=sink,
            do_summary=False,
            do_polish=not no_polish,
            keep_audio=keep_audio,
            work_dir=work_dir,
            resume=resume,
        )
        if not sink.json_events:
            sink.emit(
                ProgressEvent(
                    stage="write",
                    event="log",
                    message=f"JSON sidecar: {result.json_path}",
                )
            )

    _guard(ctx, action)


@app.command()
def summarize(
    ctx: typer.Context,
    source: Annotated[Path, typer.Argument(help="Meeting JSON written by transcribe or run.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Markdown path. Defaults to the JSON stem."),
    ] = None,
    speaker: Annotated[
        list[str] | None,
        typer.Option("--speaker", "-s", help="Map a speaker id to a name. Repeatable."),
    ] = None,
    base_url: Annotated[
        str | None, typer.Option("--base-url", help="OpenAI-compatible API base URL.")
    ] = None,
    llm_provider: Annotated[
        str | None,
        typer.Option(
            "--llm-provider", help="omlx or llama-server. Apple Silicon defaults to omlx."
        ),
    ] = None,
    llm_model: Annotated[
        str | None,
        typer.Option(
            "--llm-model",
            help="Model alias. Overrides /v1/models and selects the sampling profile.",
        ),
    ] = None,
) -> None:
    """Summarize a transcript JSON file and rewrite the Markdown note."""

    def action(sink: StdioSink) -> None:
        config = _config(
            ctx,
            llm_base_url=base_url,
            llm_provider=llm_provider,
            llm_model=llm_model,
            speaker_map=parse_speaker_map(speaker) if speaker else None,
        )
        summarize_file(config, source, output=output, sink=sink)

    _guard(ctx, action)


@app.command()
def run(
    ctx: typer.Context,
    source: Annotated[
        Path | None,
        typer.Argument(help="Mic audio, or a session directory. Omit to record first."),
    ] = None,
    system: Annotated[Path | None, typer.Option("--system", help="System-audio file.")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Markdown path.")] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Recording directory, used when no audio file is given."),
    ] = None,
    title: Annotated[str | None, typer.Option("--title", help="Meeting title.")] = None,
    duration: Annotated[
        float | None,
        typer.Option("--duration", "-d", help="Recording length in seconds, excluding pauses."),
    ] = None,
    asr_backend: Annotated[str | None, typer.Option("--asr-backend")] = None,
    asr_model: Annotated[str | None, typer.Option("--asr-model")] = None,
    diarization: Annotated[str | None, typer.Option("--diarization")] = None,
    language: Annotated[str | None, typer.Option("--language")] = None,
    speaker: Annotated[list[str] | None, typer.Option("--speaker", "-s")] = None,
    capture_backend: Annotated[str | None, typer.Option("--capture-backend")] = None,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    llm_provider: Annotated[
        str | None,
        typer.Option(
            "--llm-provider", help="omlx or llama-server. Apple Silicon defaults to omlx."
        ),
    ] = None,
    llm_model: Annotated[
        str | None,
        typer.Option(
            "--llm-model",
            help="Model alias. Overrides /v1/models and selects the sampling profile.",
        ),
    ] = None,
    no_polish: Annotated[bool, typer.Option("--no-polish")] = False,
    keep_audio: Annotated[bool, typer.Option("--keep-audio")] = False,
    live: Annotated[
        bool, typer.Option("--live", help="Stream chunk transcripts while recording.")
    ] = False,
    mic_device: Annotated[str | None, typer.Option("--mic-device")] = None,
    system_device: Annotated[str | None, typer.Option("--system-device")] = None,
    chunk_seconds: Annotated[float | None, typer.Option("--chunk-seconds")] = None,
    work_dir: Annotated[
        Path | None, typer.Option("--work-dir", help="Directory for persistent checkpoints.")
    ] = None,
    resume: Annotated[
        bool, typer.Option("--resume", help="Continue from compatible completed stages.")
    ] = False,
) -> None:
    """Record or load audio, then transcribe, polish, and summarize."""

    def action(sink: StdioSink) -> None:
        if resume and source is None:
            raise MeetscribeError(
                "Pass the original source file or recording directory with --resume."
            )
        config = _config(
            ctx,
            asr_backend=asr_backend,
            asr_model=asr_model,
            diarization_backend=diarization,
            language=language,
            capture_backend=capture_backend,
            llm_base_url=base_url,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_polish=False if no_polish else None,
            speaker_map=parse_speaker_map(speaker) if speaker else None,
            mic_device=mic_device,
            system_device=system_device,
            chunk_seconds=chunk_seconds,
        )
        mic = source
        if mic is None:
            directory = output_dir or Path("recording")
            record_audio(config, directory, duration_s=duration, live=live, sink=sink)
            mic = directory
        run_meeting(
            config,
            mic=mic,
            system=system,
            output=output,
            title=title,
            sink=sink,
            do_polish=not no_polish,
            do_summary=True,
            keep_audio=keep_audio,
            work_dir=work_dir,
            resume=resume,
        )

    _guard(ctx, action)


@app.command()
def devices() -> None:
    """List microphones and system-audio sources for this machine."""

    typer.echo(_device_report())


@app.command()
def backends(
    ctx: typer.Context,
    llm_provider: Annotated[
        str | None,
        typer.Option(
            "--llm-provider", help="omlx or llama-server. Apple Silicon defaults to omlx."
        ),
    ] = None,
    llm_model: Annotated[
        str | None,
        typer.Option(
            "--llm-model",
            help="Model alias. Overrides /v1/models and selects the sampling profile.",
        ),
    ] = None,
) -> None:
    """List backends and the hardware defaults for this machine."""

    hardware = detect_hardware()
    registry = build_registry()
    config = _config(ctx, llm_provider=llm_provider, llm_model=llm_model)
    active = select_active_llm(config)
    sampling = active.sampling
    payload = {
        "hardware": {
            "system": hardware.system,
            "machine": hardware.machine,
            "apple_silicon": hardware.apple_silicon,
            "intel_cpu": hardware.intel_cpu,
            "intel_igpu": hardware.intel_igpu,
            "recommended_asr": recommended_asr(hardware),
            "recommended_capture": recommended_capture(hardware),
            "vulkan_gpu": hardware.vulkan_gpu,
            "diarization": "track if two tracks else single-speaker",
        },
        "llm": {
            "provider": config.llm.provider,
            "active_model": active.model,
            "source": active.source,
            "profile": active.profile,
            "base_url": config.llm.base_url,
            "sampling": {
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "top_k": sampling.top_k,
                "min_p": sampling.min_p,
                "presence_penalty": sampling.presence_penalty,
            },
        },
        "backends": [
            {
                "kind": spec.kind,
                "name": spec.name,
                "description": spec.description,
                "device": spec.device,
                "implemented": status.implemented,
                "dependency_present": status.dependency_present,
                "platform_ok": status.platform_ok,
                "usable": status.usable,
                "reason": status.reason,
                "extra": spec.extra,
                "platforms": list(spec.platforms),
            }
            for spec in registry.list()
            for status in (describe_backend(spec, hardware),)
        ],
    }
    if ctx.obj.get("json_events"):
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    typer.echo(_format_backends(payload))


def _config(ctx: typer.Context, **overrides: object) -> AppConfig:
    cleaned = {key: value for key, value in overrides.items() if value is not None}
    return load_config(ctx.obj.get("config_path"), overrides=cleaned)


def _guard(ctx: typer.Context, action: Callable[[StdioSink], None]) -> None:
    sink = StdioSink(json_events=bool(ctx.obj.get("json_events")))
    try:
        action(sink)
    except MeetscribeError as exc:
        sink.emit(ProgressEvent(stage="pipeline", event="error", status="error", message=str(exc)))
        raise typer.Exit(code=1) from exc


def _format_backends(payload: dict) -> str:
    hardware = payload["hardware"]
    igpu = hardware["intel_igpu"]
    igpu_text = "unknown" if igpu is None else ("yes" if igpu else "no")
    sampling = payload["llm"]["sampling"]
    lines = [
        "Hardware",
        f"  system: {hardware['system']}",
        f"  machine: {hardware['machine']}",
        f"  apple_silicon: {'yes' if hardware['apple_silicon'] else 'no'}",
        f"  intel_cpu: {'yes' if hardware['intel_cpu'] else 'no'}",
        f"  intel_igpu: {igpu_text}",
        f"  vulkan_gpu: {'yes' if hardware.get('vulkan_gpu') else 'no'}",
        f"  recommended_asr: {hardware['recommended_asr']}",
        f"  recommended_capture: {hardware['recommended_capture']}",
        f"  diarization: {hardware['diarization']}",
        "",
        "LLM",
        f"  provider: {payload['llm']['provider']}",
        f"  base_url: {payload['llm']['base_url']}",
        f"  model: {payload['llm']['active_model']} ({payload['llm']['source']})",
        f"  profile: {payload['llm']['profile'] or 'built-in default'}",
        (
            "  sampling: "
            f"temperature={sampling['temperature']} top_p={sampling['top_p']} "
            f"top_k={sampling['top_k']} min_p={sampling['min_p']} "
            f"presence_penalty={sampling['presence_penalty']}"
        ),
    ]
    current = None
    for spec in payload["backends"]:
        if spec["kind"] != current:
            current = spec["kind"]
            lines.append("")
            lines.append(current)
        state = "implemented" if spec["implemented"] else "stub"
        usable = "usable=yes" if spec["usable"] else "usable=no"
        extra = f"  extra={spec['extra']}" if spec["extra"] else ""
        lines.append(
            f"  {spec['name']:<16} {state:<12} {usable:<12} device={spec['device']}{extra}"
        )
        lines.append(f"    {spec['description']}")
        lines.append(f"    {spec['reason']}")
    return "\n".join(lines)


def _device_report() -> str:
    import platform
    import shutil
    import subprocess

    system = platform.system()
    lines = [f"Capture devices ({system})"]
    if system == "Linux":
        pactl = shutil.which("pactl")
        if pactl is None:
            lines.append(
                "  pactl is not installed. Install PipeWire or PulseAudio, "
                "then uv sync --extra capture-linux."
            )
            return "\n".join(lines)
        completed = subprocess.run(
            [pactl, "list", "short", "sources"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if completed.returncode != 0:
            lines.append(completed.stderr.strip() or "pactl failed")
            return "\n".join(lines)
        sources = parse_pactl_short(completed.stdout)
        if not sources:
            lines.append("  No PulseAudio/PipeWire sources.")
        for source in sources:
            lines.append(f"  {source.kind:<8} {source.id:>4}  {source.name}")
        lines.append("  Record with uv sync --extra capture-linux. Ctrl+C stops.")
        return "\n".join(lines)
    if system == "Windows":
        lines.append("  WASAPI devices come from PyAudioWPatch.")
        lines.append("  uv sync --extra capture-windows")
        try:
            from meetscribe.audio.windows import list_wasapi_devices

            found, default_input, default_output = list_wasapi_devices()
        except Exception as exc:
            lines.append(f"  {exc}")
            return "\n".join(lines)
        lines.append(f"  default input: {default_input or '(none)'}")
        lines.append(f"  default output: {default_output or '(none)'}")
        for device in found:
            lines.append(f"  {device.kind:<8} {device.id:>4}  {device.name}")
        return "\n".join(lines)
    if system == "Darwin":
        lines.append(
            "  System audio is ScreenCaptureKit via scripts/macos/SystemAudioCapture.swift."
        )
        lines.append("  Microphones are Core Audio inputs. uv sync --extra capture-macos")
        lines.append(
            "  swiftc -parse-as-library -O -o ~/.local/bin/meetscribe-system-audio "
            "scripts/macos/SystemAudioCapture.swift"
        )
        return "\n".join(lines)
    lines.append("  No capture backend for this operating system. Transcribe existing wav files.")
    return "\n".join(lines)
