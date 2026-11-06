"""Per-model sampling. The id from GET /v1/models picks the profile."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, replace

from meetscribe.config import DEFAULT_LLM_MODEL, AppConfig, LlmSettings, SamplingProfile


@dataclass(frozen=True, slots=True)
class ActiveLlm:
    model: str
    source: str  # cli | server | config
    sampling: SamplingProfile
    profile: str | None


def fetch_model_ids(base_url: str, api_key: str, *, timeout: float = 2.0) -> list[str] | None:
    """Return ids from ``GET {base_url}/models``, or None if the server is down."""

    url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list):
        return None
    ids: list[str] = []
    for item in data:
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
    return ids


def resolve_listed_model(
    model_ids: list[str],
    preferred: str,
    *,
    allow_singleton: bool,
) -> str | None:
    """Pick the server id that corresponds to ``preferred``.

    oMLX lists every model in the directory. The configured name may be the
    directory leaf (``gemma-4-12B-it-4bit``) or the Hugging Face repo id.
    llama-server in single-model mode lists one alias, which wins even when
    it differs from the configured name. A longer list with no match returns
    None so the caller keeps the configured id instead of ``data[0]``.
    """

    ids = [model_id.strip() for model_id in model_ids if model_id.strip()]
    if not ids:
        return None
    wanted = preferred.strip().lower()
    if wanted:
        for model_id in ids:
            if model_id.lower() == wanted:
                return model_id
        contained = [model_id for model_id in ids if model_id.lower() in wanted]
        if contained:
            contained.sort(key=len, reverse=True)
            return contained[0]
        containing = [model_id for model_id in ids if wanted in model_id.lower()]
        if containing:
            containing.sort(key=len)
            return containing[0]
    if allow_singleton and len(ids) == 1:
        return ids[0]
    return None


def fetch_loaded_model(
    base_url: str,
    api_key: str,
    *,
    timeout: float = 2.0,
    preferred: str = "",
) -> str | None:
    """Return the model id to call, or None when the server is down or ambiguous."""

    ids = fetch_model_ids(base_url, api_key, timeout=timeout)
    if ids is None:
        return None
    return resolve_listed_model(ids, preferred, allow_singleton=True)


def select_active_llm(
    config: AppConfig,
    *,
    detected: str | None = None,
    probe: bool = True,
    api_key: str | None = None,
) -> ActiveLlm:
    """Pick the model id, then the sampling profile.

    ``--llm-model`` sets ``model_locked`` and wins. Otherwise a successful
    ``/v1/models`` response wins. The configured model is the fallback.
    Explicit ``[llm]`` sampling fields replace the profile field by field.
    """

    settings = config.llm
    if settings.model_locked:
        model = settings.model
        source = "cli"
    else:
        if detected is None and probe:
            key = config.resolved_api_key() if api_key is None else api_key
            detected = fetch_loaded_model(
                settings.base_url,
                key,
                preferred=settings.model,
            )
        if detected:
            model = detected
            source = "server"
        else:
            model = settings.model
            source = "config"
    profile_name, profile = match_profile(model, settings.profiles)
    return ActiveLlm(
        model=model,
        source=source,
        sampling=apply_sampling_overrides(profile, settings),
        profile=profile_name,
    )


def match_profile(
    model_id: str,
    profiles: Mapping[str, SamplingProfile],
) -> tuple[str | None, SamplingProfile]:
    if not profiles:
        return None, SamplingProfile(temperature=1.0, top_p=0.95, top_k=64)
    exact = profiles.get(model_id)
    if exact is not None:
        return model_id, exact
    lowered = model_id.lower()
    for key, profile in profiles.items():
        if key.lower() == lowered:
            return key, profile
    hits = [
        (key, profile)
        for key, profile in profiles.items()
        if key.lower() in lowered or lowered in key.lower()
    ]
    if hits:
        hits.sort(key=lambda item: len(item[0]), reverse=True)
        return hits[0]
    fallback = profiles.get(DEFAULT_LLM_MODEL)
    if fallback is not None:
        return None, fallback
    return None, next(iter(profiles.values()))


def apply_sampling_overrides(profile: SamplingProfile, settings: LlmSettings) -> SamplingProfile:
    updates: dict[str, object] = {}
    if settings.temperature is not None:
        updates["temperature"] = settings.temperature
    if settings.top_p is not None:
        updates["top_p"] = settings.top_p
    if settings.top_k is not None:
        updates["top_k"] = settings.top_k
    if settings.min_p is not None:
        updates["min_p"] = settings.min_p
    if settings.presence_penalty is not None:
        updates["presence_penalty"] = settings.presence_penalty
    if not updates:
        return profile
    return replace(profile, **updates)


def profile_from_table(raw: Mapping[str, object], base: SamplingProfile | None = None) -> SamplingProfile:
    current = base or SamplingProfile(temperature=1.0, top_p=0.95, top_k=64)
    presence = raw.get("presence_penalty", current.presence_penalty)
    return SamplingProfile(
        temperature=float(raw["temperature"]) if "temperature" in raw else current.temperature,
        top_p=float(raw["top_p"]) if "top_p" in raw else current.top_p,
        top_k=int(raw["top_k"]) if "top_k" in raw else current.top_k,
        min_p=float(raw["min_p"]) if "min_p" in raw else current.min_p,
        presence_penalty=None if presence is None else float(presence),
    )
