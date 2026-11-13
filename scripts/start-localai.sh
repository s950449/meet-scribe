#!/usr/bin/env bash
# Foreground LocalAI (Intel SYCL image) serving Gemma 4 12B and Breeze-ASR-25.
# Published on 127.0.0.1:8080 only. Stop with Ctrl+C.
# Uploads are whole 16 kHz mono tracks (~115 MB per hour); LocalAI's default
# 15 MB limit rejects anything over ~7.8 minutes, so it is raised to 2048 MB.
#
# Reuses the backends volume of the existing local-ai-intel container, which
# holds intel-sycl-f16-llama-cpp and intel-sycl-f16-whisper. Install a missing
# backend with:
#   docker run --rm -e LOCALAI_FORCE_META_BACKEND_CAPABILITY=intel \
#     -v local-ai-intel-backends:/backends --entrypoint /local-ai \
#     localai/localai:latest-gpu-intel backends install localai@whisper
set -euo pipefail
here="$(dirname "$(readlink -f "$0")")"
image="${MEETSCRIBE_LOCALAI_IMAGE:-localai/localai:latest-gpu-intel}"
data_dir="${MEETSCRIBE_DATA_DIR:-$HOME/.local/share/meetscribe}"
models="$data_dir/models"
tty=()
[[ -t 0 && -t 1 ]] && tty=(-it)
# oneDNN matmul aborts with "could not create a memory object" on Arc B390
# (LocalAI v4.11.0 SYCL whisper backend), so it is off unless overridden.
export GGML_SYCL_ENABLE_DNN="${GGML_SYCL_ENABLE_DNN:-0}"
# Forward GGML_SYCL_* tuning variables from the host.
sycl_env=()
while IFS= read -r name; do sycl_env+=(-e "$name"); done < <(compgen -e | grep '^GGML_SYCL_' || true)
exec docker run --rm "${tty[@]}" --name meetscribe-localai \
  --device /dev/dri \
  -p 127.0.0.1:8080:8080 \
  -e LOCALAI_FORCE_META_BACKEND_CAPABILITY=intel \
  -e ZES_ENABLE_SYSMAN=1 \
  -e LOCALAI_SINGLE_ACTIVE_BACKEND="${LOCALAI_SINGLE_ACTIVE_BACKEND:-false}" \
  -e LOCALAI_DISABLE_WEBUI=true \
  -e LOCALAI_UPLOAD_LIMIT="${LOCALAI_UPLOAD_LIMIT:-2048}" \
  "${sycl_env[@]}" \
  -v local-ai-intel-backends:/backends \
  -v "$models:/models:ro" \
  -e LOCALAI_MODELS_CONFIG_FILE=/configuration/meetscribe-models.yaml \
  -v "$here/localai-models.yaml:/configuration/meetscribe-models.yaml:ro" \
  "$image" "$@"
