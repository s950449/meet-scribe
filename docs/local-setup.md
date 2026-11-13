# 本機執行（Linux / Intel Arc B390，LocalAI SYCL）

## 環境位置

- Python 3.12 虛擬環境：專案下的 `.venv/`
- 設定：`~/.config/meetscribe/config.toml`
- 模型：`~/.local/share/meetscribe/models/`
  - `ggml-breeze-asr-25.bin`（Breeze-ASR-25 轉成的 GGML f16，3.1 GB）
  - `gemma-4-12b-it-UD-Q4_K_XL.gguf`（7.37 GB）
- 測試輸入：`~/.local/share/meetscribe/smoke-test/`

ASR 和 LLM 都由 LocalAI 的 Intel image（`localai/localai:latest-gpu-intel`，
v4.11.0，oneAPI 2025.3）在 SYCL / Level Zero 上服務：

| 模型 ID | LocalAI backend | 用途 |
| --- | --- | --- |
| `breeze-asr-25` | `intel-sycl-f16-whisper` | 逐字稿（`/v1/audio/transcriptions`） |
| `gemma-4-12b-it` | `intel-sycl-f16-llama-cpp` | 潤飾、摘要（`/v1/chat/completions`） |

meetscribe 用 `openai-api` ASR backend 和 `llama-server` LLM provider
連到同一個 `127.0.0.1:8080`。處理音訊只走本機 loopback，不需要 API key。
說話者依麥克風／系統雙軌區分，不是聲紋辨識。

## 啟動

第一個終端保持 LocalAI 運行（只綁定 `127.0.0.1:8080`，Ctrl+C 停止）：

```bash
bash scripts/start-localai.sh
```

模型設定在 `scripts/localai-models.yaml`。容器名稱 `meetscribe-localai`，
沿用既有 `local-ai-intel` 容器的 `local-ai-intel-backends` volume，
不影響 `local-ai-intel` 本身（8081）。第一次請求會載入模型：
Breeze 約 5 秒，Gemma 約 1.5 分鐘。

另一個終端：

```bash
uv run --no-sync meetscribe run /path/to/meeting.m4a --title "會議" -o ~/Meetings/meeting.md
```

輸出包含 Markdown、JSON sidecar 以及可續跑的 `.md.work/`。
LLM 階段失敗時，修正後用原命令加 `--resume`，不用重做 ASR。

## 已知問題與設定

- SYCL whisper 的 oneDNN matmul 在 B390 上會 abort（`could not create a memory
  object`），所以 `start-localai.sh` 預設 `GGML_SYCL_ENABLE_DNN=0`。主機上設定的
  `GGML_SYCL_*` 環境變數都會傳進容器，可用來覆寫。
- LocalAI 不採用請求中的 `chat_template_kwargs`，Gemma 的 thinking 由
  `localai-models.yaml` 的 `reasoning.disable: true` 關閉。
- LocalAI 的轉錄只回傳段落時間，沒有逐字時間戳。
- 缺少 backend 時（例如重建 volume 後）：

  ```bash
  docker run --rm -e LOCALAI_FORCE_META_BACKEND_CAPABILITY=intel \
    -v local-ai-intel-backends:/backends --entrypoint /local-ai \
    localai/localai:latest-gpu-intel backends install localai@whisper
  ```

## 錄音（需手動開始）

```bash
uv run --no-sync meetscribe devices
uv run --no-sync meetscribe record --duration 60 -o ~/Meetings/take
uv run --no-sync meetscribe run ~/Meetings/take -o ~/Meetings/take.md
```

可用 `--mic-device` / `--system-device` 指定裝置。請先確認參與者同意錄音。

## 無 LLM 逐字稿

```bash
uv run --no-sync meetscribe transcribe /path/to/meeting.m4a --no-polish -o ~/Meetings/raw.md
```

## 重新轉換 Breeze-ASR-25

```bash
git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git
git clone --depth 1 https://github.com/openai/whisper.git
uv run --no-project --python 3.12 --with huggingface_hub --with torch --with transformers \
  python scripts/convert_ggml.py --convert --whisper-cpp whisper.cpp --whisper-repo whisper \
  --output ~/.local/share/meetscribe/models/ggml-breeze-asr-25.bin
```

完成後可刪除兩個 clone 和 `~/.cache/huggingface/hub/models--MediaTek-Research--Breeze-ASR-25`
（腳本只下載權重和 tokenizer，不含訓練狀態）。

## 重建 Python 環境

```bash
uv sync --python 3.12 --extra faster-whisper --extra capture-linux --extra audio
```

一般使用 `uv run --no-sync`，避免不帶 extras 的同步移除錄音依賴。
