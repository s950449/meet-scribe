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

## 潤飾和摘要改用雲端 API

本機 Gemma 4 12B 潤稿一場 10 分鐘的會議要 20 分鐘以上（見下方效能紀錄），所以潤飾和摘要
改用 OpenAI 相容的雲端 API。**逐字稿文字會送到該供應商**；音訊和 ASR 仍只在本機。

```toml
[llm]
provider = "openai-compatible"
base_url = "https://<provider>/v1"
model = "<model id>"
api_key_env = "LLM_API_KEY"   # 金鑰只放環境變數
context_tokens = 16384        # 每批約 6.5k token 輸入、相近長度輸出
timeout = 300
```

使用雲端 provider 時，只會送出 config 有明確設定的 `temperature` / `top_p`；
`top_k`、`min_p`、`chat_template_kwargs` 只送給本機 server。遇到 429 或 5xx 會重試 2 次。
模型一律依 config 指定，不會從 `/v1/models` 自動挑。這時 LocalAI 只需要服務 Breeze；
Gemma 不會被載入，模型檔可以保留，當作離線時的備援。

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

## 效能紀錄（2026-11-13，Arc B390，Core Ultra X7 358H，30 GB）

用 espeak-ng 合成的 10 分鐘雙軌中文會議（mic 和 system 兩軌，共 175 句）測試：

| 階段 | 時間 | 備註 |
| --- | --- | --- |
| ASR mic（含 Breeze 冷載入） | 70 s | GPU 計算引擎約 70–100% 忙碌，容器 CPU 約 1.1 核 |
| ASR system | 52 s | 約 11.5 倍即時 |
| Gemma 冷載入 | 約 80 s | SYCL 關閉 mmap，等於整個 7.4 GB 複製一次 |
| 潤稿（本機 Gemma） | 超過 600 s，逾時 | 第一批輸入 6487 token，處理約 460 tok/s；生成約 10 tok/s 起跳，持續負載後掉到 4.5–5.3 tok/s |

- 兩個模型都載入後，主機的 MemAvailable 少了約 13.5 GB；只用 Breeze 時約 4 GB。
- 持續生成 15–30 秒後，`throttle/reasons` 顯示 `pl1`，GPU 從 2500 MHz 被壓到 1000 MHz。
  同時 llama.cpp 用了約 11 顆 CPU 核心空轉等待，跟 GPU 搶同一份功耗額度。
  之後如果要改回本機 LLM，可以先在模型 YAML 設定 `threads: 2` 再量一次。

## 已知問題與設定

- **睡眠喚醒後 GPU 會失效**：筆電睡眠再喚醒後，第一個 SYCL 工作會卡住，接著 kernel 重置
  GPU 引擎，LocalAI 回 `UR_RESULT_ERROR_DEVICE_LOST`。喚醒後請先執行
  `docker restart meetscribe-localai`。
- **上傳大小**：LocalAI 預設上傳上限是 15 MB，大約只能處理 7.8 分鐘的 16 kHz 單聲道音軌。
  `start-localai.sh` 已改為 2048 MB（`LOCALAI_UPLOAD_LIMIT`）。

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

## 推送到 Slack

`meetscribe run` 跑完會依 `[publish]` 設定推送：摘要、重點、決議、待辦當作訊息，
完整的 `.md` 當作附件。送不出去時存到 `~/.local/state/meetscribe/outbox/`（0600），
之後用 `meetscribe publish --flush` 補送；部分對象失敗時只補送沒送成功的對象。

### 直接用 Slack API（`kind = "slack"`）

1. 到 api.slack.com/apps，選擇 *Create New App → From an app manifest*，
   貼上 `deploy/slack/manifest.yaml`，安裝到 workspace。
2. 把 *Bot User OAuth Token*（`xoxb-…`）存在 repo 以外的地方：

   ```bash
   install -m 600 /dev/null ~/.config/meetscribe/slack-token
   $EDITOR ~/.config/meetscribe/slack-token
   echo 'export SLACK_BOT_TOKEN="$(cat ~/.config/meetscribe/slack-token)"' >> ~/.bashrc
   ```

3. 在要發送的頻道執行 `/invite @meetscribe`。
4. 在 `~/.config/meetscribe/config.toml` 設定發送對象，只能用 ID：

   ```toml
   [publish]
   kind = "slack"
   auto = true
   slack_targets = ["C0123ABCD", "U0456EFGH"]   # 頻道 ID；使用者 ID 會發私訊
   ```

   頻道 ID 在頻道詳細資料的 About 分頁最下方；使用者 ID 在個人檔案的 ⋮ → Copy member ID。
5. 用既有的會議記錄測試：`uv run --no-sync meetscribe publish ~/Meetings/meeting.md`

團隊 workspace 改用同事建立的 Slack App 時，只要換掉 token 和 `slack_targets`。
那個 App 需要相同的 scopes。要切換 workspace，可以用另一份設定檔加 `-c`，
並用 `slack_token_env` 指定另一個環境變數。

### 透過 n8n（`kind = "webhook"`）

拿不到 bot token 時，改用團隊的 n8n.cloud 或自架的 n8n，設定見 `deploy/n8n/README.md`。

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
