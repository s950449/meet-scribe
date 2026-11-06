# 支付 API 遷移週會

- **時間**：2026-04-14 15:00
- **參與者**：林可安、陳怡君、林浩然

## 摘要

會議確認 checkout service 由 REST 遷移到 gRPC，以 feature flag 控制上線，並約定 runbook 與告警的負責人。

### 重點

- sandbox 的 contract test 已通過，idempotency key 與 production 一致。
- timeout 維持 800ms。
- rollback 使用 feature flag payments.grpc_checkout，不直接改 database migration。

### 決議

- 以 feature flag 上線 gRPC checkout，不做不可逆的 database migration。

### 待辦

- 補上 gRPC checkout 的 runbook（林可安） — 下週三
- 為 Grafana dashboard 加上 p99 告警（陳怡君） — 週五

## 逐字稿

### 林可安（00:00:00–00:00:08）

我們今天確認 checkout service 從 REST 改到 gRPC 的時程，timeout 維持 800ms。

### 陳怡君（00:00:09–00:00:17）

sandbox 已經跑過 contract test，idempotency key 的行為與 production 一致。

### 林浩然（00:00:18–00:00:26）

rollback 還是走 feature flag 嗎？我不希望再直接改 database migration。

### 林可安（00:00:25–00:00:33） · 重疊

對，flag 名稱維持 payments.grpc_checkout，下週三前我補 runbook。

### 陳怡君（00:00:34–00:00:43）

那就由林可安補 runbook，我週五前把 Grafana dashboard 的 p99 告警加上。

