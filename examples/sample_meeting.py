"""Build the checked-in sample note from fake utterances.

The numbers and sentences are hand-written. No ASR model is loaded.
Run from the repo root after installing meetscribe:

    PYTHONPATH=src python examples/sample_meeting.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from meetscribe.render.markdown import render_markdown
from meetscribe.types import ActionItem, AttributedUtterance, MeetingDocument, MeetingSummary


def build_document() -> MeetingDocument:
    utterances = (
        AttributedUtterance(
            speaker_id="Me",
            text="我們今天確認 checkout service 從 REST 改到 gRPC 的時程。timeout 還是 800ms。",
            polished_text=(
                "我們今天確認 checkout service 從 REST 改到 gRPC 的時程，timeout 維持 800ms。"
            ),
            start=0.4,
            end=8.6,
            track_id="mic",
        ),
        AttributedUtterance(
            speaker_id="SPEAKER_00",
            text="sandbox 已經跑過 contract test，idempotency key 跟 production 一致。",
            polished_text=(
                "sandbox 已經跑過 contract test，idempotency key 的行為與 production 一致。"
            ),
            start=9.2,
            end=17.8,
            track_id="system",
        ),
        AttributedUtterance(
            speaker_id="SPEAKER_01",
            text="rollback 還是走 feature flag 嗎？不要再直接改 database migration。",
            polished_text="rollback 還是走 feature flag 嗎？我不希望再直接改 database migration。",
            start=18.4,
            end=26.0,
            track_id="system",
        ),
        AttributedUtterance(
            speaker_id="Me",
            text="對，flag 名稱維持 payments.grpc_checkout，下週三前我補 runbook。",
            polished_text="對，flag 名稱維持 payments.grpc_checkout，下週三前我補 runbook。",
            start=25.2,
            end=33.5,
            track_id="mic",
            overlap=True,
        ),
        AttributedUtterance(
            speaker_id="SPEAKER_00",
            text="那林可安補 runbook，我週五前把 Grafana dashboard 的 p99 告警加上。",
            polished_text="那就由林可安補 runbook，我週五前把 Grafana dashboard 的 p99 告警加上。",
            start=34.8,
            end=43.0,
            track_id="system",
        ),
    )
    summary = MeetingSummary(
        overview=(
            "會議確認 checkout service 由 REST 遷移到 gRPC，"
            "以 feature flag 控制上線，並約定 runbook 與告警的負責人。"
        ),
        key_points=(
            "sandbox 的 contract test 已通過，idempotency key 與 production 一致。",
            "timeout 維持 800ms。",
            "rollback 使用 feature flag payments.grpc_checkout，不直接改 database migration。",
        ),
        decisions=("以 feature flag 上線 gRPC checkout，不做不可逆的 database migration。",),
        action_items=(
            ActionItem(task="補上 gRPC checkout 的 runbook", owner="林可安", due="下週三"),
            ActionItem(task="為 Grafana dashboard 加上 p99 告警", owner="陳怡君", due="週五"),
        ),
    )
    return MeetingDocument(
        title="支付 API 遷移週會",
        started_at=datetime.fromisoformat("2026-04-14T15:00:00+08:00"),
        summary=summary,
        utterances=utterances,
        speaker_map={
            "Me": "林可安",
            "SPEAKER_00": "陳怡君",
            "SPEAKER_01": "林浩然",
        },
    )


def main() -> None:
    destination = Path(__file__).with_name("sample_meeting.md")
    destination.write_text(render_markdown(build_document()), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
