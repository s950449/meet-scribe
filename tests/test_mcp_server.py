import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from meetscribe.errors import MeetscribeError
from meetscribe.mcp_server import TranscriptionJobs, create_server


def test_stdio_protocol_and_error(tmp_path):
    async def exercise():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "meetscribe", "mcp", "--output-dir", str(tmp_path / "jobs")],
        )
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as client,
        ):
            await client.initialize()
            listed = await client.list_tools()
            assert {tool.name for tool in listed.tools} == {
                "start_transcription",
                "transcription_status",
                "read_transcript",
                "save_agent_revision",
                "cancel_transcription",
            }
            result = await client.call_tool(
                "start_transcription", {"source": str(tmp_path / "missing.wav")}
            )
            assert result.isError
            result = await client.call_tool("read_transcript", {"job_id": "../../etc/passwd"})
            assert result.isError

    asyncio.run(exercise())


def test_worker_failure_and_cleanup(tmp_path):
    from test_pipeline import _write_silence

    source = tmp_path / "audio.wav"
    _write_silence(source)
    config = tmp_path / "bad.toml"
    config.write_text('[asr]\nbackend = "nonexistent-test-backend"\n')
    jobs = TranscriptionJobs(tmp_path / "jobs", config)
    try:
        job_id = jobs.start(str(source))["job_id"]
        jobs.jobs[job_id].process.wait(timeout=20)
        status = jobs.status(job_id)
        assert status["status"] == "failed"
        assert "nonexistent-test-backend" in status["error"]
        with pytest.raises(MeetscribeError, match="not completed"):
            jobs.source(job_id)
        with pytest.raises(MeetscribeError, match="Unknown"):
            jobs.status("../")
    finally:
        jobs.close()


def test_tools_read_pages_and_save_revision(tmp_path):
    # Use a completed fake ASR job, but exercise actual MCP tool dispatch/validation.
    import subprocess
    from unittest.mock import patch

    from meetscribe.config import AppConfig
    from meetscribe.mcp_server import Job
    from meetscribe.pipeline import run_meeting
    from test_pipeline import FakeAsr, _write_silence

    source = tmp_path / "audio.wav"
    _write_silence(source)
    config = AppConfig()
    config.llm.polish_mode = "agent"
    run_meeting(config, mic=source, system=source, output=tmp_path / "transcript.md", asr=FakeAsr())
    worker = subprocess.Popen([sys.executable, "-c", "pass"])
    worker.wait(timeout=10)
    jobs = TranscriptionJobs(tmp_path)
    jobs.jobs["test"] = Job(worker, tmp_path)
    with patch("meetscribe.mcp_server.TranscriptionJobs", return_value=jobs):
        server = create_server(tmp_path)

    async def exercise():
        _, payload = await server.call_tool("read_transcript", {"job_id": "test", "limit": 1})
        assert payload["total"] == 2
        assert payload["next_offset"] == 1
        _, second = await server.call_tool("read_transcript", {"job_id": "test", "offset": 1})
        assert second["utterances"][0]["id"] == "u000001"
        assert second["source_revision"] == payload["source_revision"]
        assert second["next_offset"] is None
        _, saved = await server.call_tool(
            "save_agent_revision",
            {
                "job_id": "test",
                "source_revision": payload["source_revision"],
                "edits": {"u000000": "我們改 API。"},
                "summary": {"overview": "討論 API。", "decisions": ["修改 API"]},
            },
        )
        assert Path(saved["markdown_path"]).is_file()
        assert "討論 API。" in Path(saved["markdown_path"]).read_text()
        with pytest.raises(Exception, match="offset"):
            await server.call_tool("read_transcript", {"job_id": "test", "offset": -1})

    asyncio.run(exercise())


def test_busy_worker_cancel_and_close(tmp_path):
    import subprocess

    from meetscribe.mcp_server import Job

    jobs = TranscriptionJobs(tmp_path)
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    jobs.jobs["busy"] = Job(worker, tmp_path)
    (tmp_path / "worker.log").write_text("")
    try:
        assert jobs.status("busy")["status"] == "running"
        with pytest.raises(MeetscribeError, match="already running"):
            jobs.start("unused.wav")
        jobs.cancel("busy")
        assert worker.poll() is not None
    finally:
        jobs.close()
