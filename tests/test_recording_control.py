from datetime import datetime
from pathlib import Path

import numpy as np

from meetscribe.audio.control import RecordingControl
from meetscribe.audio.sync import SyncedWavSession


def test_pause_clips_both_tracks_and_delayed_buffers(tmp_path: Path) -> None:
    now = [10.0]
    control = RecordingControl(clock=lambda: now[0])
    session = SyncedWavSession(tmp_path, sample_rate=10, control=control)
    session.write("mic", 10.0, np.ones(10), 10)
    now[0] = 11.0
    control.toggle_pause()
    now[0] = 13.0
    control.toggle_pause()
    # Delayed buffers crossing both pause boundaries use the same retained timebase.
    session.write("mic", 11.0, np.ones(30), 10)
    session.write("system", 10.0, np.ones(40), 10)
    result = session.finish(datetime.now().astimezone())
    assert {track.duration_s for track in result.tracks} == {2.0}
    assert all(stats.silence_frames == 0 for stats in session.stats().values())


def test_duration_excludes_pause_and_stop_is_explicit() -> None:
    now = [2.0]
    control = RecordingControl(clock=lambda: now[0])
    now[0] = 5.0
    control.toggle_pause()
    now[0] = 20.0
    assert control.elapsed() == 3.0
    control.toggle_pause()
    now[0] = 22.0
    assert control.elapsed() == 5.0
    control.stop()
    assert control.stopped


def test_monitor_warns_once_until_audio_recovers(tmp_path: Path) -> None:
    now = [10.0]
    control = RecordingControl(clock=lambda: now[0])
    session = SyncedWavSession(tmp_path, sample_rate=10, control=control)
    session.write("mic", 10.0, np.zeros(10), 10)
    events = session.monitor(["mic", "system"], elapsed=6.0)
    assert {e.track_id for e in events if e.status == "warning"} == {"mic", "system"}
    assert not [e for e in session.monitor(["mic", "system"], elapsed=7.0) if e.status == "warning"]
    now[0] = 17.0
    session.write("mic", 17.0, np.ones(10), 10)
    session.monitor(["mic", "system"], elapsed=7.0)
    events = session.monitor(["mic", "system"], elapsed=14.0)
    assert any(e.track_id == "mic" and e.status == "warning" for e in events)
    session.finish(datetime.now().astimezone())


def test_wait_duration_counts_only_active_time(monkeypatch) -> None:
    import meetscribe.audio.record as record

    now = [0.0]
    control = RecordingControl(clock=lambda: now[0])
    ticks = [1.0, 2.0, 12.0, 13.0]

    class Wake:
        def wait(self, seconds: float) -> None:
            now[0] = ticks.pop(0)
            if now[0] in (2.0, 12.0):
                control.toggle_pause()

    monkeypatch.setattr(record.threading, "Event", Wake)
    events = []
    record._wait(3.0, control=control, on_event=events.append)
    assert now[0] == 13.0
    assert control.elapsed() == 3.0
    assert any(event.status == "paused" for event in events)


def test_keyboard_does_not_read_redirected_input(monkeypatch) -> None:
    import meetscribe.audio.record as record

    class Redirected:
        def isatty(self) -> bool:
            return False

        def read(self, size: int) -> str:
            raise AssertionError("must not consume piped input")

    monkeypatch.setattr(record.sys, "stdin", Redirected())
    with record._keyboard(True) as read_key:
        assert read_key() == ""


def test_paused_samples_are_absent_from_saved_audio(tmp_path: Path) -> None:
    from meetscribe.audio.io import read_audio

    now = [0.0]
    control = RecordingControl(clock=lambda: now[0])
    session = SyncedWavSession(tmp_path, sample_rate=10, control=control)
    now[0] = 1.0
    control.toggle_pause()
    now[0] = 2.0
    control.toggle_pause()
    samples = np.concatenate([np.full(10, 0.2), np.full(10, 0.9), np.full(10, 0.4)])
    session.write("mic", 0.0, samples, 10)
    session.finish(datetime.now().astimezone())
    saved, _ = read_audio(tmp_path / "mic.wav")
    np.testing.assert_allclose(
        saved, np.concatenate([np.full(10, 0.2), np.full(10, 0.4)]), atol=0.0001
    )


def test_wait_disables_keyboard_for_json_sink(monkeypatch) -> None:
    from contextlib import contextmanager

    import meetscribe.audio.record as record

    control = RecordingControl()
    enabled_values = []

    @contextmanager
    def keyboard(enabled):
        enabled_values.append(enabled)
        control.stop()
        yield lambda: ""

    class Sink:
        json_events = True

        def emit(self, event):
            pass

    monkeypatch.setattr(record, "_keyboard", keyboard)
    record._wait(None, control=control, on_event=Sink().emit)
    assert enabled_values == [False]


def test_posix_keyboard_reads_burst_and_restores_terminal(monkeypatch) -> None:
    import os

    import pytest

    if os.name == "nt":
        pytest.skip("POSIX terminal only")
    import pty
    import termios

    import meetscribe.audio.record as record

    master, slave = pty.openpty()
    with os.fdopen(slave, "r") as terminal:
        monkeypatch.setattr(record.sys, "stdin", terminal)
        before = termios.tcgetattr(terminal)
        try:
            with record._keyboard(True) as read_key:
                os.write(master, b"pq")
                assert read_key() == "p"
                assert read_key() == "q"
            after = termios.tcgetattr(terminal)
            # Darwin may set the transient PENDIN bit when changing canonical mode.
            after[3] &= ~getattr(termios, "PENDIN", 0)
            before[3] &= ~getattr(termios, "PENDIN", 0)
            assert after == before
        finally:
            os.close(master)


def test_empty_recording_still_writes_session_manifest(tmp_path: Path) -> None:
    directory = tmp_path / "empty"
    session = SyncedWavSession(directory)
    result = session.finish(datetime.now().astimezone())
    assert result.tracks == ()
    assert (directory / "session.json").is_file()


def test_duration_starts_after_device_initialization(tmp_path: Path, monkeypatch) -> None:
    import meetscribe.audio.record as record
    from meetscribe.audio.devices import AudioDevice

    now = [10.0]
    control = RecordingControl(clock=lambda: now[0])

    class Stream:
        def start(self) -> None:
            now[0] += 4.0

        def close(self) -> None:
            pass

    def opener(device, track, session, sample_rate):
        now[0] += 3.0
        return Stream()

    def wait(duration, **kwargs):
        assert duration == 2.0
        assert kwargs["control"] is control
        assert control.elapsed() == 0.0
        now[0] += 2.0
        assert control.elapsed() == 2.0

    monkeypatch.setattr(record, "_wait", wait)
    record.run_synced_recording(
        tmp_path,
        devices=[("mic", AudioDevice("m", "fake", "mic"))],
        opener=opener,
        sample_rate=10,
        duration_s=2.0,
        chunk_seconds=0.0,
        on_chunk=None,
        on_event=None,
        control=control,
    )


def test_start_timer_keeps_injected_pause_and_host_timeline() -> None:
    now = [1.0]
    control = RecordingControl(clock=lambda: now[0])
    now[0] = 2.0
    control.toggle_pause()
    now[0] = 5.0
    control.start_timer()
    assert control.paused
    assert control.elapsed() == 0.0
    now[0] = 7.0
    control.toggle_pause()
    now[0] = 8.0
    assert control.elapsed() == 1.0
    assert control.retained(1.0, 8.0) == [(1.0, 2.0, 1.0), (7.0, 8.0, 2.0)]


def test_monitor_missing_audio_after_long_pause(tmp_path: Path) -> None:
    now = [10.0]
    control = RecordingControl(clock=lambda: now[0])
    session = SyncedWavSession(tmp_path, sample_rate=10, control=control)
    session.write("mic", 10.0, np.ones(10), 10)
    now[0] = 11.0
    control.toggle_pause()
    now[0] = 111.0
    control.toggle_pause()
    session.write("mic", 111.0, np.ones(10), 10)
    now[0] = 117.0
    events = session.monitor(["mic"], elapsed=control.elapsed())
    assert any(e.status == "warning" and dict(e.data)["reason"] == "missing" for e in events)
    assert next(dict(e.data)["peak"] for e in events if e.status == "level") == 0.0
    session.finish(datetime.now().astimezone())


def test_windows_keyboard_ignores_extended_keys(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    import meetscribe.audio.record as record

    keys = iter(["\xe0", "P", "\x00", "Q", "p", "q"])
    console = SimpleNamespace(kbhit=lambda: True, getwch=lambda: next(keys))
    monkeypatch.setitem(sys.modules, "msvcrt", console)
    monkeypatch.setattr(record, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(record.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    with record._keyboard(True) as read_key:
        assert read_key() == ""
        assert read_key() == ""
        assert read_key() == "p"
        assert read_key() == "q"
