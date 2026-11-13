import io
import wave
from types import SimpleNamespace

import numpy as np

from meetscribe.audio.macos import HELPER_RATE, _SystemHelper
from meetscribe.audio.sync import SyncedWavSession


def test_pipe_delivery_jitter_does_not_cut_or_insert_audio(tmp_path, monkeypatch):
    samples = (np.sin(np.arange(14400) * 0.03) * 12000).astype("<i2")
    process = SimpleNamespace(stdout=io.BytesIO(samples.tobytes()))
    monkeypatch.setattr("meetscribe.audio.macos.subprocess.Popen", lambda *a, **k: process)
    times = iter([100.1, 100.25, 100.26])
    monkeypatch.setattr("meetscribe.audio.macos.time.monotonic", lambda: next(times))
    session = SyncedWavSession(output_dir=tmp_path, sample_rate=HELPER_RATE)
    helper = _SystemHelper("helper", session, "system", HELPER_RATE)
    helper._read()
    stats = session.stats()["system"]
    assert stats.silence_frames == 0
    assert stats.dropped_frames == 0
    assert stats.frames == len(samples)
    # Flush the writer before checking every sample survived in order.
    session.finish("test")
    with wave.open(str(tmp_path / "system.wav")) as wav:
        actual = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    np.testing.assert_allclose(actual, samples, atol=1)
