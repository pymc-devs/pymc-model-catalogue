"""Peak-RSS tracker. Stubbed RSS: a real bytearray need not raise RSS at all."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import asv_bench._runner as runner  # noqa: E402
from asv_bench._runner import PeakRSS  # noqa: E402


def _stub_rss(monkeypatch, values):
    """Feed the sampler a fixed sequence, repeating the last value forever."""
    it, last = iter(values), [values[0]]

    def fake():
        last[0] = next(it, last[0])
        return last[0]

    monkeypatch.setattr(runner, "_rss_bytes", fake)


def test_reports_the_max_not_the_last_reading(monkeypatch):
    _stub_rss(monkeypatch, [100, 500, 900, 200, 150])
    with PeakRSS(interval=0.001) as tracker:
        while tracker.peak < 900:
            pass
    assert tracker.peak == 900


def test_final_read_catches_what_the_sampler_missed(monkeypatch):
    # long interval: only __exit__'s final read can see the 7000
    _stub_rss(monkeypatch, [100, 100, 7000])
    with PeakRSS(interval=1000.0) as tracker:
        pass
    assert tracker.peak == 7000


def test_ru_maxrss_rise_is_taken_as_exact(monkeypatch):
    """A rise in the kernel high-water mark can only have happened in-window."""
    _stub_rss(monkeypatch, [1000, 1000])
    ru = iter([5000, 9000])
    monkeypatch.setattr(runner, "_ru_maxrss_bytes", lambda: next(ru, 9000))
    with PeakRSS(interval=1000.0) as tracker:
        pass
    assert tracker.peak == 9000 and tracker.delta == 8000


def test_flat_ru_maxrss_does_not_inflate_the_window(monkeypatch):
    _stub_rss(monkeypatch, [100, 400])
    monkeypatch.setattr(runner, "_ru_maxrss_bytes", lambda: 9_000_000)
    with PeakRSS(interval=1000.0) as tracker:
        pass
    assert tracker.peak == 400
