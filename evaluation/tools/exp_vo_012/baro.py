"""`EXP-VO-012`: the takeoff-relative height channel, as a **model** with declared parameters.

No pressure is read anywhere in this repository and no device is reproduced. This module turns a
known true altitude series into what a takeoff-relative height channel would have reported, through
the error mechanisms `LIT-VO-006` §4 names, with every parameter swept rather than assumed. The
sweep endpoints are bracketed by a representative MEMS datasheet (`LIT-VO-006` §6) — **not** by any
DJI specification, which is unmeasured.

Two things this module is careful about, because both are places where a plausible-looking shortcut
would quietly answer a different question:

**Causality.** The default resampling policy is a **causal zero-order hold**: the value at a frame
time is the most recent sample at or before it. `linear` interpolation uses the *next* sample too
and is therefore **offline only**; it exists to measure what causality costs, and every caller has to
name it explicitly. `SampledHeight.causal` records which was used so a plot cannot mislabel itself.

**What is and is not a barometer error.** `h₀` error and terrain are *not* modelled here. They are
architectural (`DEC-VO-007` D2, `LIT-VO-006` §4 rows 8–9) and charging them to the sensor would hide
that. `h₀` enters in `metric_readout.py`; terrain enters in the renderer.

The channel reports **relative** altitude — zero at the reference frame by construction, exactly as
a flight controller zeroes at takeoff. That is what makes the absolute-accuracy term cancel
(`LIT-VO-006` §1) and it is why `bias` is not a parameter here: a constant offset is removed by the
zeroing, and what survives is drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class BaroSpec:
    """One point in the pre-registered sweep. All defaults are the ideal channel."""

    #: Output rate in Hz. `None` means "every frame" (i.e. the imagery's own rate).
    rate_hz: float | None = None
    #: Transport/filtering latency in seconds. Positive means the reported value is old.
    latency_s: float = 0.0
    #: Zero-mean white noise standard deviation, metres.
    noise_m: float = 0.0
    #: TOTAL drift accrued over the sequence, metres, linear in time. Signed.
    drift_m: float = 0.0
    #: Output quantisation step in metres. 0 disables.
    quantise_m: float = 0.0
    #: Dropout windows as (start_s, duration_s). The channel reports nothing inside them.
    dropouts: tuple[tuple[float, float], ...] = ()
    #: How long a held value may be used after the last valid sample. `DEC-VO-007` D6.
    tau_stale_s: float = 2.0
    #: 'zoh' (causal, default) or 'linear' (OFFLINE ONLY -- uses future samples).
    policy: str = "zoh"
    #: RNG seed for the noise draw.
    seed: int = 0

    def label(self) -> str:
        bits = []
        if self.rate_hz is not None:
            bits.append(f"{self.rate_hz:g}Hz")
        if self.latency_s:
            bits.append(f"lat{self.latency_s:g}s")
        if self.noise_m:
            bits.append(f"n{self.noise_m:g}m")
        if self.drift_m:
            bits.append(f"d{self.drift_m:+g}m")
        if self.quantise_m:
            bits.append(f"q{self.quantise_m:g}m")
        if self.dropouts:
            bits.append("drop" + ",".join(f"{s:g}+{d:g}" for s, d in self.dropouts))
        if self.policy != "zoh":
            bits.append(self.policy)
        return "ideal" if not bits else "·".join(bits)


@dataclass
class SampledHeight:
    """The channel's output at frame times, plus the flags a consumer needs."""

    h_baro: np.ndarray          #: relative altitude at each frame time, metres (NaN where invalid)
    stale: np.ndarray           #: bool — value is a hold, not a fresh sample
    valid: np.ndarray           #: bool — inside `tau_stale_s` of a real sample
    causal: bool                #: False if any output used a future sample
    spec: BaroSpec = field(repr=False, default_factory=BaroSpec)

    @property
    def n_stale(self) -> int:
        return int(self.stale.sum())

    @property
    def n_invalid(self) -> int:
        return int((~self.valid).sum())


def _sample_times(t_frames: np.ndarray, rate_hz: float | None) -> np.ndarray:
    """When the channel produces a value. Anchored at the first frame, so the reference frame always
    has a genuine sample and the relative datum is exact by construction."""
    if rate_hz is None:
        return t_frames.copy()
    t0, t1 = float(t_frames[0]), float(t_frames[-1])
    n = int(np.floor((t1 - t0) * rate_hz)) + 1
    return t0 + np.arange(n) / rate_hz


def sample(t_frames: np.ndarray, up_true_m: np.ndarray, spec: BaroSpec) -> SampledHeight:
    """Produce the height channel's output at each frame time.

    `up_true_m` is the TRUE altitude above the takeoff datum. The channel reports it relative to its
    own first sample, with the mechanisms of `spec` applied, then resampled to the frame times.
    """
    t_frames = np.asarray(t_frames, float)
    up_true = np.asarray(up_true_m, float)
    ts = _sample_times(t_frames, spec.rate_hz)

    # True altitude at the channel's own sample times, relative to the first sample.
    truth_at_ts = np.interp(ts, t_frames, up_true)
    rel = truth_at_ts - truth_at_ts[0]

    # --- mechanisms, in the order they physically occur -----------------------------------------
    # Drift: linear in time, zero at the reference frame (it is a RELATIVE channel, so a constant
    # offset has already cancelled -- LIT-VO-006 section 1).
    span = float(ts[-1] - ts[0]) if ts.size > 1 else 1.0
    rel = rel + spec.drift_m * (ts - ts[0]) / span

    if spec.noise_m > 0:
        rel = rel + np.random.default_rng(spec.seed).normal(0.0, spec.noise_m, ts.shape)
    if spec.quantise_m > 0:
        rel = np.round(rel / spec.quantise_m) * spec.quantise_m

    # Dropouts remove SAMPLES; the hold/stale logic below then decides what a consumer sees.
    alive = np.ones(ts.shape, bool)
    for start, dur in spec.dropouts:
        alive &= ~((ts >= t_frames[0] + start) & (ts < t_frames[0] + start + dur))
    if not alive.any():
        raise ValueError("every sample dropped")

    # Latency: the sample taken at `t` only becomes available at `t + latency`.
    t_avail = ts + spec.latency_s

    return _resample(t_frames, t_avail[alive], rel[alive], spec)


def _resample(t_frames: np.ndarray, t_avail: np.ndarray, values: np.ndarray,
              spec: BaroSpec) -> SampledHeight:
    """Frame-time values from the available samples, under the declared policy.

    ZOH is causal by construction: `searchsorted(..., 'right') - 1` selects the last sample whose
    availability time is at or before the frame time, and nothing else is consulted. Frames before
    the first available sample have no causal value at all; rather than inventing one, the reference
    frame's own value is used and the frame is marked `stale` — which is the honest description of
    what a real system does at startup, and it is visible in the output.
    """
    n = t_frames.size
    idx = np.searchsorted(t_avail, t_frames, side="right") - 1
    before_start = idx < 0
    idx = np.clip(idx, 0, t_avail.size - 1)

    age = t_frames - t_avail[idx]
    age[before_start] = 0.0

    if spec.policy == "zoh":
        h = values[idx]
        causal = True
        # A frame is `stale` when its value is older than one nominal sample interval.
        interval = (1.0 / spec.rate_hz) if spec.rate_hz else float(np.median(np.diff(t_frames)))
        stale = (age > interval * 1.5) | before_start
    elif spec.policy == "linear":
        # OFFLINE ONLY: np.interp consults the NEXT sample as well as the previous one.
        h = np.interp(t_frames, t_avail, values)
        causal = False
        gap = np.zeros(n)
        nxt = np.clip(np.searchsorted(t_avail, t_frames, side="left"), 0, t_avail.size - 1)
        gap = t_avail[nxt] - t_avail[idx]
        interval = (1.0 / spec.rate_hz) if spec.rate_hz else float(np.median(np.diff(t_frames)))
        stale = gap > interval * 1.5
    else:
        raise ValueError(f"unknown policy {spec.policy!r}")

    valid = age <= spec.tau_stale_s
    valid[before_start] = True          # the reference frame's own datum is exact by construction
    h = np.where(valid, h, np.nan)
    return SampledHeight(h_baro=h, stale=stale, valid=valid, causal=causal, spec=spec)


def ideal(t_frames: np.ndarray, up_true_m: np.ndarray) -> SampledHeight:
    """The perfect channel — every frame, no noise, no drift, no lag. The R1/R2 reference."""
    return sample(t_frames, up_true_m, BaroSpec())


def with_(spec: BaroSpec, **kw) -> BaroSpec:
    """`replace`, re-exported so sweep code reads as one line."""
    return replace(spec, **kw)
