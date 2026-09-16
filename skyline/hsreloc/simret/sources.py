"""The three interchangeable skyline sources for simulator sessions (Part B/C; PROT-SKY-001 §5).

One seam, three producers, nothing downstream branching on which::

    oracle:sim_exact              the simulator's own sky mask — ground truth, the ceiling
    automatic:segformer_b0_ade20k SegFormer-B0/ADE20K silver curves — the EXP-SKY-005 pipeline
    automatic:poc_robust_dp       the frozen classical extractor — the EXP-SKY-004 baseline

All three return a ``SkylineCurve`` and a ``describe()`` block; the matcher, the record and the
evaluator are untouched. A run names exactly one and the record carries its provenance, so an exact
number can never be reported in the same figure as a silver one.

What is reused rather than re-declared
--------------------------------------
The full-width policy, the sentinel-curve mechanism and both automatic provenance strings are
**imported** from ``hsreloc.placeret.sources`` (the frozen ``EXP-SKY-006`` bench) rather than copied,
so the simulator experiment and the ECL experiment cannot drift apart in the one place where a silent
difference would make their numbers incomparable. The silver conversion itself
(``hsreloc.extraction.silver.convert``) and the DP curves' on-disk format are likewise untouched.

The only thing that changes for the simulator is the **key**: ECL keys label maps by
``<condition>/<scale>/<group_id>.npz`` because its store holds one session per condition; a simulator
store holds one session per (scene, trajectory, condition), so the key is
``<session_id>/<scale>/<observation_id>.npz``. That is a layout difference, not a method difference,
and no preprocessing, revision, class index or conversion parameter moves with it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from hsreloc.extraction.auto_source import AUTO_SUBDIR, AutomaticCurveSource
from hsreloc.extraction.silver.convert import CONVERT_VERSION, STATUS_OK, silver_curve
from hsreloc.observation import read_session
from hsreloc.placeret.sources import (DP_METHOD, DP_PROVENANCE, FULL_WIDTH_POLICY, POLICY_VERSION,
                                      REFUSAL_RAISE, REFUSAL_SENTINEL, SEG_METHOD, SEG_PROVENANCE,
                                      SourceError, sentinel_curve)
from hsreloc.retrieval.skyline_curve import CurveError, SkylineCurve
from hsreloc.retrieval.sources import OracleCurveSource
from hsreloc.simret.simgt import SIM_EXACT_PROVENANCE

SOURCE_KEYS = ("sim_exact", "segformer", "dp")
PROVENANCE_BY_KEY = {"sim_exact": SIM_EXACT_PROVENANCE, "segformer": SEG_PROVENANCE, "dp": DP_PROVENANCE}


def session_map(store_root: Path | str, session_ids=None) -> dict:
    """``observation_id -> session_id`` over the ingested simulator sessions under ``store_root``."""
    root = Path(store_root)
    if session_ids is None:
        session_ids = sorted(p.name for p in root.iterdir()
                             if (p / "observations.csv").exists())
    mapping = {}
    for sid in session_ids:
        _, observations = read_session(root / sid)
        for obs in observations:
            mapping[obs.observation_id] = sid
    return mapping


class _Refusable:
    """Shared refusal semantics: raise for reference enrolment, sentinel for query matching.

    A reference whose curve a source cannot produce is a **database hole** (listed in the reference
    manifest, never dropped from the grid); a query is a declared sentinel flat curve, which the
    unchanged matcher's own degenerate-profile rule records as ``EXTRACTION_FAILURE``. Neither branch
    invents a boundary and neither needs a matcher change.
    """

    provenance = ""

    def __init__(self, image_width_px: int, image_height_px: int, refusal_mode: str = REFUSAL_RAISE):
        self.w, self.h = int(image_width_px), int(image_height_px)
        if refusal_mode not in (REFUSAL_RAISE, REFUSAL_SENTINEL):
            raise SourceError(f"refusal_mode must be {REFUSAL_RAISE!r} or {REFUSAL_SENTINEL!r}")
        self.refusal_mode = refusal_mode
        self._cache: dict = {}

    def get(self, observation_id: str) -> SkylineCurve:
        cached = self._cache.get(observation_id)
        if cached is not None:
            return cached
        try:
            curve = self._get_uncached(observation_id)
        except CurveError as exc:
            if self.refusal_mode == REFUSAL_SENTINEL:
                curve = sentinel_curve(observation_id, self.w, self.h, self.provenance, str(exc)[:120])
            else:
                raise
        self._cache[observation_id] = curve
        return curve

    def _get_uncached(self, observation_id: str) -> SkylineCurve:   # pragma: no cover - abstract
        raise NotImplementedError


class SimExactSource(_Refusable):
    """The simulator's own sky mask, via the ``skylines_oracle/`` curves the adapter wrote.

    Ground truth, not a silver label. A missing curve means the mask had a column with no sky above
    it, so no full-width curve exists — the adapter refused to fabricate one, and that refusal
    arrives here as the ordinary seam refusal.
    """

    kind = "sim_observation_store_exact"
    provenance = SIM_EXACT_PROVENANCE

    def __init__(self, store_root, sessions: dict, image_width_px: int, image_height_px: int,
                 refusal_mode: str = REFUSAL_RAISE):
        super().__init__(image_width_px, image_height_px, refusal_mode)
        self.store_root = Path(store_root)
        self._inner = OracleCurveSource(self.store_root, sessions, image_height_px, image_width_px,
                                        expect_provenance=SIM_EXACT_PROVENANCE)

    def _get_uncached(self, observation_id: str) -> SkylineCurve:
        return self._inner.get(observation_id)

    def describe(self) -> dict:
        return {"kind": self.kind, "root": str(self.store_root), "provenance": self.provenance,
                "gt": True, "refusal_mode": self.refusal_mode,
                "image_width_px": self.w, "image_height_px": self.h,
                "notes": "simulator sky-mask ground truth (top-connected sky, topmost run per column); "
                         "a curve exists only where every column has sky above it"}


class SessionDPSource(_Refusable):
    """The frozen ``poc_robust_dp`` curves, from ``<session>/skylines_auto/``. Untuned, digest-guarded."""

    kind = "sim_observation_store_auto"
    provenance = DP_PROVENANCE

    def __init__(self, store_root, sessions: dict, image_width_px: int, image_height_px: int,
                 frozen_digest_file=None, extraction_manifest=None, refusal_mode: str = REFUSAL_RAISE):
        super().__init__(image_width_px, image_height_px, refusal_mode)
        self.store_root = Path(store_root)
        self.method_digest = None
        if extraction_manifest is not None:
            m = json.loads(Path(extraction_manifest).read_text(encoding="utf-8"))
            if m.get("method_id") != DP_METHOD:
                raise SourceError(f"extraction manifest method {m.get('method_id')!r} != {DP_METHOD!r}")
            self.method_digest = m.get("method_config_digest")
            if frozen_digest_file is not None:
                frozen = Path(frozen_digest_file).read_text(encoding="utf-8").strip().split()[0]
                if frozen != self.method_digest:
                    raise SourceError(f"frozen method digest {frozen[:16]}… != extraction manifest "
                                      f"{str(self.method_digest)[:16]}… — this is not the frozen extractor")
        self._inner = AutomaticCurveSource(self.store_root, sessions, image_height_px, image_width_px,
                                           DP_METHOD)

    def _get_uncached(self, observation_id: str) -> SkylineCurve:
        return self._inner.get(observation_id)

    def describe(self) -> dict:
        return {"kind": self.kind, "root": str(self.store_root), "provenance": self.provenance,
                "method_id": DP_METHOD, "method_config_digest": self.method_digest,
                "refusal_mode": self.refusal_mode, "curve_subdir": AUTO_SUBDIR,
                "image_width_px": self.w, "image_height_px": self.h}


class SessionSilverSource(_Refusable):
    """SegFormer-B0/ADE20K silver curves for simulator sessions.

    Identical to the ECL source in every methodological respect — the pinned revision, the NVlabs
    preprocessing, sky class 2, the unchanged ``silver_curve`` conversion and the frozen full-width
    policy — and different only in where a label map is keyed:
    ``<mask_root>/<session_id>/s<scale>/<observation_id>.npz``.

    The model is never tuned here, and its output remains a **silver label**: on simulator imagery it
    is additionally *out of its training domain* (ADE20K is photographic), which is why
    ``PROT-SKY-001`` §5 requires a DEV inspection of a declared handful of rendered frames before any
    freeze names it primary. This class does not and cannot perform that inspection; it records the
    identifiers that make it auditable.
    """

    kind = "sim_label_map_silver_full_width"
    provenance = SEG_PROVENANCE

    def __init__(self, mask_root, sessions: dict, image_width_px: int, image_height_px: int,
                 scale_short_side: int = 512, closing_px: int = 0, min_valid_frac: float = 0.5,
                 expected_model_revision=None, refusal_mode: str = REFUSAL_RAISE):
        super().__init__(image_width_px, image_height_px, refusal_mode)
        self.mask_root = Path(mask_root)
        self.sessions = dict(sessions)
        self.scale_tag = f"s{int(scale_short_side)}"
        self.closing_px = int(closing_px)
        self.min_valid_frac = float(min_valid_frac)
        manifest_path = self.mask_root / "inference_manifest.json"
        if not manifest_path.exists():
            raise SourceError(f"no inference_manifest.json under {self.mask_root} — run "
                              f"scripts/silver_infer.py in the isolated venv first")
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.model_revision = m.get("model", {}).get("revision")
        self.label_map_digest = m.get("output_digest")
        self.keying = m.get("keying")
        if expected_model_revision is not None and self.model_revision != expected_model_revision:
            raise SourceError(f"label maps were produced by revision {self.model_revision!r}, "
                              f"expected {expected_model_revision!r}")
        if m.get("model", {}).get("sky_class_index", 2) != 2:
            raise SourceError("label maps do not declare ADE20K class 2 as sky")

    def label_map_path(self, observation_id: str) -> Path:
        try:
            session = self.sessions[observation_id]
        except KeyError:
            raise CurveError(f"{observation_id}: no session known for this observation") from None
        return self.mask_root / session / self.scale_tag / f"{observation_id}.npz"

    def _get_uncached(self, observation_id: str) -> SkylineCurve:
        path = self.label_map_path(observation_id)
        if not path.exists():
            raise CurveError(f"{observation_id}: no label map at {path}")
        with np.load(path) as z:
            label = z["label"]
        if label.shape != (self.h, self.w):
            raise CurveError(f"{observation_id}: label map shape {label.shape} != ({self.h}, {self.w})")
        res = silver_curve(label, closing_px=self.closing_px, min_valid_frac=self.min_valid_frac)
        if res["status"] != STATUS_OK:
            raise CurveError(f"{observation_id}: silver status {res['status']} "
                             f"(valid fraction {res['valid_frac']:.2f}) — no curve by the full-width policy")
        rows = np.where(res["valid"], res["rows"], FULL_WIDTH_POLICY["top_edge_row"])
        from hsreloc.retrieval.skyline_curve import make_curve
        return make_curve(observation_id, np.clip(rows, 0.0, self.h - 1.0), self.w, self.h,
                          SEG_PROVENANCE, str(path))

    def describe(self) -> dict:
        return {"kind": self.kind, "root": str(self.mask_root), "provenance": self.provenance,
                "method_id": SEG_METHOD, "model_revision": self.model_revision,
                "label_map_digest": self.label_map_digest, "keying": self.keying,
                "scale_tag": self.scale_tag, "convert_version": CONVERT_VERSION,
                "closing_px": self.closing_px, "min_valid_frac": self.min_valid_frac,
                "full_width_policy": FULL_WIDTH_POLICY, "policy_version": POLICY_VERSION,
                "refusal_mode": self.refusal_mode, "silver_label": True,
                "image_width_px": self.w, "image_height_px": self.h,
                "notes": "SILVER LABEL, never ground truth; ADE20K is photographic and this is rendered "
                         "imagery — an out-of-domain application that PROT-SKY-001 §5 requires be "
                         "inspected on declared frames before any freeze names it primary"}


def build_source(key: str, spec: dict, sessions: dict, image_width_px: int, image_height_px: int,
                 store_root=None, config_dir=None, refusal_mode: str = REFUSAL_RAISE):
    """Construct one source from its run-config block."""
    def resolve(v):
        p = Path(v)
        return p if p.is_absolute() else ((Path(config_dir) / p).resolve() if config_dir else p)

    if key == "sim_exact":
        return SimExactSource(store_root or resolve(spec["store_root"]), sessions,
                              image_width_px, image_height_px, refusal_mode=refusal_mode)
    if key == "dp":
        return SessionDPSource(store_root or resolve(spec["store_root"]), sessions,
                               image_width_px, image_height_px,
                               frozen_digest_file=resolve(spec["frozen_digest_file"]) if spec.get("frozen_digest_file") else None,
                               extraction_manifest=resolve(spec["extraction_manifest"]) if spec.get("extraction_manifest") else None,
                               refusal_mode=refusal_mode)
    if key == "segformer":
        return SessionSilverSource(resolve(spec["mask_root"]), sessions, image_width_px, image_height_px,
                                   scale_short_side=spec.get("scale_short_side", 512),
                                   closing_px=spec.get("closing_px", 0),
                                   min_valid_frac=spec.get("min_valid_frac", 0.5),
                                   expected_model_revision=spec.get("expected_model_revision"),
                                   refusal_mode=refusal_mode)
    raise SourceError(f"unknown source key {key!r}; expected one of {SOURCE_KEYS}")


# --------------------------------------------------------------------------------------------------
# DP extraction over a simulator session
# --------------------------------------------------------------------------------------------------

def extract_dp_session(store_root: Path, session_id: str, method_id: str = DP_METHOD,
                       params: dict | None = None, run_root: Path | None = None) -> dict:
    """Run the frozen extractor over one ingested session and write ``skylines_auto/``.

    Mirrors ``hsreloc.extraction.auto_source.extract_session`` but resolves each frame from the
    session's own ``observations.csv`` rather than globbing ``images/``, so a session ingested with
    ``image_mode: "reference"`` (frames left in the run directory) extracts identically to a copied
    one. The method, its parameters and the storage rule are unchanged.
    """
    import cv2

    from hsreloc.extraction.auto_source import AutoStoreError, write_auto_curve
    from hsreloc.extraction.methods import get_method

    store_root = Path(store_root)
    session_dir = store_root / session_id
    _, observations = read_session(session_dir)
    method = get_method(method_id)
    method.configure(dict(params or {}))
    stored, skipped = [], {}
    for obs in observations:
        candidates = [session_dir / obs.image_path]
        source_rel = obs.extra.get("sim_image_source")
        if source_rel and run_root is not None:
            candidates.append(Path(run_root) / source_rel)
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise AutoStoreError(
                f"{obs.observation_id}: no frame found (tried {[str(c) for c in candidates]}). A session "
                f"ingested with image_mode='reference' needs run_root to locate its frames.")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise AutoStoreError(f"{obs.observation_id}: unreadable image {path}")
        output = method.extract(image)
        output.validate()
        written, reason = write_auto_curve(store_root, session_id, obs.observation_id, output)
        if written is None:
            skipped[obs.observation_id] = reason
        else:
            stored.append(obs.observation_id)
    return {"session": session_id, "method": method_id, "n_images": len(observations),
            "stored": stored, "skipped": skipped}


def write_silver_frame_list(store_root: Path, session_ids: list, out_path: Path) -> Path:
    """The frame list ``scripts/silver_infer.py`` consumes in session mode (isolated venv).

    Written from this side so the inference script never imports ``hsreloc``: it reads a plain CSV of
    ``session_id,observation_id,image_path`` and writes ``<session>/s<scale>/<observation_id>.npz``.
    """
    root = Path(store_root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in session_ids:
        session_dir = root / sid
        _, observations = read_session(session_dir)
        for obs in observations:
            rows.append({"session_id": sid, "observation_id": obs.observation_id,
                         "image_path": str((session_dir / obs.image_path).resolve()),
                         "width_px": obs.image_width_px, "height_px": obs.image_height_px})
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["session_id", "observation_id", "image_path",
                                          "width_px", "height_px"])
        w.writeheader()
        w.writerows(rows)
    return out_path
