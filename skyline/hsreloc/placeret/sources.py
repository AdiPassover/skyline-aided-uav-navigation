"""The two frozen skyline sources behind the unchanged ``CurveSource`` seam (research R4, R5).

Observation ids carry the condition (``<group_id>__<condition>``) because the ECL store keeps one
session per condition and the seam maps one id to one session.

``ConditionedDPSource`` — the classical curves the EXP-SKY-004 frozen run already wrote to
``observations_ecl/ecl_<condition>/skylines_auto``; guarded by the method digest; provenance
``automatic:poc_robust_dp``. Nothing is extracted here.

``SilverFullWidthSource`` — the pinned SegFormer-B0 label maps converted by the **unchanged**
``hsreloc.extraction.silver.convert.silver_curve`` rule, then made full-width by the declared,
pre-registered policy: image status other than ``ok`` → no curve (``CurveError`` → the matcher's
``EXTRACTION_FAILURE``); invalid columns of an ``ok`` image → the top image edge (row 0). Provenance
``automatic:segformer_b0_ade20k``. The model output is a silver label, never ground truth.

Neither source may be tuned on retrieval; both ``describe()`` their frozen identifiers so a record
names exactly the curves it used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hsreloc.extraction.auto_source import AutomaticCurveSource
from hsreloc.extraction.silver.convert import CONVERT_VERSION, STATUS_OK, silver_curve
from hsreloc.placeret.split import parse_observation_id
from hsreloc.retrieval.skyline_curve import CurveError, SkylineCurve, make_curve

DP_METHOD = "poc_robust_dp"
DP_PROVENANCE = f"automatic:{DP_METHOD}"
SEG_METHOD = "segformer_b0_ade20k"
SEG_PROVENANCE = f"automatic:{SEG_METHOD}"
FULL_WIDTH_POLICY = {"status_not_ok": "extraction_failure", "invalid_columns": "top_edge", "top_edge_row": 0.0,
                     "refusal_mechanism": "no curve for references (database hole); for queries a sentinel flat "
                                          "curve at the top edge, which the unchanged matcher's degenerate-profile "
                                          "rule turns into EXTRACTION_FAILURE with no position"}
POLICY_VERSION = "1.1.0"
REFUSAL_RAISE = "raise"          # reference enrolment: a missing curve is a database hole
REFUSAL_SENTINEL = "sentinel"    # query matching: the unchanged preflight needs a curve per query


def sentinel_curve(observation_id: str, width: int, height: int, provenance: str, reason: str) -> SkylineCurve:
    """A flat curve at the top edge: mean-removed it is all zeros, so ``profile.is_degenerate`` fires
    and the matcher records EXTRACTION_FAILURE — the contracted outcome for a refused query — without
    any matcher module being touched. Never enrolled as a reference."""
    return make_curve(observation_id, np.zeros(width, dtype=np.float64), width, height, provenance,
                      f"sentinel:{reason}")


class SourceError(Exception):
    """A source cannot be constructed as the pre-registration requires."""


def _session(condition: str) -> str:
    return f"ecl_{condition}"


class ConditionedDPSource:
    kind = "observation_store_auto_conditioned"

    def __init__(self, store_root, image_height_px: int, image_width_px: int,
                 frozen_digest_file=None, extraction_manifest=None, refusal_mode: str = REFUSAL_RAISE) -> None:
        self.store_root = Path(store_root)
        self.refusal_mode = refusal_mode
        self.h, self.w = int(image_height_px), int(image_width_px)
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
                                      f"{str(self.method_digest)[:16]}… — the store is not the frozen run")
        self._inner: dict = {}

    def _source_for(self, session: str, gid: str) -> AutomaticCurveSource:
        return AutomaticCurveSource(self.store_root, {gid: session}, self.h, self.w, DP_METHOD)

    def get(self, observation_id: str) -> SkylineCurve:
        cached = self._inner.get(observation_id)
        if cached is not None:
            return cached
        curve = self._get_uncached(observation_id)
        self._inner[observation_id] = curve
        return curve

    def _get_uncached(self, observation_id: str) -> SkylineCurve:
        gid, cond = parse_observation_id(observation_id)
        try:
            inner = self._source_for(_session(cond), gid).get(gid)
        except CurveError as exc:
            if self.refusal_mode == REFUSAL_SENTINEL:
                return sentinel_curve(observation_id, self.w, self.h, DP_PROVENANCE, "extractor_refused")
            raise CurveError(f"{observation_id}: {exc}") from exc
        return make_curve(observation_id, inner.row_per_col, self.w, self.h, DP_PROVENANCE, inner.source_ref)

    def describe(self) -> dict:
        return {"kind": self.kind, "root": str(self.store_root), "provenance": DP_PROVENANCE,
                "method_id": DP_METHOD, "method_config_digest": self.method_digest,
                "refusal_mode": self.refusal_mode,
                "image_width_px": self.w, "image_height_px": self.h}


class SilverFullWidthSource:
    kind = "label_map_silver_full_width"

    def __init__(self, mask_root, image_height_px: int, image_width_px: int, scale_short_side: int = 512,
                 closing_px: int = 0, min_valid_frac: float = 0.5, expected_model_revision=None,
                 refusal_mode: str = REFUSAL_RAISE) -> None:
        self.mask_root = Path(mask_root)
        self.refusal_mode = refusal_mode
        self.h, self.w = int(image_height_px), int(image_width_px)
        self.scale_tag = f"s{int(scale_short_side)}"
        self.closing_px = int(closing_px)
        self.min_valid_frac = float(min_valid_frac)
        manifest_path = self.mask_root / "inference_manifest.json"
        if not manifest_path.exists():
            raise SourceError(f"no inference_manifest.json under {self.mask_root}")
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.model_revision = m.get("model", {}).get("revision")
        self.label_map_digest = m.get("output_digest")
        if expected_model_revision is not None and self.model_revision != expected_model_revision:
            raise SourceError(f"label maps were produced by revision {self.model_revision!r}, "
                              f"expected {expected_model_revision!r}")
        if m.get("model", {}).get("sky_class_index", 2) != 2:
            raise SourceError("label maps do not declare ADE20K class 2 as sky")
        self._cache: dict = {}

    def label_map_path(self, observation_id: str) -> Path:
        gid, cond = parse_observation_id(observation_id)
        return self.mask_root / cond / self.scale_tag / f"{gid}.npz"

    def get(self, observation_id: str) -> SkylineCurve:
        cached = self._cache.get(observation_id)
        if cached is not None:
            return cached
        curve = self._get_uncached(observation_id)
        self._cache[observation_id] = curve
        return curve

    def _get_uncached(self, observation_id: str) -> SkylineCurve:
        path = self.label_map_path(observation_id)
        if not path.exists():
            if self.refusal_mode == REFUSAL_SENTINEL:
                return sentinel_curve(observation_id, self.w, self.h, SEG_PROVENANCE, "no_label_map")
            raise CurveError(f"{observation_id}: no label map at {path}")
        with np.load(path) as z:
            label = z["label"]
        if label.shape != (self.h, self.w):
            raise CurveError(f"{observation_id}: label map shape {label.shape} != ({self.h}, {self.w})")
        res = silver_curve(label, closing_px=self.closing_px, min_valid_frac=self.min_valid_frac)
        if res["status"] != STATUS_OK:
            if self.refusal_mode == REFUSAL_SENTINEL:
                return sentinel_curve(observation_id, self.w, self.h, SEG_PROVENANCE, res["status"])
            raise CurveError(f"{observation_id}: silver status {res['status']} "
                             f"(valid fraction {res['valid_frac']:.2f}) - no curve by the full-width policy")
        rows = np.where(res["valid"], res["rows"], FULL_WIDTH_POLICY["top_edge_row"])
        rows = np.clip(rows, 0.0, self.h - 1.0)
        return make_curve(observation_id, rows, self.w, self.h, SEG_PROVENANCE, str(path))

    def describe(self) -> dict:
        return {"kind": self.kind, "root": str(self.mask_root), "provenance": SEG_PROVENANCE,
                "method_id": SEG_METHOD, "model_revision": self.model_revision,
                "label_map_digest": self.label_map_digest, "scale_tag": self.scale_tag,
                "convert_version": CONVERT_VERSION, "closing_px": self.closing_px,
                "min_valid_frac": self.min_valid_frac, "full_width_policy": FULL_WIDTH_POLICY,
                "policy_version": POLICY_VERSION, "refusal_mode": self.refusal_mode,
                "image_width_px": self.w, "image_height_px": self.h}


def build_source(key: str, spec: dict, benchmark_root, image_height_px: int, image_width_px: int,
                 config_dir=None, refusal_mode: str = REFUSAL_RAISE):
    """Construct a source from its run-config block (contracts/placeret-run-config.md)."""
    root = Path(benchmark_root)

    def resolve(v):
        p = Path(v)
        return p if p.is_absolute() else (Path(config_dir) / p).resolve() if config_dir else p
    if key == "dp":
        return ConditionedDPSource(root / spec["store_root"], image_height_px, image_width_px,
                                   frozen_digest_file=resolve(spec["frozen_digest_file"]) if spec.get("frozen_digest_file") else None,
                                   extraction_manifest=resolve(spec["extraction_manifest"]) if spec.get("extraction_manifest") else None,
                                   refusal_mode=refusal_mode)
    if key == "segformer":
        return SilverFullWidthSource(root / spec["mask_root"], image_height_px, image_width_px,
                                     scale_short_side=spec.get("scale_short_side", 512),
                                     closing_px=spec.get("closing_px", 0),
                                     min_valid_frac=spec.get("min_valid_frac", 0.5),
                                     expected_model_revision=spec.get("expected_model_revision"),
                                     refusal_mode=refusal_mode)
    raise SourceError(f"unknown source key {key!r}")


PROVENANCE_BY_KEY = {"dp": DP_PROVENANCE, "segformer": SEG_PROVENANCE}
