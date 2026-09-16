"""The producer side of the seam: ``skylines_auto/`` writer + ``AutomaticCurveSource``.

Contracted in `DEC-SKY-001` and `skyline-curve-source.md` before any extractor existed: an
automatic source reads ``<root>/<session_id>/skylines_auto/<observation_id>.csv`` in the identical
``col,row`` format, differing from the oracle source in one path segment and one provenance
string. The frozen 006 observation record already carries ``auto_skyline_path`` — this module
fills exactly that slot and nothing else. **No matcher module changes**; conformance is proven by
the same contract battery the oracle source passes plus a schema-level synthetic round-trip
(spec FR-016/017, explicitly *not* a retrieval-quality measurement — FR-009).

Failure semantics at the seam: an extraction failure, or a curve with any invalid column, produces
**no file** — the seam's validation demands contiguous full-width coverage, and a partially valid
curve stored as if complete would fabricate boundaries (spec FR-013). A missing file surfaces
downstream as ``CurveError`` → the matcher's per-query ``EXTRACTION_FAILURE`` outcome, which is
the correct, contracted propagation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from hsreloc.extraction.methods import get_method
from hsreloc.extraction.methods.base import STATUS_OK, ExtractionOutput
from hsreloc.retrieval.skyline_curve import CurveError, SkylineCurve, make_curve

AUTO_SUBDIR = "skylines_auto"


class AutoStoreError(Exception):
    """The automatic-curve store cannot be written as the contract requires."""


def write_auto_curve(store_root: Path, session_id: str, observation_id: str,
                     output: ExtractionOutput):
    """Store one extractor output; returns the path, or ``None`` (with reason) when not storable."""
    if output.status != STATUS_OK:
        return None, output.reason
    curve = output.curve
    if not curve.valid_mask.all():
        n_bad = curve.width_px - curve.n_valid
        return None, (f"curve has {n_bad} invalid columns — the seam requires contiguous "
                      f"full-width coverage, partial curves are not stored")
    path = Path(store_root) / session_id / AUTO_SUBDIR / f"{observation_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("col,row\n")
        for c in range(curve.width_px):
            f.write(f"{c},{float(curve.rows[c])}\n")
    return path, None


class AutomaticCurveSource:
    """Automatic curves from an observation store — the third producer behind the seam.

    Mirrors ``OracleCurveSource`` (same layout, same validation, same digesting via
    ``make_curve``) with ``skylines_auto`` for ``skylines_oracle`` and
    ``provenance="automatic:<method>"``.
    """

    kind = "observation_store_auto"

    def __init__(self, root, sessions: dict, image_height_px: int, image_width_px: int,
                 method_id: str) -> None:
        self.root = Path(root)
        self.sessions = dict(sessions)
        self.image_height_px = int(image_height_px)
        self.image_width_px = int(image_width_px)
        self.method_id = method_id
        self.provenance = f"automatic:{method_id}"

    def curve_path(self, observation_id: str) -> Path:
        try:
            session_id = self.sessions[observation_id]
        except KeyError:
            raise CurveError(
                f"{observation_id}: no session known for this observation -- it is not in the "
                f"query/reference set this source was built from"
            ) from None
        return self.root / session_id / AUTO_SUBDIR / f"{observation_id}.csv"

    def get(self, observation_id: str) -> SkylineCurve:
        path = self.curve_path(observation_id)
        if not path.exists():
            raise CurveError(
                f"{observation_id}: no automatic curve at {path} (the extractor refused or "
                f"failed on this observation)")
        rows: dict = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows[int(row["col"])] = float(row["row"])
        expected = list(range(self.image_width_px))
        if sorted(rows) != expected:
            raise CurveError(
                f"{observation_id}: stored curve at {path} does not cover columns "
                f"0..{self.image_width_px - 1} contiguously (has {len(rows)} columns)")
        values = np.array([rows[c] for c in expected], dtype=np.float64)
        return make_curve(observation_id, values, self.image_width_px, self.image_height_px,
                          self.provenance, str(path))

    def describe(self) -> dict:
        return {"kind": self.kind, "root": str(self.root), "n_known": len(self.sessions),
                "provenance": self.provenance, "method_id": self.method_id,
                "image_width_px": self.image_width_px, "image_height_px": self.image_height_px}


def extract_session(store_root: Path, session_id: str, method_id: str, params: dict) -> dict:
    """Run one method over ``<root>/<session>/images/*.png`` and store the storable curves."""
    store_root = Path(store_root)
    images = sorted((store_root / session_id / "images").glob("*.png"))
    if not images:
        raise AutoStoreError(f"no images under {store_root / session_id / 'images'}")
    method = get_method(method_id)
    method.configure(params)
    stored, skipped = [], {}
    for img_path in images:
        obs_id = img_path.stem
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise AutoStoreError(f"{obs_id}: unreadable image {img_path}")
        output = method.extract(image)
        output.validate()
        path, reason = write_auto_curve(store_root, session_id, obs_id, output)
        if path is None:
            skipped[obs_id] = reason
        else:
            stored.append(obs_id)
    return {"session": session_id, "method": method_id, "n_images": len(images),
            "stored": stored, "skipped": skipped}


def run_extract(config: dict, config_path: Path) -> dict:
    summary = extract_session(Path(config["store_root"]), config["session"],
                              config["method"]["id"], config["method"].get("params", {}))
    print(f"[ext] extract: {summary['method']} on {summary['session']}: "
          f"{len(summary['stored'])}/{summary['n_images']} stored, "
          f"{len(summary['skipped'])} refused/unstorable")
    return summary


def run_roundtrip(config: dict, config_path: Path) -> dict:
    """T1 schema-level round-trip: synthetic corpus → extract → seam → matcher → record checks.

    Explicitly NOT a retrieval-quality measurement (spec FR-009): the constructed answers are never
    read, and no outcome is compared against them. Checked instead: the run completes, the record
    parses, provenance and digests are what the contract requires.
    """
    import importlib.util
    import tempfile

    from hsreloc.retrieval import runconfig as rc
    from hsreloc.retrieval.run import execute

    # The Stage-1 corpus builder lives in the repo's own test infrastructure. It is loaded by
    # explicit file path because the bare name ``tests`` resolves to ``evaluation/tests`` (the
    # hsreloc bootstrap puts ``evaluation/`` first on sys.path).
    synth_path = Path(__file__).resolve().parents[2] / "tests" / "synth_corpus.py"
    spec = importlib.util.spec_from_file_location("_ext_synth_corpus", synth_path)
    synth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(synth)
    build_corpus = synth.build_corpus

    method_block = config["method"]
    method_id = method_block["id"]
    work = Path(config.get("work_dir") or tempfile.mkdtemp(prefix="ext_roundtrip_"))
    work.mkdir(parents=True, exist_ok=True)
    corpus = build_corpus(work)

    obs_root = work / "observations"
    query_session = "synth_query"
    summary = extract_session(obs_root, query_session, method_id,
                              method_block.get("params", {}))

    run_cfg_path = work / "roundtrip-run.json"
    run_cfg_path.write_text(json.dumps({
        "schema_version": "1.1.0",
        "run_id": f"ext-roundtrip-{method_id}",
        "inputs": {
            "reference_set": str(work / "skyline_refdb" / "refset-synth-sky-v1"),
            "query_set": str(work / "datasets" / "skyquery-synth-sky-v1"),
            "query_curves": {"kind": "observation_store", "root": str(obs_root),
                             "expect_provenance": f"automatic:{method_id}"},
        },
        "geodetic_origin": {"lat_deg": 63.0, "lon_deg": 10.0, "alt_m": 100.0},
        "profile": {"n_samples": 256, "normalize_mean": True, "detrend": False,
                    "units": "image_fraction"},
        "match": {"baseline": "ncc", "recall_k": 5, "lag_search": False},
        "acceptance": {"confidence_k": 5.0, "reject_threshold": 0.5, "viable_sigma": 3.0,
                       "ambiguity_margin_sigma": 1.0, "degenerate_profile_std": 1e-06},
        "output": {"root": str(work / "skyline_runs")},
        "evidence": {"tier": "T1",
                     "caveat": "Synthetic seam round-trip; schema-level only; no retrieval-quality "
                               "claim of any kind follows from this run."},
        "environment": {"is_target_hardware": False},
    }, indent=2), encoding="utf-8")

    run_config = rc.load_config(run_cfg_path)
    # Sessions and image dims come from the matcher's own ground-truth-free query-set reader
    # (research R3: the query set determines the mapping; nothing extra is configured).
    from hsreloc.retrieval.queryset import load_matcher_query_set
    qs = load_matcher_query_set(run_config.query_set)
    source = AutomaticCurveSource(obs_root, qs.session_map(),
                                  image_height_px=qs.image_height_px,
                                  image_width_px=qs.image_width_px, method_id=method_id)

    # Check 0 — the anti-mixing rule holds at the seam (found by this round-trip, 2026-08-24):
    # the corpus's sidecar declares its curves 'sim_exact', so consuming automatic curves against
    # it MUST be refused by preflight. That refusal is contract behaviour, asserted here.
    from hsreloc.retrieval.runconfig import ConfigError
    try:
        execute(run_config, curve_source=source, overwrite=True)
        raise AutoStoreError(
            "preflight accepted automatic curves against a query set declaring oracle "
            "provenance — the anti-mixing rule did not fire")
    except ConfigError:
        anti_mixing_refused = True

    # A dataset revision built *for* automatic curves declares them in its sidecar; model that
    # in the scratch copy (T1 fixture, never a tracked artifact) and run the chain for real.
    # The sidecar column stores the provenance *suffix* only ("manual", "sim_exact", or here the
    # method id) — documented in the matcher's own cross-check, which also records that the
    # column cannot distinguish oracle: from automatic: prefixes (a contract limitation noted
    # there, not resolved here).
    sidecar = Path(run_config.query_set) / "skyline_queries.csv"
    lines = sidecar.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    prov_idx = header.index("oracle_provenance")
    patched = [lines[0]]
    for line in lines[1:]:
        cells = line.split(",")
        cells[prov_idx] = method_id
        patched.append(",".join(cells))
    sidecar.write_text("\n".join(patched) + "\n", encoding="utf-8")

    record = execute(run_config, curve_source=source, overwrite=True)

    record_dir = Path(record["record_dir"]) if "record_dir" in record else \
        work / "skyline_runs" / f"ext-roundtrip-{method_id}"
    manifest = json.loads((record_dir / "manifest.json").read_text(encoding="utf-8"))
    checks = {
        "anti_mixing_refused": anti_mixing_refused,
        "provenance_is_automatic":
            manifest["relocalizer_config"]["curve_source"]["provenance"] == f"automatic:{method_id}",
        "query_curves_digest_present": bool(
            manifest.get("query_curves_digest")
            or manifest["relocalizer_config"].get("query_curves_digest")),
        "code_revision_present": "code_revision" in manifest,
        "queries_csv_parses": (record_dir / "queries.csv").exists(),
        "n_extracted_stored": len(summary["stored"]),
        "n_refused": len(summary["skipped"]),
    }
    failed = [k for k, v in checks.items() if v is False]
    if failed:
        raise AutoStoreError(f"round-trip schema checks failed: {failed}")
    print(f"[ext] roundtrip {method_id}: OK — provenance automatic, digest + revision present, "
          f"{checks['n_extracted_stored']} curves stored, {checks['n_refused']} refused "
          f"(refusals surface as EXTRACTION_FAILURE outcomes)")
    return {"checks": checks, "record_dir": str(record_dir), "work_dir": str(work)}


