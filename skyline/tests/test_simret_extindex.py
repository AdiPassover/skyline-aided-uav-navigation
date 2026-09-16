"""Extended-index tests (research R2): the split is by level and nothing else, counts match the
committed inventory, and the validation refuses broken structures by name."""

import json
from pathlib import Path

import pytest

from hsreloc.simret import extindex

REPO = Path(__file__).resolve().parents[2]
INDEX_DIR = REPO / "evaluations" / "sim-ext-index"
CONFIG = REPO / "skyline" / "configs" / "sim-final.json"


def _cfg():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _row(**kw):
    base = {"level": "L", "split": "dev", "session_id": "S", "observation_id": "S__sky_000001",
            "kind": "anchor_capture", "anchor_id": "a00", "time_of_day": "DAY", "clouds": "CLEAR"}
    base.update(kw)
    return base


class TestValidation:
    CFG = {"levels": {"L": {"split": "dev", "task_key": "l"}}}

    def test_wrong_split_refused(self):
        rows = [_row(split="final")]
        with pytest.raises(extindex.ExtIndexError, match="split"):
            extindex._validate(self.CFG, rows)

    def test_incomplete_condition_matrix_refused(self):
        rows = [_row(observation_id=f"S__sky_{i:06d}", time_of_day=t, clouds=c)
                for i, (t, c) in enumerate((t, c) for t in ("DAY", "DAWN") for c in ("CLEAR",))]
        with pytest.raises(extindex.ExtIndexError, match="exactly 9"):
            extindex._validate(self.CFG, rows)

    def test_duplicate_cell_refused(self):
        conds = [(t, c) for t in ("DAY", "DAWN", "DUSK") for c in ("CLEAR", "CLOUDY", "VERY_CLOUDY")]
        conds[8] = conds[0]                                  # 9 members, one duplicated cell
        rows = [_row(observation_id=f"S__sky_{i:06d}", time_of_day=t, clouds=c)
                for i, (t, c) in enumerate(conds)]
        with pytest.raises(extindex.ExtIndexError, match="duplicate condition cells"):
            extindex._validate(self.CFG, rows)


@pytest.mark.skipif(not (INDEX_DIR / "index.json").exists(), reason="extended index not built")
class TestCommittedIndex:
    def test_totals_and_partition(self):
        cfg = _cfg()
        idx = extindex.load_index(INDEX_DIR)
        rows = idx["rows"]
        assert len(rows) == 2395
        assert sum(1 for r in rows if r["kind"] == "anchor_capture") == 603
        assert sum(1 for r in rows if r["kind"] == "swipe") == 1792
        assert sum(1 for r in rows if r["gt_status"] != "ok") == 46
        # the split is EXACTLY by level (spec FR-003)
        for r in rows:
            assert r["split"] == cfg["levels"][r["level"]]["split"], r["observation_id"]
        assert {r["level"] for r in rows if r["split"] == "dev"} == {
            "asian_village_hills_background"}

    def test_every_anchor_has_nine_cells(self):
        idx = extindex.load_index(INDEX_DIR)
        per: dict = {}
        for r in idx["rows"]:
            if r["kind"] == "anchor_capture":
                per.setdefault((r["level"], r["anchor_id"]), set()).add(
                    f"{r['time_of_day']}+{r['clouds']}")
        assert len(per) == 67
        assert all(len(v) == 9 for v in per.values())

    def test_hard_tags_only_on_hard_paths(self):
        idx = extindex.load_index(INDEX_DIR)
        tagged = {r["path_id"] for r in idx["rows"] if r["hard_tag"]}
        assert tagged == {"hard", "HardSwipe"}
        untagged_swipes = {r["path_id"] for r in idx["rows"]
                           if r["kind"] == "swipe" and not r["hard_tag"]}
        assert all("hard" not in p.lower() for p in untagged_swipes)

    def test_swipe_rows_carry_leg_decomposition(self):
        idx = extindex.load_index(INDEX_DIR)
        swipes = [r for r in idx["rows"] if r["kind"] == "swipe"]
        with_leg = [r for r in swipes if r["leg_id"] is not None]
        assert with_leg and all(r["leg_axis"] in ("lateral", "longitudinal", "vertical")
                                for r in with_leg)
        assert {r["phase"] for r in swipes} == {"outbound", "inbound", "center"}
