"""Simulator skyline retrieval — the SKY lane's preparation for the UE5 experiments (PROT-SKY-001).

State, 2026-09-01: the simulator export format is fixed enough to integrate, and this package now
implements the whole chain up to (but deliberately not including) a final experiment.

============== ====================================================================================
module         what it does
============== ====================================================================================
``conventions`` the single UE→ENU seam: cm→m, the declared axis mapping, and the checks that verify
                the declaration against the data rather than trusting field names
``adapter``     simulator run directory → the frozen spec-006 observation record. The only
                format-dependent module in the path; refuses invalid runs and repairs nothing
``simgt``       simulator sky mask → skyline curve. Actual ground truth, ``oracle:sim_exact``;
                invalid columns are preserved, never filled in
``sources``     the three interchangeable ``CurveSource``s — simulator GT, SegFormer silver, frozen DP
``geometry``    pose-only viewpoint offsets, stages, bins, grids and pose-defined correctness
``groups``      the human-authored experiment sidecar: identity is declared, geometry is derived
``sets``        spec-004 query sets and spec-006 reference sets, in the frozen contracts
``report``      GT deformation vs translation, extraction accuracy vs GT, matcher-variant tables
``render``      the pair panel and the trend plots — real records only, no demo mode
``cli``         config-driven ``validate`` / ``ingest`` / ``extract-dp`` / ``silver-list`` /
                ``groups`` / ``tasks`` / ``sets``
============== ====================================================================================

**No experiment has been run and no simulator data exists in this repository.** There is no ``run``
or ``evaluate`` command here on purpose: the first simulator retrieval experiment is a pre-registered
event that runs the frozen C0 NCC baseline on all three curve sources *before* any matcher variant
(``PROT-SKY-001`` §5; the brief's Part H). The candidate variants live in ``hsreloc.matchers`` and are
additive — ``hsreloc/retrieval/`` is unchanged.
"""
