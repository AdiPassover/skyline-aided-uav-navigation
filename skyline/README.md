# skyline/

The Python side of the skyline place-evidence component: it turns the horizon views of the UE5
recordings into the one-dimensional profiles the Java integration layer consumes, and it contains
the component studies behind the skyline results.

| Path | Contents |
|---|---|
| `hsreloc/simret/` | Ingest of UE5 recordings into observation sessions (`adapter`, `conventions`), simulator ground-truth sky curves (`simgt`), curve sources, and the study modules (`dualview`, `recog`, `lagstudy`, `relpose`, and the `ext*` modules of the nine-condition appearance matrix). CLI: `python -m hsreloc.simret.cli`. |
| `hsreloc/extraction/` | Skyline extraction methods (the classical `poc_robust_dp` and baselines), the SegFormer label-map to curve conversion, and curve sources. |
| `hsreloc/matchers/`, `hsreloc/retrieval/` | Profile normalisation and the NCC matcher family (C0 pointwise NCC is the adopted matcher; C1 adds a bounded shift and is kept for comparison). |
| `hsreloc/placeret/` | Frame and source helpers the simulator modules reuse. |
| `skyline/` | Profile resampling and descriptor utilities used by `hsreloc`. |
| `scripts/` | Study drivers (`sky_dual_study.py`, `sky_dual_hardcity.py`, `sky_matcher_resolution.py`, `sky_recog_radius.py`), `sim_ext_inventory.py` (inventory and digest of the appearance batch), `silver_infer.py` (SegFormer inference), and helper modules the drivers import. |
| `configs/` | One config per study. `configs/frozen/` holds the pre-registration and method digests the guarded commands compare against. |
| `tests/` | Test suite; all data is generated synthetically at test time. |

Run everything from this directory. Importing `hsreloc` puts `../evaluation` (for `naveval`) and
this directory on `sys.path`, so no installation step is needed. The study drivers set up their own
path; `sim_ext_inventory.py` and `sim_inventory.py` need `PYTHONPATH=.`.

```bash
python -m pytest tests -q
python -m hsreloc.simret.cli validate --config <ingest-config.json>
```

The studies read the simulator recordings from the paths listed in `../docs/data.md`, and
`../docs/workflows.md` summarises their stages and inputs.

Paths inside the study configs resolve relative to the config file, except the SegFormer
`mask_root` entries of `sky-dual*.json` and `sky-matcher-resolution.json`, which are read relative to
the working directory (this directory). By default the label maps are expected under
`../silver_masks/`.

Some commands are guarded. `ext-run`/`ext-exp-*` with `--population final` refuse unless the built
task and index digests match `configs/frozen/sim-final.prereg`, and they refuse to overwrite an
existing complete record. Curves from the classical extractor are only read when the digest of its
parameters, recorded by `extract-dp`, matches `configs/frozen/poc_robust_dp.digest`.

## SegFormer inference

`scripts/silver_infer.py` runs SegFormer-B0 (ADE20K fine-tune,
`nvidia/segformer-b0-finetuned-ade-512-512` at the revision pinned in each `silver-infer-*.json`)
on CPU and writes one label map per view. It imports nothing from this repository and is meant to
run in its own virtual environment with `torch` (CPU build), `transformers` and `pillow`; those
packages are deliberately not project dependencies. The weights are downloaded from Hugging Face on
first use and are subject to NVIDIA's licence for that model (non-commercial use); neither the
weights nor derived label maps are distributed here.

```bash
HF_HOME=<cache dir> <segformer-venv>/python scripts/silver_infer.py --config configs/silver-infer-dual.json --resume
```
