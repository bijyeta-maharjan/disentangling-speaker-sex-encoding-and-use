# Disentangling Speaker Sex Encoding and Use in Spoken-Digit Classifiers

Code, configurations, saved results and figure scripts for the paper

> **Disentangling Speaker Sex Encoding and Use in Spoken-Digit Classifiers**<br>
> Bijyeta Maharjan (Mount Holyoke College)<br>
> Supervisor: Dr. Nils Gumpfer (AAI Lab, Technische Hochschule Mittelhessen)

Fairness audits of speech models usually check whether accuracy is equal across
demographic groups, but equal accuracy says nothing about whether a model internally
*encodes* a protected attribute, or whether its decisions *use* it. This study separates
the two claims for spoken-digit classification on AudioMNIST, across four architectures
(a raw-waveform 1-D CNN, a residual 1-D CNN, a convolutional-recurrent GRU model and a
log-mel 2-D CNN) and three training seeds. Linear probes recover speaker sex from the
convolutional networks with up to 99% accuracy, yet an untrained copy of each network
decodes sex just as well: the information is inherited from the audio, not induced by
training. Band occlusion and faithfulness-checked DFT-LRP show that digit decisions rely
on the formant region below 750 Hz. Sex-probe directions lie 79–90° from the digit
decision at every layer. For a sex direction validated against an exhaustive null of 219
speaker re-splits, the readout's sensitivity never exceeds the 95th percentile of random
directions (0 of 60 tests), and shifting representations along it leaves accuracy
unchanged. Behaviourally, a model
trained only on male speakers still loses 5–8 points on unseen female speakers, and
fine-tuning recovers this through late-layer recalibration rather than
frequency-selective change. All sex-related claims rest on three held-out female
speakers.

A condensed account of each experiment (question, design decision, result) is in
[`docs/research_journal.md`](docs/research_journal.md).

## Repository layout

```
src/        library code: data loading/preprocessing, splits, models, training,
            probing, fairness metrics, plotting helpers
configs/    YAML configs (preprocessing, training protocol, architectures, probing)
splits/     committed speaker splits and fine-tuning folds (JSON)
scripts/    one entry point per experiment (see the table below)
figures/    figure scripts + the 17 figures as they appear in the paper
results/    saved result tables behind every number and figure (CSV / JSON / NPY)
tests/      unit tests + a CPU end-to-end smoke test
paper/      main.tex, refs.bib, figures/
docs/       research journal, figure/number sources
Dockerfile  approximate GPU environment for the three GPU-based scripts
```

## Setup

Python 3.12.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                      # unit tests + ~1 min CPU smoke test; dataset tests skip without data
```

`requirements.txt` pins the package versions installed in the laptop environment used to
package this release. **The original runs did not record exact package versions**, so
these are the closest available record, not a guaranteed match; small numerical
differences on re-running are possible. The GPU-based analyses used a different
environment (see below).

Scripts are run from the repository root (`python scripts/<name>.py`); they add the
repository to `sys.path` themselves, so no install step is needed.

## Data

The AudioMNIST dataset is **not** redistributed here. It is available from
<https://github.com/soerenab/AudioMNIST> (60 speakers × 500 recordings, 48 kHz WAV, plus
`audioMNIST_meta.txt`).

```bash
python scripts/download_data.py        # git clone -> data/raw/AudioMNIST/{01..60, audioMNIST_meta.txt}
python scripts/preprocess.py           # 8 kHz, 8,192 samples, centre pad, peak norm
                                       #   -> data/processed/waveforms_8k_8192{.npy,_meta.npz,_manifest.parquet} (~1 GB)
python scripts/build_spectrograms.py   # 64-band log-mel cache for spec_cnn2d (~0.5 GB)
python scripts/compute_acoustic_features.py   # per-utterance F0 etc. for the EDA figures
```

To put the dataset elsewhere, either place it manually at
`data/raw/AudioMNIST/` or set environment variables that move whole roots without
editing configs:

| Variable | Replaces | Holds |
|---|---|---|
| `AMNIST_DATA_ROOT` | `data/` | raw audio, waveform/mel caches, activation caches |
| `AMNIST_EXPERIMENTS_ROOT` | `experiments/` | checkpoints and per-run logs |
| `AMNIST_RESULTS_ROOT` | `results/` | result tables written by the scripts |

The split files in `splits/` are the ones used for every run in the paper and should be
reused as they are (`scripts/make_splits.py` rebuilds the speaker-independent,
speaker-dependent and sex-balanced splits from seed 42).

## Redrawing the paper figures from saved results

These commands read only `results/` and write to `figures/output/` (git-ignored); each
takes a few seconds on a laptop and needs neither the dataset nor a GPU.

```bash
python figures/plot_baseline.py                      # leakage_gap, accuracy_by_sex
python figures/plot_eda.py --f0-only                 # f0_by_sex
python figures/plot_layerwise_probes.py --decodability-only --run results/probing_layerwise_cnn1d
python figures/plot_seed_margins.py                  # seed_margins (+ checks Table II margins)
python figures/plot_necessity_profiles.py            # necessity_all_arch
python figures/plot_dftlrp_band_share.py             # dftlrp_band_share (+ DFT-LRP shares, r values)
python scripts/readout_sensitivity.py --redraw       # readout_jacobian_heatmap
python scripts/readout_counterfactual.py --redraw    # counterfactual_shift
python scripts/cross_sex_controls.py --redraw        # male_only_gap
python scripts/female_finetune_report.py             # finetune_recovery (+ recovery tables)
python figures/plot_finetune_band_change.py          # finetune_band_change
```

Three appendix figures cannot be redrawn from saved data because their per-point inputs
were not saved (DFT-LRP faithfulness curves, the 219 re-split null cosines, per-recording
projections onto the sex direction); re-running the producing script regenerates them.
`sex_accent_heatmap` needs the dataset metadata. Details:
[`docs/figure_sources.md`](docs/figure_sources.md).

## Paper figures and tables → code → results

"Redraw" times are for the commands above. "Re-run" times are the recorded wall-clock
times of the original runs where they were logged (laptop CPU, 8 cores, unless marked
GPU); "not recorded" means no timing was logged.

| Paper item | Script / command | Result file it reads | Expected runtime |
|---|---|---|---|
| Fig. 2a sex × accent | `figures/plot_eda.py` | AudioMNIST metadata | < 1 min (after download) |
| Fig. 2b F0 by sex | `figures/plot_eda.py --f0-only` (data: `scripts/compute_acoustic_features.py`) | `results/eda/acoustic_features.csv` | redraw seconds; features a few min |
| Fig. 3a/b, baseline numbers | `figures/plot_baseline.py` (data: `scripts/train.py`, `scripts/evaluate.py`, config `day2_baseline.yaml`) | `results/baseline/*/eval_metrics.json` | redraw seconds; training about 25–30 min per run (logged estimate) |
| Fig. 4a/b layer-wise probes | `figures/plot_layerwise_probes.py --decodability-only` (data: `scripts/extract_features.py`, `scripts/run_probes.py`) | `results/probing_layerwise_cnn1d/probe_results.csv` | redraw seconds; probing not recorded |
| Table II, Fig. A1a margins | `figures/plot_seed_margins.py` (data: `scripts/run_week2_sweep.py`, `scripts/week2_probe.py` per seed) | `results/probing_penultimate/seed{42,43,44}/margins.csv`, `multiseed_margin_summary.csv` | redraw seconds; training 2.5–72 min per run (16 runs); probing not recorded |
| CKA | `scripts/week2_cka_funnel.py` | `results/probing_penultimate/cka_matrix.csv` | not recorded |
| Fig. 5a necessity | `figures/plot_necessity_profiles.py` (data: `scripts/xai_freq_occlusion.py`, GPU) | `results/freq_occlusion/occlusion_results.csv` | redraw seconds; occlusion not recorded (GPU) |
| Fig. 5b necessity vs. DFT-LRP | `figures/plot_dftlrp_band_share.py` (data: `scripts/xai_dft_lrp.py`, GPU) | `results/freq_occlusion/occlusion_results.csv`, `results/dft_lrp/dftlrp_band_relevance.csv` | redraw seconds; DFT-LRP not recorded (GPU) |
| Fig. A1b faithfulness | `scripts/xai_dft_lrp.py` (GPU) | `results/dft_lrp/faithfulness_summary.json` | not recorded (GPU) |
| Within-digit decodability, penultimate angles | `scripts/geom_within_digit.py` (GPU) | `results/within_digit_geometry/` | ~2 h (7,065 s, GPU) |
| Unit overlap, PCA | `scripts/unit_selectivity.py --stage seed-stability`, `scripts/direction_stability.py` | `results/sex_direction/unit_seed_stability.csv`, `direction_stability*.{csv,json}` | not recorded |
| Fig. A1c 219-split null | `scripts/direction_resplit_control.py` | `results/sex_direction/resplit_control.csv` | not recorded |
| Fig. A1d, F0/formant correlations | `scripts/direction_f0_correlation.py`, `scripts/direction_formant_correlation.py` | `results/sex_direction/f0_*.csv` | not recorded |
| Layer-wise angles | `scripts/layer_sex_gradient_angle.py` | `results/layer_angle/layer_angle_table.csv` | not recorded |
| Fig. 6a readout sensitivity | `scripts/readout_sensitivity.py [--redraw]` | `results/sex_direction/readout_sensitivity_seed42.csv` | redraw seconds |
| Fig. 6b counterfactual, 60-test summary | `scripts/readout_counterfactual.py [--redraw]` | `results/sex_direction/readout_counterfactual.csv`, `readout_crossseed.csv` | redraw seconds |
| Fig. 7 male-only gap | `scripts/cross_sex_controls.py --redraw` (data: `scripts/make_cross_sex_split.py`, `scripts/cross_sex_generalization.py`, `scripts/make_male_subset_control_split.py`, `scripts/cross_sex_controls.py`) | `results/cross_sex_controls/same_size_control.csv`, `results/cross_sex_generalization/results_overall.csv` | redraw seconds; 6 male-only + 6 same-size control training runs, not recorded |
| Fig. 8a fine-tuning recovery | `scripts/female_finetune_report.py` (data: `scripts/pin_male_only_checkpoints.py`, `scripts/make_*_finetune_localization_folds.py`, `scripts/female_finetune.py --mode full --epochs_locked 2`) | `results/female_finetune/recovery_raw_results.json` | redraw seconds; 48 fine-tuning runs, not recorded |
| Late-layer change | `scripts/female_finetune_layer_change.py` → `scripts/female_finetune_layer_change_report.py` | `results/female_finetune/layer_change_profile.csv` | not recorded |
| Fig. 8b band change | `figures/plot_finetune_band_change.py` (data: `scripts/female_finetune_band_shift.py --include-male-control`, GPU) | `results/female_finetune_band_shift/band_relevance_shift.csv` | redraw seconds; not recorded (GPU) |

## Re-running the experiments

In the order of the paper (all CPU unless marked):

```bash
# training: 4 architectures x seeds 42/43/44 (speaker-independent) + seed 42 speaker-dependent
python scripts/run_week2_sweep.py --phases core_si_seed42 core_si_extra_seeds core_sd_seed42
# cnn1d baseline (Fig. 3) with the baseline config, both regimes, then evaluation
python scripts/train.py --regime speaker_independent && python scripts/evaluate.py --regime speaker_independent
python scripts/train.py --regime speaker_dependent   && python scripts/evaluate.py --regime speaker_dependent

# probing
python scripts/extract_features.py && python scripts/run_probes.py     # layer-wise, cnn1d (edit checkpoint path in configs/day3_probing.yaml)
python scripts/week2_probe.py                                         # penultimate, 4 archs, seed 42
python scripts/week2_cka_funnel.py

# input level (GPU, see next section)
python scripts/xai_freq_occlusion.py --seed 42
python scripts/xai_dft_lrp.py --seed 42

# representation level
python scripts/geom_within_digit.py --seeds 42
python scripts/unit_selectivity.py --stage selectivity
python scripts/unit_selectivity.py --stage seed-stability
python scripts/direction_stability.py
python scripts/direction_resplit_control.py
python scripts/direction_f0_correlation.py
python scripts/direction_formant_correlation.py
python scripts/layer_sex_gradient_angle.py
python scripts/readout_sensitivity.py
python scripts/readout_counterfactual.py

# behaviour
python scripts/make_cross_sex_split.py && python scripts/cross_sex_generalization.py
python scripts/make_male_subset_control_split.py && python scripts/cross_sex_controls.py
python scripts/pin_male_only_checkpoints.py
python scripts/make_female_finetune_localization_folds.py
python scripts/make_male_control_finetune_localization_folds.py
python scripts/female_finetune.py --mode pilot --epochs 2 3
python scripts/female_finetune.py --mode full --epochs_locked 2
python scripts/female_finetune_report.py
python scripts/female_finetune_layer_change.py && python scripts/female_finetune_layer_change_report.py
python scripts/female_finetune_band_shift.py --include-male-control   # GPU
```

The split and fold builders are deterministic (seed 42) and reproduce the committed JSON
files; skip them to keep the committed splits. The scripts that compare against the
original speaker-independent checkpoints (`cross_sex_generalization.py`,
`cross_sex_controls.py`, `layer_sex_gradient_angle.py`) list the original run directories
in `SI_BASELINE_CHECKPOINTS` (`scripts/cross_sex_generalization.py`); after retraining,
point them at your own `experiments/` run directories. Re-running an analysis overwrites
its tables in `results/` — compare with `git diff`.

## Running the GPU-based analyses

`scripts/xai_freq_occlusion.py`, `scripts/xai_dft_lrp.py` and
`scripts/female_finetune_band_shift.py` were run on a CUDA GPU. The original environment
was:

- a CUDA GPU (the original runs used NVIDIA T4s);
- Ubuntu 24.04, Python 3.12, PyTorch 2.9, CUDA 13;
- numpy 2.1, scikit-learn 1.7.1, librosa 0.11.0, zennit (version not recorded).

The [`Dockerfile`](Dockerfile) approximates this environment (it has not been built and
checked against the published numbers):

```bash
docker build -t speaker-sex-gpu .
docker run --gpus all -it -v "$PWD":/work -w /work speaker-sex-gpu bash
```

**Inputs.**

| Input | Created by |
|---|---|
| cached 8 kHz waveforms, `data/processed/waveforms_8k_8192*` | `python scripts/download_data.py && python scripts/preprocess.py` |
| speaker-independent checkpoints for seeds 42/43/44, `experiments/speaker_independent_seed{S}_*/best.pt`, and the sweep log `experiments/week2_sweep/results.jsonl` (training `spec_cnn2d` also needs the log-mel cache from `scripts/build_spectrograms.py`; occlusion itself recomputes mels from the occluded waveform) | `python scripts/run_week2_sweep.py --phases core_si_seed42 core_si_extra_seeds` |
| (band shift only) pinned male-only checkpoints and fine-tuning folds | `scripts/make_cross_sex_split.py`, `scripts/cross_sex_generalization.py`, `scripts/pin_male_only_checkpoints.py`, `scripts/make_*_finetune_localization_folds.py` |

```bash
python scripts/xai_freq_occlusion.py --seed 42 --device cuda       # -> results/freq_occlusion/
python scripts/xai_dft_lrp.py --seed 42 --device cuda              # -> results/dft_lrp/
python scripts/female_finetune_band_shift.py --include-male-control --device cuda   # -> results/female_finetune_band_shift/
```

**Missing helper: `si_checkpoints`.** These scripts import `si_checkpoints` from
`src/probing/extract_week2.py`
(`female_finetune_band_shift.py` imports it indirectly through `xai_dft_lrp.py`, so it
fails at import time too). That function existed only on the original GPU machine and
**is not included**; as released, the three scripts stop with an `ImportError`. To run
them, add a function to `src/probing/extract_week2.py` that satisfies the contract below.

It is called as `ckpts = si_checkpoints(cfg)` (not with a bare seed), where `cfg` is
`configs/week2_probing.yaml` loaded as a dict with `cfg["seed"]` overwritten by the
script's `--seed` (default 42). It must return a `dict[str, pathlib.Path]` mapping each
architecture name to the `best.pt` of the **speaker-independent** run trained with that
seed:

- keys: `cnn1d`, `resnet1d`, `crnn_gru`, `spec_cnn2d` — `xai_freq_occlusion.py` asserts all
  four are present; `xai_dft_lrp.py` needs at least the requested `--archs`
  (default `cnn1d,resnet1d`);
- values: existing paths (`.exists()` is asserted); each file is loaded with
  `torch.load(..., weights_only=False)` and must contain `"model_state"` (and optionally
  `"config"`), as written by `src/train/trainer.py`.

The laptop-side code already contains two equivalent lookups that read
`experiments/week2_sweep/results.jsonl`: `si_seed42_checkpoints(cfg)` in the same module
(seed 42 only) and `seed_checkpoints(cfg, seed)` in `scripts/unit_selectivity.py`. A
function returning `seed_checkpoints(cfg, cfg["seed"])` meets the contract.

**Saved outputs** of the original GPU runs: `results/freq_occlusion/`,
`results/dft_lrp/` (and `results/dft_lrp/rerun/`, a repeat run whose faithfulness AUCs
match to 6 decimal places and whose cnn1d faithfulness plot is the paper's Fig. A1b) and
`results/female_finetune_band_shift/`.

## Known limitations

- **Missing `si_checkpoints` helper.** See above: the three GPU-based scripts do not run
  as released.
- **Seed-43/44 probing configs are not included.** `configs/week2_probing_seed43.yaml`
  and `configs/week2_probing_seed44.yaml` existed only on the GPU machine. They would be
  copies of `configs/week2_probing.yaml` with `seed: 43` / `seed: 44`,
  `paths.activations_dir: data/processed/activations_week2_seed43` (`_seed44`) and
  `paths.runs_dir: experiments/week2_probing_seed43` (`_seed44`), the directory names
  used by the saved runs. Unknown: whether any other field differed, the exact seed used
  to initialise the untrained control (the seed-43/44 untrained accuracies differ from
  seed 42's, so a different initialisation was used), and the `random_state` of the
  identity probe's 70/30 split for those seeds. In addition, the released
  `run_extraction` in `src/probing/extract_week2.py` always loads the **seed-42**
  checkpoints; the seed-43/44 runs relied on the same seed-aware checkpoint lookup as
  `si_checkpoints`. Re-running `scripts/week2_probe.py` for seeds 43/44 (Table II's
  3-seed margins, Fig. A1a) or `scripts/geom_within_digit.py --seeds 42,43,44` therefore
  needs both the configs and that lookup; the saved per-seed results are in
  `results/probing_penultimate/seed43/` and `seed44/`.
- **Three held-out women.** All sex-related results rest on the three female speakers in
  the speaker-independent test fold (and, for fine-tuning, on folds of three women). They
  are within-sample statements about those speakers, not population estimates.
- **Package versions.** No run recorded its package versions; `requirements.txt` and the
  Dockerfile are the closest available record.
- **Stretch architectures.** `src/models/tcn.py` and `src/models/conv_attn.py` are
  implemented and unit-tested but were never trained and are not part of the paper; they
  are kept because the model factory, configs and tests reference them.

## Citation

```bibtex
@misc{maharjan2026disentangling,
  title  = {Disentangling Speaker Sex Encoding and Use in Spoken-Digit Classifiers},
  author = {Maharjan, Bijyeta and Gumpfer, Nils},
  year   = {2026},
  note   = {Code: https://github.com/bijyeta-maharjan/disentangling-speaker-sex-encoding-and-use}
}
```

See also [`CITATION.cff`](CITATION.cff).

## License

Code: MIT (see [`LICENSE`](LICENSE)). The AudioMNIST dataset is not included and keeps its
own license.
