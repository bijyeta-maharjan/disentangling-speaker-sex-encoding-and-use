# Figure changelog

## Redesign to IEEE style (October 2026)

All paper figures except `dftlrp_faithfulness` were redrawn from the saved results in
`results/` with a shared style module, `figures/ieee_style.py`. No model was retrained and
no number changed.

**Common to every redrawn figure**

- Drawn at the final printed size (IEEE column 3.5 in, page width 7.16 in), so fonts are
  true size: serif STIXGeneral (mathtext "stix"); axis labels 8 pt; ticks, legends and
  annotations 7 pt; nothing smaller.
- No figure titles, explanatory notes or statistics text boxes; these belong in the LaTeX
  captions. Short panel titles name the model only.
- Plain axis labels with units (e.g. "Accuracy drop (pp)", "Frequency band centre (Hz)").
- Model names: 1-D CNN, Residual CNN, GRU-CRNN, Spectrogram CNN (no code identifiers).
- Okabe-Ito colours with one meaning everywhere: female #D55E00, male #0072B2; models
  1-D CNN #0072B2 (o), Residual CNN #E69F00 (s), GRU-CRNN #009E73 (^), Spectrogram CNN
  #CC79A7 (D), always with markers; necessity #0072B2 vs. attribution #E69F00; null
  distribution #BBBBBB, observed value black, confounded reference grey dashed; trained
  solid, untrained dashed, raw input dotted, chance thin grey dash-dot.
- Lines 1.0 pt, markers 3–3.5 pt, thin error bars with small caps, light grid
  (0.3 pt #DDDDDD), no top/right spines, frameless legends placed off the data.
- Speech regions shaded and labelled "F0" (0–250 Hz) and "F1/F2" (250–750 Hz).
- Each figure saved as a vector PDF (TrueType fonts) and a 600-dpi PNG with the paper's
  base filename, in both `figures/` and `paper/figures/`.

**Per figure**

| Figure | Size (in) | Script | Changes |
|---|---|---|---|
| `sex_accent_heatmap` | 2.0 × 2.0 | `figures/plot_eda.py` | Redrawn from the AudioMNIST metadata. Removed the title ("… over SPEAKERS (n=60)") and colour bar; cell counts at 7 pt; rows sorted by count; "Female"/"Male" columns. |
| `f0_by_sex` | 3.1 × 2.0 | `figures/plot_eda.py` | Removed the title; female/male colours; axis "Median F0 per utterance (Hz)". Same KDE of the same 600-utterance sample. |
| `leakage_gap` | 1.7 × 1.8 | `figures/plot_baseline.py` | Removed the title ("Leakage gap = +0.033"); bars labelled "Unseen speakers" / "Familiar speakers" instead of split codes; neutral greys. |
| `accuracy_by_sex` | 1.7 × 1.8 | `figures/plot_baseline.py` | Removed the title and the text block inside the bars; bars replaced by the per-speaker accuracies plus mean with bootstrap 95% interval; n in the tick labels. |
| `probe_layerwise_sex`, `probe_layerwise_speaker` | 2.9 × 1.9 | `figures/plot_layerwise_probes.py` | Removed titles; trained solid / untrained dashed (1-D CNN colour), raw input dotted, chance dash-dot; axis "Probe accuracy". |
| `necessity_all_arch` | 2.85 × 1.7 | `figures/plot_necessity_profiles.py` | Removed the title, the "8 kHz SR …" note and the three literature band labels; F0 and F1/F2 regions shaded and labelled; y in percentage points; model names and markers. |
| `dftlrp_band_share` | 4.15 × 1.7 | `figures/plot_dftlrp_band_share.py` | Panel titles now just the model name (r values moved to the caption); necessity/attribution colours; same normalisation (each profile / its sum, negative drops clipped to 0). |
| `readout_jacobian_heatmap` | 3.5 × 1.3 | `scripts/readout_sensitivity.py --redraw` | Removed the five-line title block; model names; 7-pt cell values; colour bar "Percentile". |
| `counterfactual_shift` | 3.5 × 1.5 | `scripts/readout_counterfactual.py --redraw` | Removed titles and the in-plot notes naming the direction vector and the swap position; model colours and markers; x axis "Shift δ (× sex gap)"; accuracy axis 0.95–1.0; legend moved above both panels in one row so the δ = 0 line no longer crosses it. |
| `male_only_gap` | 2.2 × 1.8 | `scripts/cross_sex_controls.py --redraw` | Removed the title and the internal shorthand / power notes in the legend; legend now "Trained without women, tested on women" (female colour) and "Same-size control, tested on men" (male colour), placed above the axes and aligned to the figure edge so it stays within the 2.2-in width; y in percentage points. |
| `finetune_recovery` | 3.9 × 1.9 | `scripts/female_finetune_report.py --redraw` | Removed the title and the italic note under the plot; "Women" vs. "Male control"; y in percentage points; panel titles name the model. |
| `finetune_band_change` | 3.1 × 1.9 | `figures/plot_finetune_band_change.py` | Removed the title; "Women" vs. "Male control"; regions labelled; y "Relevance change (%)". |
| `seed_margins` | 6.3 × 2.0 | `figures/plot_seed_margins.py` | Removed the "(mean ± seed-range, n=3 seeds)" titles; bars in model colours with the three seeds shown as markers; panel titles "Sex" / "Speaker identity". |
| `whitened_direction_null` | 3.0 × 2.2 | `scripts/direction_resplit_control.py --redraw` | See below. |
| `direction_vs_f0` | 6.3 × 2.2 | `scripts/direction_f0_correlation.py --redraw` | See below. |
| `dftlrp_faithfulness` | — | — | **Unchanged** (needs the GPU script to redraw); only a PNG exists. |

**Appendix figures redrawn from new per-point files**

`whitened_direction_null` (Fig. A1c) and `direction_vs_f0` (Fig. A1d) had no saved per-point
data, so `scripts/direction_resplit_control.py` and `scripts/direction_f0_correlation.py`
were re-run once on CPU from the cached penultimate activations (no training). They now
also write:

- `results/sex_direction/resplit_null_cosines.csv`: per model, the 219 re-split null
  cosines (`kind = null`) and the observed sex-direction cosine (`kind = sex`).
- `results/sex_direction/f0_projection_per_recording.csv`: per recording (1-D CNN and
  Spectrogram CNN), median F0 and the projection on the sex direction.

The re-run reproduced `resplit_control.csv` and `f0_correlation.csv` byte-for-byte, and the
per-point files reproduce every summary number in them (sex cosine, null median and 95th
percentile, percentile rank, Spearman ρ and n per stratum). The run-meta JSONs from the
re-run were not kept, so every existing file in `results/` is unchanged.

- `whitened_direction_null`: 2 × 2 grid by model; null histogram light grey, observed sex
  direction black, raw mean-difference reference grey dashed; shared legend above the grid.
  Removed the three-line title, the per-panel legends and the panel-title tags.
- `direction_vs_f0`: one panel per model, female/male colours, speaker counts in the legend;
  removed the title block and the two ρ text boxes (ρ values go in the caption).
- `--redraw` reads the null file with `keep_default_na=False`, because pandas otherwise
  parses the `null` label as missing.

**Other**

- `figures/ieee_style.save` renders each format once and copies it, so `figures/` and
  `paper/figures/` hold byte-identical files.
