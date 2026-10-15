# Figure and number sources

Every figure, table and number in `paper/main.tex` → the script that produced it → the
saved result file it rests on. "Redraw" commands read only `results/` and write to
`figures/output/` (git-ignored), so the committed figures in `figures/*.png` and
`paper/figures/*.png` are never overwritten. All commands run from the repository root.

Seven of the committed paper figures (`necessity_all_arch`, `male_only_gap`,
`whitened_direction_null`, `direction_vs_f0`, `readout_jacobian_heatmap`,
`counterfactual_shift`, `finetune_recovery`) are the script output with the title area
cropped off for the paper; the redrawn versions keep a title.

## Figures

| Paper figure | File | Produced by | Redraw from saved results | Result file(s) |
|---|---|---|---|---|
| Fig. 1 (study design) | — | TikZ in `paper/main.tex` | — | — |
| Fig. 2a | `sex_accent_heatmap.png` | `figures/plot_eda.py` | needs the dataset: `python scripts/download_data.py && python scripts/compute_acoustic_features.py && python figures/plot_eda.py` | AudioMNIST `audioMNIST_meta.txt` (not redistributed) |
| Fig. 2b | `f0_by_sex.png` | `scripts/compute_acoustic_features.py` → `figures/plot_eda.py` | `python figures/plot_eda.py --f0-only` | `results/eda/acoustic_features.csv` |
| Fig. 3a | `leakage_gap.png` | `scripts/train.py` + `scripts/evaluate.py` (configs/day2_baseline.yaml) | `python figures/plot_baseline.py` | `results/baseline/{speaker_independent,speaker_dependent}/eval_metrics.json` |
| Fig. 3b | `accuracy_by_sex.png` | same | `python figures/plot_baseline.py` | `results/baseline/speaker_independent/eval_metrics.json` |
| Fig. 4a, 4b | `probe_layerwise_sex.png`, `probe_layerwise_speaker.png` | `scripts/extract_features.py` + `scripts/run_probes.py` | `python figures/plot_layerwise_probes.py --decodability-only --run results/probing_layerwise_cnn1d` (writes `day3_decodability_{sex,speaker_id}.png`) | `results/probing_layerwise_cnn1d/probe_results.csv` |
| Fig. 5a | `necessity_all_arch.png` | `scripts/xai_freq_occlusion.py` (GPU) | `python figures/plot_necessity_profiles.py` | `results/freq_occlusion/occlusion_results.csv` |
| Fig. 5b | `dftlrp_band_share.png` | `scripts/xai_freq_occlusion.py`, `scripts/xai_dft_lrp.py` (GPU) | `python figures/plot_dftlrp_band_share.py` | `results/freq_occlusion/occlusion_results.csv`, `results/dft_lrp/dftlrp_band_relevance.csv` |
| Fig. 6a | `readout_jacobian_heatmap.png` | `scripts/readout_sensitivity.py` | `python scripts/readout_sensitivity.py --redraw` | `results/sex_direction/readout_sensitivity_seed42.csv` |
| Fig. 6b | `counterfactual_shift.png` | `scripts/readout_counterfactual.py` | `python scripts/readout_counterfactual.py --redraw` | `results/sex_direction/readout_counterfactual.csv` |
| Fig. 7 | `male_only_gap.png` | `scripts/cross_sex_generalization.py` + `scripts/cross_sex_controls.py` | `python scripts/cross_sex_controls.py --redraw` | `results/cross_sex_controls/same_size_control.csv` |
| Fig. 8a | `finetune_recovery.png` | `scripts/female_finetune.py` → `scripts/female_finetune_report.py` | `python scripts/female_finetune_report.py` (also rewrites its tables, identically) | `results/female_finetune/recovery_raw_results.json` |
| Fig. 8b | `finetune_band_change.png` | `scripts/female_finetune_band_shift.py` (GPU) | `python figures/plot_finetune_band_change.py` | `results/female_finetune_band_shift/band_relevance_shift.csv` |
| Fig. A1a | `seed_margins.png` | `scripts/week2_probe.py` (one run per seed) | `python figures/plot_seed_margins.py` | `results/probing_penultimate/seed{42,43,44}/margins.csv` |
| Fig. A1b | `dftlrp_faithfulness.png` | `scripts/xai_dft_lrp.py` (GPU) | **not redrawable**: only the AUCs were saved, not the curves; re-run the script | `results/dft_lrp/rerun/faithfulness_summary.json` (the published PNG comes from this repeat run; its AUCs equal `results/dft_lrp/faithfulness_summary.json` to 6 d.p.) |
| Fig. A1c | `whitened_direction_null.png` | `scripts/direction_resplit_control.py` | **not redrawable**: the 219 null cosines were not saved, only their percentiles; re-run the script (CPU, needs the activation caches) | `results/sex_direction/resplit_control.csv` |
| Fig. A1d | `direction_vs_f0.png` | `scripts/direction_f0_correlation.py` | **not redrawable**: per-recording projections were not saved; re-run the script (CPU, needs the waveform and activation caches) | `results/sex_direction/f0_correlation.csv` |

## Tables

| Paper table | Source |
|---|---|
| Table I (settings) | `configs/week2_architectures.yaml`, `configs/week2_probing.yaml`, `configs/day2_baseline.yaml`, script defaults (see each script's docstring) |
| Table II, columns "Sex, seed 42" | `results/probing_penultimate/seed42/margins.csv` (`trained_acc`, `untrained_acc`, `margin`; probe = linear, attribute = sex) — `scripts/week2_probe.py` |
| Table II, columns "Margin, 3 seeds" | `results/probing_penultimate/multiseed_margin_summary.csv` (`mean`) — recomputed and checked by `figures/plot_seed_margins.py` |
| Table III (RQ map) | text only |

## Numbers in the text

| Claim (section) | Result file | Producer |
|---|---|---|
| 30,000 recordings, 60 speakers, 41/60 German accent; F0 ranges (Data) | AudioMNIST metadata; `results/eda/acoustic_features.csv` (10th–90th percentile of median F0 by sex) | `scripts/compute_acoustic_features.py` |
| SI split 36/12/12, 3 female test speakers (Data) | `splits/speaker_independent.json` | `scripts/make_splits.py` |
| 95.5% SI, 98.7% SD, 3.3-point gap; 0.914 / 0.968 with CIs (Baseline) | `results/baseline/*/eval_metrics.json`, `results/baseline/leakage_gap.md` | `scripts/train.py`, `scripts/evaluate.py` |
| Digit 9 hardest, confused with 7, 0, 1 | `results/baseline/speaker_independent/confusion_matrix.csv` | `scripts/evaluate.py` |
| "runs stopped after 5–8 epochs"; per-run accuracies | `results/training/results.jsonl`, `results/training/arch_table.md` | `scripts/run_week2_sweep.py` |
| Sex 0.99 (conv3), identity 0.92, untrained/raw ≈ 0.97, raw identity 0.94 | `results/probing_layerwise_cnn1d/probe_results.csv`, `day3_summary.json` | `scripts/run_probes.py` |
| Sex margins −0.06 to −0.01; identity +0.08 to +0.13, +0.38 | `results/probing_penultimate/multiseed_margin_summary.csv` | `scripts/week2_probe.py` (per seed) |
| CKA 0.93; 0.60–0.69 | `results/probing_penultimate/cka_matrix.csv` | `scripts/week2_cka_funnel.py` |
| Largest drops 250–750 Hz; > 1.5 kHz costs; peaks 4.6–25 points | `results/freq_occlusion/occlusion_results.csv`, `necessity_summary.csv` | `scripts/xai_freq_occlusion.py`, `figures/plot_necessity_profiles.py` |
| 88–90% below 750 Hz; largest share 0–250 Hz; r = 0.62 / 0.67; female–male r = 0.41–0.93, 625 Hz | `results/dft_lrp/dftlrp_band_relevance.csv`, `results/dft_lrp/present_vs_used_summary.csv` | `scripts/xai_dft_lrp.py`, `figures/plot_dftlrp_band_share.py` (prints the shares, rewrites the summary identically) |
| Faithfulness AUCs 0.15 / 0.41 / 0.54 and 0.16 / 0.39 / 0.55 | `results/dft_lrp/faithfulness_summary.json` | `scripts/xai_dft_lrp.py` |
| Within-digit decodability 0.94–0.95; GRU 0.80; GRU gate | `results/within_digit_geometry/conditional_decodability.csv`, `sex_probe_gate.csv` | `scripts/geom_within_digit.py` |
| Top-20 overlap at chance, ρ ≈ 0 | `results/sex_direction/unit_seed_stability.csv` | `scripts/unit_selectivity.py --stage seed-stability` |
| PCA 94–98% in ten directions; raw mean-difference confound | `results/sex_direction/direction_stability.csv`, `direction_stability_run_meta.json` | `scripts/direction_stability.py` |
| 219-split null: exceeds all in cnn1d, spec_cnn2d; inside for resnet1d, GRU | `results/sex_direction/resplit_control.csv` | `scripts/direction_resplit_control.py` |
| F0 ρ = 0.66 / 0.64; within men 0.19 / 0.13; within women 0.06–0.08 | `results/sex_direction/f0_correlation.csv` (+ `f0_vs_formant_correlation.csv`, `f0_correlation_crossseed.csv`) | `scripts/direction_f0_correlation.py`, `scripts/direction_formant_correlation.py` |
| Angles 82.5° / 83.1° / 83.5° | `results/within_digit_geometry/direction_geometry.json` | `scripts/geom_within_digit.py` |
| Layer-wise 79–90° | `results/layer_angle/layer_angle_table.csv` | `scripts/layer_sex_gradient_angle.py` |
| Readout ≤ 35th percentile (seed 42); 58/60 below median, max 53rd, 0/60 > 95th | `results/sex_direction/readout_sensitivity_seed42.csv`, `readout_crossseed.csv` | `scripts/readout_sensitivity.py`, `scripts/readout_counterfactual.py` |
| Counterfactual Δaccuracy < 0.002 | `results/sex_direction/readout_counterfactual.csv` | `scripts/readout_counterfactual.py` |
| Male-only loss 7.9 / 5.0 points (3 held-out women); confounded all-12 comparison | `results/cross_sex_generalization/results_overall.csv`, `run_meta.json` | `scripts/cross_sex_generalization.py` |
| Same-size control 2.9 / 2.1 | `results/cross_sex_controls/same_size_control.csv` | `scripts/cross_sex_controls.py` |
| Gap survives leaving out any one woman | `results/cross_sex_controls/leave_one_out.csv` | `scripts/cross_sex_controls.py` |
| Gap direction stable across seeds | `results/cross_sex_generalization/results_per_digit.csv` | `scripts/cross_sex_generalization.py` |
| Fine-tuning 0.87 → 0.96, 0.86 → 0.96; all folds positive; male control ≈ 0 | `results/female_finetune/recovery_summary_table.csv`, `recovery_fold_seed_avg.csv` | `scripts/female_finetune.py`, `scripts/female_finetune_report.py` |
| Pilot: no gain from a third epoch | `results/female_finetune/pilot_epoch_gate.json` | `scripts/female_finetune.py --mode pilot` |
| Second-environment reproduction within 0.7 points | `results/female_finetune_band_shift/consistency_check_vs_recovery.json` vs. `results/female_finetune/recovery_raw_results.json` | `scripts/female_finetune_band_shift.py` |
| Change concentrates in late layers | `results/female_finetune/layer_change_profile.csv` | `scripts/female_finetune_layer_change.py`, `scripts/female_finetune_layer_change_report.py` |
| DFT-LRP band change −15% to −27%; male control near zero | `results/female_finetune_band_shift/band_relevance_shift.csv` | `scripts/female_finetune_band_shift.py`, `figures/plot_finetune_band_change.py` |
