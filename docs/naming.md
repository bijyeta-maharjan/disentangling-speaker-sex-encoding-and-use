# Naming

A few configs, scripts and output files keep chronological prefixes from the project:

| Prefix | Meaning | Examples |
|---|---|---|
| `day2_` | cnn1d baseline: preprocessing, splits, training protocol | `configs/day2_baseline.yaml` |
| `day3_` | layer-wise probing of the cnn1d baseline | `configs/day3_probing.yaml`, `figures/output/day3_decodability_*.png` |
| `week2_` | the four-architecture, three-seed comparison (training sweep, penultimate probing, CKA) | `configs/week2_architectures.yaml`, `configs/week2_probing.yaml`, `scripts/run_week2_sweep.py`, `scripts/week2_probe.py`, `scripts/week2_cka_funnel.py` |

Other names used in code:

- **SI / SD** — speaker-independent split (primary) / speaker-dependent split (memorisation diagnostic).
- **core / stretch** architectures — the four architectures in the paper / two further
  architectures that were implemented but never trained (`tcn`, `conv_attn`).
- **margin** — trained-minus-untrained probe accuracy.
- **û, whitened direction** — the sex direction: a logistic probe fit on per-unit
  standardised penultimate activations, mapped back to raw coordinates and normalised.
- **male control** — the same fine-tuning applied to the male-only model's validation
  speakers (a confounded reference, not a specificity measure).
