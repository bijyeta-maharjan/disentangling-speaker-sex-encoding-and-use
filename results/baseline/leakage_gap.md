# Leakage Gap (H1) — Day 2

_Generated 2026-06-17 · seed 42 · SI git 39a40f6 · SD git 39a40f6_

Same architecture, seed and training budget for both regimes. The **speaker_dependent** number is a *memorization-inflated diagnostic* (speakers appear in both train and test), not a generalization estimate.

| Regime | Test acc | Test speakers | Test utts | Note |
|---|---|---|---|---|
| speaker_independent (no leakage) | **0.9545** | 12 | 6000 | honest generalization |
| speaker_dependent (leaky diagnostic) | 0.9872 | 60 | 6000 | upper bound w/ memorization |

**Leakage gap (SD − SI): +0.0327**

## Per-sex accuracy (speaker-level mean ± bootstrap 95% CI)

| Regime | female | male | CIs overlap? |
|---|---|---|---|
| speaker_independent | 0.914 [0.770, 0.998] (n=3) | 0.968 [0.940, 0.990] (n=9) | True |
| speaker_dependent | 0.985 [0.973, 0.994] (n=12) | 0.988 [0.983, 0.992] (n=48) | True |

## Interpretation

The model reaches 98.7% when test speakers were also seen in training but only 95.5% on unseen speakers — a gap of 3.3 points. A positive gap means the model leans on speaker-identity cues that do not transfer to new speakers; the larger the gap, the more the apparent skill is speaker memorization rather than digit recognition. **Only the speaker_independent number is a valid generalization estimate; speaker_dependent is reported solely to size the leakage.**

Figures: `figures/output/leakage_gap.png`, `confusion_<regime>.png`, `acc_by_sex_<regime>.png`, `reliability_<regime>.png`, `confidence_dist_<regime>.png`.
