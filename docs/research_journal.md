# Research journal

A condensed record of the experiments reported in *Disentangling Speaker Sex Encoding and
Use in Spoken-Digit Classifiers*, in the order the paper presents them. For each: the
question, the design decision that shaped it and why, and the result as reported in the
paper. Scripts and result files for every number are listed in
[`figure_sources.md`](figure_sources.md).

All work ran between June and September 2026. Training and most analyses ran on a laptop
CPU; band occlusion, DFT-LRP, the within-digit geometry, the seed-43/44 probing and the
fine-tuning band analysis ran on a CUDA GPU.

---

## 1. Data and splits (June 2026)

**Question.** How should AudioMNIST be split so that sex-related claims are about unseen
speakers, not memorised voices?

**Design.** AudioMNIST has 30,000 recordings from 60 speakers (48 male, 12 female, 500
each). Exploratory analysis showed that median F0 nearly separates the sexes (roughly
100–150 Hz for men, 180–260 Hz for women) and that sex and accent are entangled (41 of 60
speakers have a German accent; most non-German sex × accent cells hold one speaker or
none). Recordings were resampled to 8 kHz, centre zero-padded to 8,192 samples and
peak-normalised.

- The primary split is **speaker-independent (SI)**, built with stratified group 5-fold
  partitioning (speakers as groups, stratified by sex, seed 42). Stratifying by sex
  guarantees that the test fold contains women at all: 12 test speakers (3 female, 9
  male), 12 validation (2 female), 36 training (7 female). An automated test enforces zero
  speaker overlap.
- A **speaker-dependent (SD)** split (random 60/20/20 by recording) is kept only as a
  diagnostic for voice memorisation, never as a performance number.
- The same split files are reused by every architecture and seed, so differences between
  models cannot come from different partitions.
- Accent and age were dropped from all later analyses: with this speaker pool, any null
  result for them would be underpowered rather than informative.

**Result.** Because group metrics are computed per speaker, the effective sample size for
any sex comparison is three women. This is treated as a hard ceiling throughout.

## 2. Baseline: the model is accurate and looks fair (June 2026)

**Question.** Does a standard output audit find a sex problem?

**Design.** A raw-waveform 1-D CNN (five conv blocks, 226k parameters) trained with
cross-entropy, Adam (lr 1e-3), batch 128, at most 8 epochs, early stopping on validation
accuracy (patience 3), seed 42, CPU. The SD split was trained with the identical protocol
to size the memorisation gap. Per-sex accuracy is the mean of per-speaker accuracies with
a speaker-level bootstrap interval (1,000 resamples).

**Result.** 95.5% digit accuracy on unseen speakers vs. 98.7% on familiar ones: a
3.3-point memorisation gap, a lower bound because training stopped early. Digit 9 is the
hardest and is confused with 7, 0 and 1. Speaker-level accuracy is 0.914 [0.770, 0.998]
for the three test women and 0.968 [0.940, 0.990] for the nine test men. The intervals
overlap, so a standard audit would pass this model.

## 3. Sex is encoded, but inherited from the audio (June–July 2026)

**Question.** Is sex (and speaker identity) linearly decodable from the network's
layers, and did training put it there?

**Design.**
- Each block's output is average-pooled over time (never flattened, to avoid
  dimensionality artefacts) and read with a standardised, L2-regularised logistic
  regression (C = 1).
- The sex probe is speaker-disjoint (trained on the 48 SI train+val speakers, tested on
  the 12 test speakers; majority chance 0.75). A 60-way identity probe cannot name unseen
  speakers, so it measures *re-identification*: 70/30 split of each speaker's
  recordings.
- The key control: every probe is also fit on an **untrained copy** of the same
  architecture (PyTorch default initialisation, frozen), and for the 1-D CNN on a
  **raw-input** baseline (time-averaged 64-band log-mel spectrum). The *margin* is
  trained minus untrained accuracy. Without these controls, high decodability would have
  been read as "training encodes demographics".
- To test whether the pattern is a property of one design, the study was extended to four
  architectures spanning three inductive-bias axes (1-D CNN; residual 1-D CNN; a conv + GRU
  recurrent model; a 2-D CNN on log-mel spectrograms), capacity-matched to within
  0.84–1.19× of the baseline, each trained from seeds 42, 43 and 44 with the identical
  protocol. Linear CKA (4,000 recordings) compares the architectures' penultimate
  representations.

**Result.** In the 1-D CNN, sex is decodable at up to 0.99 (conv3) and speakers are
re-identified at up to 0.92. But the untrained network and the raw input read sex about
equally well (≈ 0.97). Identity rises to a peak mid-network and falls towards the head,
ending below the raw input (0.94) but well above chance. At the penultimate layer of all
four architectures, the sex margin is negative or near zero across three seeds
(−0.06 to −0.01), while the identity margin is positive (≈ +0.08 to +0.13 for the
convolutional models, +0.38 for the GRU model): **sex is inherited, identity is partly
induced.** CKA is 0.93 between the two waveform CNNs and 0.60–0.69 for every other pair,
so the same pattern holds across genuinely different representations.

## 4. Digit recognition needs formants, not pitch (July 2026)

**Question.** Which frequency bands does digit recognition need, and do they coincide with
the strongest sex cue (pitch, 0–250 Hz)?

**Design.**
- **Band occlusion (necessity):** zero one of 16 bands of 250 Hz in the 8,192-point real
  DFT of the waveform and record the accuracy drop on all 6,000 SI test recordings. The
  band-stop is applied once to the shared waveform and the spectrogram model's input is
  recomputed from the occluded waveform, so all four architectures receive the identical
  intervention.
- **DFT-LRP (attribution):** LRP (zennit `EpsilonPlusFlat`, batch norm merged into the
  preceding convolution) mapped onto the real DFT by a virtual inspection layer, averaged
  over 1,000 test recordings, for the two waveform CNNs whose layers these rules cover.
- **Faithfulness gate first:** the attribution was only interpreted after a
  feature-flipping test (remove bins most-relevant-first, least-relevant-first and at
  random) showed that MoRF destroys the prediction fastest.

**Result.** Accuracy depends almost entirely on energy below about 1 kHz, with the largest
drops in the 250–750 Hz formant region in all four architectures. Above 1.5 kHz, removing
a band costs the 1-D CNN and the GRU model less than one point, the residual CNN at most
1.5 and the spectrogram CNN up to 3.4, against peaks of 4.6–25 points below 750 Hz.
DFT-LRP agrees: 88–90% of absolute relevance lies below 750 Hz, and attribution and
necessity profiles correlate at r = 0.62 (cnn1d) and 0.67 (resnet1d). The attribution is
faithful (cnn1d AUC: MoRF 0.15, random 0.41, LeRF 0.54; resnet1d 0.16, 0.39, 0.55). The
methods differ in one place: DFT-LRP gives the 0–250 Hz pitch band the largest single
share, yet removing it costs less than removing 250–750 Hz — the pitch band is weighted
but redundant. Necessity profiles split by speaker sex have similar shapes (r = 0.41–0.93),
diverging most around 625 Hz; with three women this is low-to-medium confidence.

## 5. No sex neuron, but a sex direction (July–August 2026)

**Question.** Is sex carried by identifiable units, or by a direction in the
representation — and is that direction specific to sex?

**Design and results, in sequence.**
- **Within-digit decodability.** Sex probes restricted to one digit at a time reach
  0.94–0.95 on average in the three convolutional models. The GRU model reaches 0.80, near
  the 0.75 majority rate, and its global penultimate sex probe does not clear chance; a
  pre-set gate therefore excludes it from all direction analyses (angles between
  directions fit on noise would look like results but mean nothing).
- **Units.** Ranking penultimate units by female–male Cohen's d and comparing across seeds:
  top-20 overlap is at chance and full rankings correlate at ρ ≈ 0. PCA explains why:
  94–98% of penultimate variance lies in about ten directions, so units are
  interchangeable coordinates in a near-rank-10 space. The analysis therefore moved from
  units to a direction.
- **The confounded first estimator.** A raw mean-difference direction (female mean minus
  male mean) reproduced across seeds, but a diagnostic showed that in a near-rank-10 space
  almost *any* 3-vs-9 grouping of the same 12 speakers reproduces equally well. Its
  apparent stability was a property of the estimator, not of sex.
- **The validated direction û.** A logistic probe fit on per-unit standardised penultimate
  activations of the 12 SI test speakers, mapped back to the original coordinates and
  normalised. To test sex-specificity, it was refit for **every** alternative 3-vs-9
  grouping of the same 12 speakers — all C(12,3) − 1 = 219 of them, an exact null rather
  than a Monte Carlo sample of a 220-point space — and cross-seed reproducibility was
  compared after orthogonal Procrustes alignment of the seeds' penultimate spaces.

**Result.** û's cross-seed reproducibility exceeds all 219 alternative re-splits in
cnn1d and spec_cnn2d, but falls inside the null for resnet1d and the GRU model. The later
direction tests use only the two validated models. This removes the metric confound but
not the sample one: the direction is sex-specific across these 12 speakers, and its
generality beyond three women cannot be tested.

## 6. The sex direction is not a pitch axis — a falsified prediction (August 2026)

**Question.** What acoustic property does û track?

**Design.** The prediction, written down before running the test, was that û would track
F0. F0 (pYIN, 65–300 Hz) and F1–F3 (Praat Burg, five formants, 3.8 kHz ceiling) were
extracted from the 8 kHz model input, and each recording's projection onto û was
correlated with them (Spearman). The decisive test was set in advance to be the
correlation **within the nine male speakers**, which does not depend on the three women;
F0–formant comparisons use only recordings where every extractor succeeded.

**Result.** Across sexes the projection correlates with F0 at ρ = 0.66 (cnn1d) and 0.64
(spec_cnn2d), but only because the sexes form two separate clusters. Within the nine men
the correlation is ρ = 0.19 and 0.13 (2–4% of variance), and within women 0.06–0.08.
**The pitch-axis prediction is falsified**: the direction separates the sexes
categorically, using more than pitch or formants. What it is remains open.

## 7. The digit decision does not use the sex direction (July–September 2026)

**Question.** Does the digit readout depend on the sex direction?

**Design.** Three independent measurements, each with its own null:
- **Geometry.** At the penultimate layer, the angle between the mean of ten per-digit
  sex-probe directions (fit on the 48 train+val speakers, in one shared standardised
  basis so that directions from different digits are comparable) and the subspace spanned
  by the ten digit class means. At every layer, the angle between that layer's sex-probe
  direction and the dominant direction of the per-recording correct-digit-logit
  gradients, against 300 random directions.
- **Sensitivity.** The classifier head is a three-layer MLP, so there is no single weight
  matrix to project onto. The readout's "weights" were therefore operationalised as the
  Jacobian of the real trained head's logits with respect to the penultimate vector,
  averaged over the test recordings, and its projection on û was expressed per digit as a
  percentile of 1,000 random unit directions.
- **Counterfactual.** Shift each correctly classified test recording's penultimate vector
  along û by δ times the female–male projection gap (11 values of δ in [−3, 3]) and
  record the correct-digit logit and accuracy.

**Result.** The mean per-digit sex direction lies 82.5° (cnn1d), 83.1° (resnet1d) and
83.5° (spec_cnn2d) from the digit class-mean subspace. At every layer of both validated
models and all three seeds, the sex-probe direction lies 79–90° from the digit-gradient
direction. For seed 42 the readout sensitivity along û is at or below the 35th percentile
of random directions for every digit in both models; across all 60 tests (3 seeds × 2
architectures × 10 digits) it is below the median in 58, reaches at most the 53rd
percentile and never exceeds the 95th, including for the confusable digits 0, 1, 7 and 9.
Shifting representations by up to three times the full male–female gap changes accuracy by
less than 0.002.

## 8. Yet a model that never heard women reads them worse (August–September 2026)

**Question.** Does removing women from training harm women specifically?

**Design.** For cnn1d and spec_cnn2d, a male-only model was trained (38 male training, 10
male validation speakers, same protocol, seeds 42/43/44) and compared with the SI
baseline on women, per speaker, with a paired speaker-level bootstrap.

- **Memorisation-confound fix.** The first comparison used all 12 women. But the SI
  baseline had *trained* on 7 of them and used 2 for validation, so on those 9 it carried a
  memorisation advantage the male-only model could not have. The headline comparison was
  therefore restricted to the three SI test women, whom neither model heard.
- **Same-size control.** New models were trained on 38 men (matching the male-only model's
  training-set size) and compared with the SI baseline on the nine SI test men, who are
  held out from both — so this comparison needs no confound fix and is well powered.
- **Leave-one-out.** Without retraining, the female gap was recomputed leaving out each of
  the three women in turn, to check that no single speaker drives it.

**Result.** Relative to the SI baseline, the male-only model loses 7.9 (cnn1d) and 5.0
(spec_cnn2d) points on the three held-out women. The same-size male control loses only 2.9
and 2.1 points on the nine held-out men, and the female gap stays negative when any one of
the three women is left out. The direction of the gap is stable across seeds; its size
rests on three women. Sex therefore matters somewhere — just not along the direction
measured above.

## 9. The gap is repaired by late-layer recalibration (September 2026)

**Question.** If the male-only model is fine-tuned on women, where in the network does the
repair happen?

**Design.**
- The 12 women form four folds of three; each fold fine-tunes the pinned male-only
  checkpoint on nine women and evaluates on the other three, so every woman is evaluated
  once. The male-only checkpoints were pinned by state-dict hash after confirming that
  repeated training runs were bit-identical.
- Fine-tuning updates all weights (Adam, lr 1e-4, batch 128) for a **fixed** two epochs,
  without early stopping, so that conditions are directly comparable. Two epochs were
  chosen by a pilot on one fold: a third epoch added nothing (+0.149 vs. +0.148).
- A male control applies the same fine-tuning to the male-only model's ten validation
  speakers (seen only for checkpoint selection), split 7–8 / 2–3. It is confounded (weak
  prior exposure, near-ceiling accuracy), so it is read as context, never subtracted as a
  specificity measure.
- Localisation: (a) the relative L2 change of each layer's activations on the evaluation
  recordings, before vs. after fine-tuning; (b) for cnn1d, DFT-LRP band relevance before
  vs. after fine-tuning, interpreted only for runs that pass the faithfulness gate before
  and after (24 of 24 did).

**Result.** Female accuracy rises from 0.87 to 0.96 (+9 points, cnn1d) and from 0.86 to
0.96 (+11, spec_cnn2d). Repeating the cnn1d fine-tuning in a second (GPU) environment
reproduced post-fine-tuning accuracy within 0.7 points. The gain is positive in all four
folds of both models (small in one cnn1d fold), while the male control changes by close to
zero. Activation change concentrates in the late, decision-adjacent layers and barely
touches the acoustic front-end. DFT-LRP confirms this independently for cnn1d: on female
speech, reliance falls by a similar proportion in every 250 Hz band (about −15% to −27%),
with no band singled out, while the male control stays near zero. Fine-tuning does not
make the model listen to different frequencies; it recalibrates how already-extracted
evidence is combined into a digit.

---

## Limits carried through every experiment

- **Three women.** Every sex-related result is a within-sample claim about the three SI
  test women (or, for fine-tuning, folds of three), not a population estimate.
- **No clean control for adaptation.** AudioMNIST has no unseen men comparable to the
  unseen women, so female-specific adaptation cannot be separated from generic
  new-speaker adaptation.
- **Linear read-outs.** Non-linear use of sex is not excluded.
- **Partial architecture coverage.** The sex direction is validated in two of four
  architectures; the GRU model's penultimate sex probe is at chance.
