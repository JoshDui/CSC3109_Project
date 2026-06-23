# External Training Data and Reliability Evaluation (Custom CNN)

*Draft section — custom CNN track (`bridge`, `freeway`, `overpass`, `railway`).*
*All figures and tables referenced here are generated artifacts under `reports/reliability/` and `reports/tables/`.*

## 1. Motivation: the metric-saturation problem

During early experimentation, every model we trained on the assigned PatternNet
partition reached a macro-F1 of approximately 1.0 within the first training
epoch. This was true across architectures, from the custom CNN to pretrained
transfer-learning backbones.

Immediate metric saturation undermines the evaluation rather than indicating
success:

- It renders models **indistinguishable**. When multiple architectures all score
  ~1.0, the headline metric provides no basis for comparing designs.
- It obscures whether the model is **learning** the task or merely exploiting an
  easy evaluation set.
- It precludes any meaningful claim about **reliability** — robustness,
  calibration, and generalisation — because no measurable variation exists to
  analyse.

Two underlying causes were identified: a flawed evaluation protocol, and the
intrinsic separability of the four PatternNet classes — which are clean,
homogeneous, and easily distinguished. The first was addressed directly; a
second dataset was introduced to address the second. Both are described below.

## 2. Evaluation protocol correction

The original experiment split was produced by `create_split_manifest`
(`reports/tables/split_manifest.csv`). Inspecting that manifest shows the source
of the inflated metrics:

- Every row is drawn from the PatternNet **training** folder only
  (`data/set 12/`, 700 images per class, 2,800 total).
- The script applied a random stratified 80/20 split to that single pool,
  producing `train` (2,240 images) and `val` (560 images).
- The sample paths confirm the mechanism: for the `bridge` class,
  `bridge001`, `bridge002`, `bridge004` were assigned to training while
  `bridge003`, `bridge006`, `bridge009` were assigned to "validation" — the same
  folder and the same capture distribution, separated only by random index.

The consequence is an **evaluation-set-selection error**. The reported
"validation" metrics measured performance on a random in-distribution slice of
the training data rather than on the official held-out validation set
(`data/raw/val`, 100 images per class, image indices 701–800). This is not literal image-duplication leakage — the indices differ — but it
answers the wrong question: whether the model can fit images from the training
distribution rather than whether it generalises to unseen data. Combined with
the high class separability, this explains why F1 reached ~1.0 at epoch 1.

**Correction.** The official PatternNet validation set (400 images, 100 per
class) is now reserved as a fixed `holdout` split that is never seen during
training or model selection. It is the single source of in-domain evaluation
metrics for all subsequent results.

## 3. Why a second dataset, and why NWPU-RESISC45

Correcting the protocol removes the measurement error, but it does not alter the
fact that the in-domain task is straightforward: even with a clean held-out set,
a well-trained model attains near-perfect scores and leaves negligible headroom
to study *reliability*. To create that headroom a **second, independent dataset**
covering the same four classes was introduced.

**NWPU-RESISC45** [1] is a widely used aerial-scene classification benchmark. It
was obtained from the `blanchon/RESISC45` repository on the Hugging Face Hub
[2]. NWPU-RESISC45 is suitable because:

- It contains the **same four target classes** (`bridge`, `freeway`,
  `overpass`, `railway`), so labels map directly with no re-annotation.
- It is drawn from a **different source and distribution** than PatternNet —
  distinct sensors, resolutions, geographies, and scene compositions. This
  distribution shift provides the conditions needed to test generalisation.
- It is large and balanced enough to supply both additional training diversity
  and a reserved out-of-distribution (OOD) test slice.

The dataset is used for **two distinct purposes**:

1. **Training augmentation** — additional, more varied examples added to the
   training pool to move the model past trivial memorisation and produce a
   genuine learning curve.
2. **Out-of-distribution evaluation** — a held-aside NWPU slice, never trained
   on, serving as a harder cross-source test that the saturated in-domain
   metric cannot supply.

Critically, NWPU is **never used for in-domain reporting**. The official
PatternNet validation set remains the sole in-domain evaluation set, so results
remain comparable to the rest of the project and to the assignment
specification.

## 4. Data handling and leakage controls

To keep the experiment valid, the external data was filtered and partitioned
under explicit controls. The full per-class accounting is recorded in
`reports/tables/nwpu_dedup_report.json`.

- **Source layout.** NWPU images were downloaded to `data/external/nwpu/<class>/`
  (700 images per class, git-ignored).
- **Leakage guard.** Every NWPU image was compared against the official
  PatternNet validation set using a DCT perceptual hash. Near-duplicates
  (Hamming distance ≤ 5) would be dropped to prevent test-set contamination.
  At the chosen threshold, **0 duplicates** were found across all four classes —
  an expected outcome, given that the two datasets are drawn from independent
  sources [1, 2] — and **0** images were dropped for quality-assurance reasons.
- **Per-class split.** Of the 700 kept NWPU images per class, **600** go to the
  training pool and **100** are reserved as the `nwpu_ood` evaluation slice.

The resulting combined manifest
(`reports/tables/combined_experiment_manifest.csv`, built by
`python -m src.data.build_combined_manifest`) defines four non-overlapping
splits:

| Split | Source | Images/class | Total | Role |
| --- | --- | ---: | ---: | --- |
| `train` | PatternNet train + NWPU train | 1,040 | 4,160 | Model fitting |
| `tune` | PatternNet train + NWPU train | 260 | 1,040 | Model selection / early stopping |
| `holdout` | **Official PatternNet val** | 100 | 400 | **In-domain evaluation** |
| `nwpu_ood` | **NWPU (reserved)** | 100 | 400 | **Out-of-distribution evaluation** |

The `train` and `tune` splits are an 80/20 partition of the pooled
PatternNet-train + NWPU-train images (700 + 600 = 1,300 per class). The two
evaluation splits are disjoint from training and from each other, balanced at
100 images per class, with verified zero cross-split leakage.

## 5. Experimental setup

The custom CNN (`src/models/custom_cnn.py`, an 8-convolution network with
batch-norm, GELU activations, global average pooling, and a dropout-regularised
head) was trained **from scratch** on the combined training pool. Key settings
(logged as EXP-003 in `experiments/results-log.md`):

- Input size 224×224; on-the-fly augmentation (random resized crop, horizontal
  and vertical flips, rotation, colour jitter).
- Optimiser AdamW, learning rate 3e-4 with cosine decay, batch size 128.
- 60 epochs; model selection on the `tune` split; final metrics reported on the
  untouched `holdout` and `nwpu_ood` splits.

## 6. Results

### 6.1 Genuine learning curve

With the harder, more diverse training pool, the model no longer saturates
immediately. Tune-set macro-F1 rises from **0.53 at epoch 1 to 0.96 by the end
of training**, and final train accuracy (0.970) lies within roughly one point of
tune accuracy (0.963). This gradual improvement and the narrow train–tune gap
constitute direct evidence of genuine learning without overfitting or
leakage.

*Figure: `reports/reliability/learning_curves.png`.*

### 6.2 In-domain vs out-of-distribution

The corrected in-domain metric is strong, but the cross-source OOD test reveals
a substantial, honest generalisation gap.

| Metric | Holdout (PatternNet, in-domain) | NWPU-OOD (cross-source) |
| --- | ---: | ---: |
| Accuracy | 0.9925 | 0.8775 |
| Macro-F1 | 0.9925 | 0.8786 |
| ECE (calibration error) | 0.0468 | 0.0537 |
| Mean confidence | 0.946 | 0.836 |

The macro-F1 generalisation gap is **0.114** (approximately 11 percentage
points). The model performs near-perfectly on data drawn from the training
distribution, but loses roughly one eighth of its F1 when presented with the
same classes captured by a different source.

*Figures: `reports/reliability/ood_confusion_matrix.png`,
`reports/reliability/calibration.png`.*

### 6.3 Corruption robustness

We also measured robustness by degrading the in-domain holdout with increasing
severities of five common corruptions (`reports/reliability/robustness.json`,
plotted in `reports/reliability/robustness.png`). Macro-F1 by severity level:

| Corruption | Sev 0 (clean) | Sev 1 | Sev 2 | Sev 3 |
| --- | ---: | ---: | ---: | ---: |
| Gaussian blur | 0.993 | 0.670 | 0.328 | 0.120 |
| Gaussian noise | 0.993 | 0.733 | 0.359 | 0.237 |
| JPEG compression | 0.993 | 0.993 | 0.985 | 0.938 |
| Brightness shift | 0.993 | 0.899 | 0.993 | 0.304 |
| Rotation | 0.993 | 0.985 | 0.925 | 0.810 |

The model is robust to JPEG compression and moderate rotation, but degrades
sharply under blur, noise, and severe darkening. These are concrete,
interpretable failure modes rather than an opaque "100% accurate" result.

Note on severity scales: for blur, noise, and rotation, higher levels are
strictly more severe. For JPEG the levels are decreasing quality factors
(100 → 50 → 30 → 15). For brightness the levels are multiplicative factors
(1.0, 0.7, 1.3, 0.5) and are therefore not monotonic — the model tolerates a
brighter image (1.3×, macro-F1 0.993) far better than a heavily darkened one
(0.5×, macro-F1 0.304). See `reports/reliability/robustness.json` for the exact
per-level parameters.

### 6.4 Calibration

Expected Calibration Error is low in-domain (ECE = 0.047) and modestly higher
out-of-distribution (ECE = 0.054), with mean confidence dropping from 0.95 to
0.84 on OOD data. The model is mildly overconfident but does exhibit reduced
confidence on harder, unfamiliar inputs — appropriate directional behaviour,
even though confidence does not fully track the accuracy drop.

## 7. Interpretation

The combined evidence reframes what "reliable" means for this task:

- The model **genuinely learns** the classification task — demonstrated by the
  gradual learning curve and the narrow train–tune gap — rather than memorising
  an easy evaluation set.
- In-domain accuracy of 99.25% is **real but incomplete**. Because the official
  PatternNet set is clean and separable, headline accuracy alone overstates how
  dependably the model would perform under field conditions.
- The **11-point OOD drop**, the **corruption-robustness curves**, and the
  **calibration error** together expose the model's actual operating limits.

The inclusion of NWPU-RESISC45 [1] is therefore central to the evaluation, not a
side experiment: it is what converts an uninformative ~1.0 score into a
defensible, multi-faceted reliability profile.

## 8. Limitations and next steps

- **Single external source.** OOD behaviour is characterised against one
  additional dataset (NWPU-RESISC45 [1]). A second independent source would
  strengthen the generalisation claim.
- **Custom CNN only.** This protocol has been applied only to the from-scratch
  custom CNN. The same combined manifest and reliability suite
  (`src/evaluation/evaluate_reliability.py`) should be run for the other models
  to produce an apples-to-apples reliability comparison.
- **Mild overconfidence.** ECE is low but non-zero; temperature scaling on the
  tune split is a low-cost follow-up if better-calibrated probabilities are
  required for deployment.

## References

[1] G. Cheng, J. Han, and X. Lu, "Remote Sensing Image Scene Classification:
Benchmark and State of the Art," *Proceedings of the IEEE*, vol. 105, no. 10,
pp. 1865–1883, 2017. doi: 10.1109/JPROC.2017.2675998.

[2] Hugging Face dataset repository: blanchon/RESISC45, Hugging Face Hub.
URL: https://huggingface.co/datasets/blanchon/RESISC45

## Artifact index

| Artifact | Path |
| --- | --- |
| Combined manifest | `reports/tables/combined_experiment_manifest.csv` |
| NWPU dedup / split report | `reports/tables/nwpu_dedup_report.json` |
| Old (buggy) split manifest | `reports/tables/split_manifest.csv` |
| Reliability summary | `reports/reliability/reliability_summary.json` |
| Learning curves | `reports/reliability/learning_curves.png` |
| Robustness curves / data | `reports/reliability/robustness.png`, `robustness.json` |
| Calibration | `reports/reliability/calibration.png` |
| OOD confusion matrix / metrics | `reports/reliability/ood_confusion_matrix.png`, `nwpu_ood_metrics.json` |
| Run log entry | `experiments/results-log.md` (EXP-003) |
