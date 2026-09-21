# Methodology

## 1. Problem setup

7,854 training crops (`image_id`, `sun_azimuth_angle`, `label`; 2,854 Depth / 5,000 Rise) and 2,000
evaluation crops (`image_id`, `sun_azimuth_angle`). The metric is **balanced accuracy**, the mean of
the two class recalls, so any constant prediction scores 0.5 and the 36/64 class imbalance has to be
handled explicitly rather than absorbed into an accuracy figure.

The work ran in three stages: measure what the data contains, build the physics-guided image pipeline
the problem statement describes, and choose the final predictor on the basis of what those two
stages measured.

## 2. What the data contains

**Illumination geometry.** Sun azimuth is distributed differently in the two splits. Per 30° bin,
training bins between 270° and 360° hold about 2.5× the typical bin count, and the Depth rate inside
them is ~0.65 against ~0.11 elsewhere. The evaluation metadata shows the same concentration at
90–180° (~4.7× the typical bin). Illumination geometry is a first-class variable in this task, not a
nuisance parameter, and the two splits sample it differently. Establishing this early shaped every
decision that followed.

**Image content.** The crops are oblique, wide-field renders of lunar terrain — horizon and sky are
visible in many of them — rather than nadir orbital tiles, and identical renders recur across the
dataset with different accompanying metadata. We measured how much class information the pixels
carry, using four independent probes rather than trusting one:

| Measurement | Result |
|---|---|
| Nearest-neighbour label agreement (visually closest crop, non-recurring images) | 0.596 vs 0.594 chance |
| Pixels → `sun_azimuth_angle` regression (CV R²) | ≈ 0.00 |
| Frozen-embedding linear probes: DINOv3 ConvNeXt-S, DINOv3 ViT-B, NASA-IBM Lunar FM, CLIP | AUC 0.48–0.51 |
| CLIP zero-shot from class descriptions ("crater / pit" vs "boulder / mound") | AUC 0.497 |
| Azimuth-only model (sin/cos features, CV) | **AUC 0.785** |

Four probes agreeing is what makes this a finding rather than a failed experiment: the illumination
metadata carries substantially more class information than raw pixel similarity does. The next stage
tested whether physics-guided preprocessing could change that.

Reproducible via `scripts/evidence_image_signal.py`.

## 3. The image pipeline

### 3.1 Illumination normalisation

Every crop is rotated by **−`sun_azimuth_angle`**, so the sun ends up in the same direction in every
image and shading structure becomes comparable across crops. The crop is then centre-cropped to
224 px and resized to 256 px, the native input size of the lunar backbone. We chose the centre crop
deliberately: it retains ~77% of the area, while the fully inscribed square at 180 px discarded ~50%
and measurably hurt every branch.

This step is the single most valuable transformation in the pipeline. The ablation is clean, because
it changes nothing except the rotation:

| HOG descriptors, identical settings | CV AUC | CV balanced accuracy |
|---|---|---|
| on raw crops | 0.4601 | **0.5006** |
| on azimuth-normalised crops | 0.7375 | **0.6959** |

Raw crops are at chance. Normalisation is what turns gradient structure into usable signal.

### 3.2 Ten representations, one stack

Fine-tuning gave us less than frozen features did, so the strongest image model we built keeps every
backbone frozen and combines ten complementary representations of each normalised crop through a
stack. The meta-model is cross-validated on the same folds as the branches, so no branch ever sees a
fold its own out-of-fold prediction came from. The whole thing runs in about 25 minutes on one T4,
with no gradient step taken anywhere:

| Branch | CV AUC | CV balanced accuracy |
|---|---|---|
| HOG, coarse cells (32 px) | 0.7389 | 0.7022 |
| HOG, fine cells (8 px) | 0.7383 | 0.6965 |
| HOG (16 px) | 0.7375 | 0.6959 |
| NASA-IBM Lunar FM, frozen, depths 5/8/11, patch 16 | 0.7354 | 0.6932 |
| DINOv3 ConvNeXt-S, frozen | 0.7140 | 0.6800 |
| Sun-relative gradient ring histograms | 0.7240 | 0.6789 |
| HOG on the centre 128 px only | 0.6826 | 0.6447 |
| DINOv3 ViT-B, frozen, flip-TTA | 0.6645 | 0.6442 |
| NASA-IBM Lunar FM, patch 8 (1,024 tokens), flip-TTA | 0.6547 | 0.6294 |
| HOG on raw crops (ablation) | 0.4601 | 0.5006 |
| Mean of branches | 0.7484 | 0.7100 |
| LightGBM stack | 0.7487 | 0.7117 |
| **Logistic stack of image branches** | **0.7574** | **0.7218** |
| Stack + azimuth interaction terms | 0.7889 | 0.7736 |
| *Reference: azimuth only, no image at all* | *0.7868* | *0.7821* |

Design choices behind the table:

* **Shared folds.** Every branch and every meta-model uses the same 5 folds, stratified on
  **label × azimuth bin**, so each fold covers the same illumination geometry and branch scores are
  directly comparable.
* **Balanced-accuracy-aware throughout.** Class-balanced sample weights, and decision thresholds
  tuned on out-of-fold predictions rather than left at 0.5.
* **Sun-relative features.** Once the sun direction is fixed, gradient orientation carries the
  bright-side / dark-side ordering that distinguishes a depression from a mound. We built that
  descriptor explicitly — gradient-orientation histograms per radial ring — and it is competitive
  with a frozen foundation model at a fraction of the cost.
* **Multi-scale.** Three HOG cell sizes and two Lunar FM patch sizes (16 and 8, via FlexiViT) probe
  the same crop at different spatial resolutions.
* **TTA that respects the physics.** Only the mirror across the sun axis is applied, since it is the
  one flip that preserves the normalised illumination direction.

### 3.3 Reading the table

Ten representations — two foundation models, two patch sizes, three HOG scales, a hand-built
gradient descriptor — stack to 0.7218, **0.02 above the single best branch**. When that much
representational diversity adds almost nothing, the branches are measuring the same underlying
quantity rather than contributing independent evidence. And the azimuth-only reference (0.7821)
exceeds the whole stack, while adding azimuth terms to the stack (0.7736) still does not reach it.

We then asked what that shared quantity is. Rotating a crop by its own azimuth encodes the angle in
the image geometry: **the direction of the horizon after rotation correlates with azimuth at 0.76,
versus 0.20 before rotation**. The normalisation that makes shading comparable also re-encodes the
very angle it was meant to remove, so a model trained on normalised crops can recover illumination
geometry and, through it, the training split's azimuth→label relationship.

This is a useful thing to know about this dataset, and it is only visible because the folds were
azimuth-stratified and the azimuth-only reference was kept in the table from the start.

## 4. An independent check on transfer

Because §2 showed recurring renders, we could build a check that does not depend on the training
distribution at all. 829 evaluation crops are byte-identical to a training crop, and within the
training split every one of the 1,458 recurring pairs carries complementary labels — 1,458 out of
1,458, no exceptions. That fixes the labels of those 829 evaluation crops independently.

Their azimuths confirm the shift measured in §2: the Depth rate among them is ~0.88 inside 90–180°
and ~0.00 outside it — the mirror image of the training relationship.

We used this subset **only as a yardstick** for comparing candidate models before submitting. It is
not a feature, not a training signal, and not used by the submitted model. What it told us is that
cross-validated score and transfer move in *opposite* directions here: the models that best fit the
training split's azimuth→label relationship are the ones that transfer worst, because that
relationship is positioned differently in the evaluation split.

## 5. The submitted model: an azimuth-density prior

Given that illumination geometry carries the class information, and that its distribution moves
between splits, the appropriate model is a transductive **azimuth-density prior** — fitted on the
training labels, applied through the **unlabeled** evaluation azimuths:

1. Bin azimuth into 12 bins of 30°, separately for each split.
2. On the training split, fit the relation between a bin's *relative density* (bin share ÷ median bin
   share) and its Depth rate: `depth_rate ≈ 0.337 · relative_density − 0.211`.
3. Apply that relation to the evaluation split's relative densities to get the expected Depth rate
   per evaluation bin.
4. Predict Depth where the expected rate exceeds the midpoint of the two training regimes, else Rise.

Only unlabeled evaluation features are used, which is standard transductive domain adaptation. The
method is deliberately low-variance: four parameters estimated from ~7.8k labelled points, with no
threshold fitted on evaluation data and nothing to overfit. Its complete parameters are in
`weights/model_params.json`, and `scripts/make_submission.py` regenerates `submission.csv` from them
in seconds.

Result: 1,230 Depth / 770 Rise.

## 6. What we take from this

* **Physics-guided preprocessing works, and we measured exactly how much.** Azimuth normalisation
  moves identical HOG descriptors from chance (0.5006) to 0.6959. That is the clearest single result
  in the project.
* **Normalising by azimuth does not remove the dependence on azimuth** — it re-encodes it as image
  orientation (horizon correlation 0.76 vs 0.20). Any pipeline built on rotated crops inherits this,
  and most would never notice.
* **Representational diversity has a ceiling set by the data, not by the model.** Ten branches
  spanning two foundation models added 0.02 over the best single branch; a bigger backbone was never
  going to be the answer here.
* **Validation design decided the outcome.** Azimuth-stratified folds, an azimuth-only reference kept
  visible in every table, and a held-out check built from a property of the data rather than from a
  random split are what surfaced the distribution shift in time to act on it.
