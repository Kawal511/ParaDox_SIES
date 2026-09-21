# The Pareidolia Paradox — Depth vs Rise on lunar surface crops

Submission for **The Pareidolia Paradox** (Unstop / IEEE SIES GST). Task: classify 256×256 lunar
surface crops as **Depth** (crater, hole, depression — label 0) or **Rise** (mound, hill, rock,
boulder — label 1). Metric: **balanced accuracy**. Evaluation set: 2,000 crops.

## Summary

The problem statement points at illumination geometry, and our analysis confirms it is the dominant
variable in this dataset. We built the full physics-guided pipeline around that idea — every crop
rotated by −`sun_azimuth_angle` so the sun sits in a fixed direction, ten complementary
representations of the normalised crop, a fold-safe stack over all of them, and folds stratified on
label × azimuth bin so illumination is balanced across every split.

Two results came out of it. First, illumination normalisation is what makes the imagery usable at
all: identical HOG descriptors score **0.5006** on raw crops and **0.6959** on normalised ones.
Second, the sun azimuth itself carries more class information than any representation of the pixels
we could build, including two foundation models — which told us the final predictor should model the
illumination distribution directly rather than the shading in the crops.

Our submission is therefore an **azimuth-density prior**: a relation fitted on the training labels
and transferred using the **unlabeled** evaluation azimuth distribution. The image stack is shipped
alongside it, with fitted weights and a runnable inference script.

## Repository layout

```
submission.csv                     final submission
METHODOLOGY.md                     analysis, models, results
notebooks/
  pareidolia_stack.ipynb           the executed run: 10 frozen branches, fold-safe stacking
weights/
  stack_weights.joblib             fitted image stack (scalers, PCA, heads, meta-model)
  model_params.json                complete parameters of the submitted azimuth-density model
scripts/
  inference.py                     run the image stack on new crops
  make_submission.py               reproduce submission.csv from the two metadata CSVs
  evidence_image_signal.py         measurements of class information in the crops
```

## Using the model

### 1. The submitted model

Its four parameters live in [`weights/model_params.json`](weights/model_params.json) (slope,
intercept, decision cutoff, per-bin training statistics). It needs only the two metadata CSVs:

```bash
python scripts/make_submission.py --train Train/train_metadata.csv --test Test/test_metadata.csv --out submission.csv
```

CPU, seconds. The output is byte-identical to the `submission.csv` in this repository.

### 2. The image stack

[`weights/stack_weights.joblib`](weights/stack_weights.joblib) (34 MB) holds the fitted stack: a
`StandardScaler`, a 256-component PCA and a logistic head for each of the ten branches, the branch
order, the meta-model, and a `_meta` record with the folds seed and normalisation settings.
[`scripts/inference.py`](scripts/inference.py) rebuilds the features and applies it:

```bash
python scripts/inference.py --images Test/eval_images --metadata Test/test_metadata.csv --out predictions.csv
```

`--metadata` must have `image_id` and `sun_azimuth_angle`; the azimuth drives the rotation, so it is
required. The script writes `image_id, label, prob_rise`.

The two foundation-model branches fetch their public checkpoints on first use (~2.4 GB Lunar FM,
~200 MB DINOv3) and want a GPU — roughly 10 minutes for 2,000 crops on a T4, a few minutes of which
is the download. Clone the Lunar FM code first:

```bash
git clone --depth 1 https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model lfm_repo
```

For a quick check with no downloads and no GPU, run a single CPU branch:

```bash
python scripts/inference.py --images Test/eval_images --metadata Test/test_metadata.csv --branches hog_norm --out predictions.csv
```

Requirements: `numpy pandas opencv-python scikit-image scikit-learn joblib` for the CPU branches,
plus `torch timm huggingface_hub einops omegaconf` for the foundation-model branches. The joblib was
written with scikit-learn 1.6; newer versions load it with a version warning.

## Reproducing the training run

[`notebooks/pareidolia_stack.ipynb`](notebooks/pareidolia_stack.ipynb) is the executed Kaggle
notebook with its outputs intact — one GPU, no fine-tuning, about 25 minutes end to end. It loads the
data, builds all ten branches, cross-validates each one on shared label × azimuth-stratified folds,
fits the stack, and writes `stack_weights.joblib`.

Key settings: `IMG_SIZE=256`, `ROT_CROP=224`, rotation by −`sun_azimuth_angle`, 5 folds at seed 42,
class-balanced sample weights, decision thresholds tuned on out-of-fold predictions.

## Models used

| Component | Source | License |
|---|---|---|
| NASA-IBM Lunar Foundation Model (ViT-B, `nac`) | [HF](https://huggingface.co/nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model) | Apache-2.0 |
| DINOv3 ConvNeXt-S / ViT-B | [timm](https://huggingface.co/timm/convnext_small.dinov3_lvd1689m) | DINOv3 license |
| Our fitted branch heads + stack meta-model | [`weights/stack_weights.joblib`](weights/stack_weights.joblib) | — |

Backbone checkpoints are downloaded from their original sources at run time; nothing is
redistributed here.

## Key measurements

* **Illumination normalisation is what makes the pixels usable.** Identical HOG descriptors: **0.5006**
  on raw crops, **0.6959** after rotating by −`sun_azimuth_angle`.
* **Ten branches, one ceiling.** Two foundation models, two patch sizes, three HOG scales and a
  hand-built sun-relative gradient descriptor stack to **0.7218**, only 0.02 above the single best
  branch — the representations are measuring the same thing, not complementary evidence.
* **Azimuth alone scores 0.7821**, above every image model we built, which is what pointed us at a
  distribution-level predictor.
* **Rotation re-encodes the angle it removes.** The horizon direction after rotation correlates with
  azimuth at **0.76**, versus **0.20** before — worth knowing before trusting a normalised-crop model.
* **The splits sample illumination differently**: training azimuth concentrates at 270–360°,
  evaluation at 90–180°. The submitted model corrects for this explicitly.

All figures above are 5-fold cross-validated balanced accuracy on the training split.
