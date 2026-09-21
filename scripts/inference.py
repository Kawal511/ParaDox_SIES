"""Run the multi-branch image stack on a folder of lunar crops.

Usage
-----
    python scripts/inference.py --images path/to/images \
                                --metadata path/to/metadata.csv \
                                --weights weights/stack_weights.joblib \
                                --out predictions.csv

--metadata needs the columns `image_id` and `sun_azimuth_angle`; the azimuth drives the
illumination normalisation, so it is required. Everything else the model needs is inside the
joblib: a scaler, PCA and logistic head per branch, plus the stack meta-model.

The two foundation-model branches download their public checkpoints on first use (~2.4 GB for
the Lunar FM, ~200 MB for DINOv3) and want a GPU. For a quick end-to-end check without any
download, run a single CPU branch: --branches hog_norm
"""
import argparse
import gc
import importlib.util
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import cv2
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

IMG_SIZE, ROT_CROP = 256, 224
# grayscale mean/std of the azimuth-normalised training crops. The Lunar FM branches were
# fitted against these exact values, so they belong to the model, not to the input batch.
LFM_MEAN, LFM_STD = 0.3988459706, 0.2367289364
# decision threshold of the image-only stack, tuned on out-of-fold predictions
DEFAULT_THRESHOLD = 0.52

LFM_REPO_URL = "https://github.com/NASA-IMPACT/NASA-IBM-Lunar-Foundation-Model"
LFM_HF_REPO = "nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model"


# ------------------------------------------------------------------ image pipeline
def load_images(folder, names, workers=8):
    def read(n):
        im = cv2.imread(os.path.join(folder, n), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(os.path.join(folder, n))
        return im if im.shape == (256, 256) else cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA)
    with ThreadPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(read, names)))


def normalise(imgs, azs, out_size=IMG_SIZE):
    """Rotate each crop by -sun_azimuth_angle, centre-crop, resize.

    The sun then sits in a fixed direction for every crop, which is what makes gradient
    structure comparable across images taken under different illumination.
    """
    def f(i):
        M = cv2.getRotationMatrix2D((128, 128), float(-azs[i]), 1.0)
        r = cv2.warpAffine(imgs[i], M, (256, 256), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
        s = (256 - ROT_CROP) // 2
        return cv2.resize(r[s:s + ROT_CROP, s:s + ROT_CROP], (out_size, out_size),
                          interpolation=cv2.INTER_LINEAR)
    with ThreadPoolExecutor(8) as ex:
        return np.stack(list(ex.map(f, range(len(imgs)))))


# ------------------------------------------------------------------ branch features
def hog_features(arr, ppc=16, size=224):
    from skimage.feature import hog

    def f(a):
        return hog(cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA), orientations=9,
                   pixels_per_cell=(ppc, ppc), cells_per_block=(2, 2), block_norm="L2-Hys",
                   feature_vector=True)
    with ThreadPoolExecutor(8) as ex:
        return np.stack(list(ex.map(f, arr))).astype(np.float32)


def sun_relative(arr, n_rings=4, n_bins=12):
    """Gradient-orientation histograms per radial ring.

    After normalisation the sun direction is fixed, so the orientation bins carry the
    bright-side / dark-side ordering that separates a depression from a mound.
    """
    H = arr.shape[1]
    yy, xx = np.mgrid[0:H, 0:H] - H / 2
    rad = np.sqrt(yy ** 2 + xx ** 2) / (H / 2)
    ring = np.clip((rad * n_rings).astype(int), 0, n_rings - 1)

    def f(a):
        a = a.astype(np.float32)
        gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        ang = (np.arctan2(gy, gx) + 2 * np.pi) % (2 * np.pi)
        b = np.clip((ang / (2 * np.pi) * n_bins).astype(int), 0, n_bins - 1)
        out = []
        for r in range(n_rings):
            m = ring == r
            h = np.bincount(b[m], weights=mag[m], minlength=n_bins)
            out.append(h / (h.sum() + 1e-6))
            out.append([a[m].mean() / 255, a[m].std() / 255, gy[m].mean(), gx[m].mean()])
        return np.concatenate([np.asarray(o).ravel() for o in out]).astype(np.float32)
    with ThreadPoolExecutor(8) as ex:
        return np.stack(list(ex.map(f, arr)))


# ------------------------------------------------------------------ frozen backbones
def _device():
    import torch
    return torch, ("cuda" if torch.cuda.is_available() else "cpu")


def load_lunar_fm(patch_size, repo_dir, weights_dir, device):
    path = os.path.join(repo_dir, "terratorch_integration", "lunar_backbone.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Lunar FM code not found at %s\n  git clone --depth 1 %s %s"
            % (path, LFM_REPO_URL, repo_dir))
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    spec = importlib.util.spec_from_file_location("lunar_backbone", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from huggingface_hub import hf_hub_download
    cfg = hf_hub_download(LFM_HF_REPO, "backbone/config.yaml", local_dir=weights_dir)
    ckpt = hf_hub_download(LFM_HF_REPO, "backbone/checkpoint.pt", local_dir=weights_dir)
    bb = mod.LunarBackbone(variant="base", modalities=["nac"], cfg=cfg,
                           patch_size=patch_size, checkpoint_path=ckpt)
    for n in ("decoder", "decoder_norm", "decoder_proj_context", "decoder_embeddings"):
        if hasattr(bb.model, n):
            delattr(bb.model, n)      # encoder only; the generative decoder is dead weight here
    return bb.to(device).eval()


def embed(model, imgs, kind, mean, std, device, bs=64):
    import torch
    import torch.nn.functional as F
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            x = torch.from_numpy(imgs[i:i + bs]).to(device)[:, None].float().div_(255.0)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
                if kind == "lfm":
                    outs = model((x - mean) / std)
                    f = torch.cat([F.layer_norm(outs[k].mean(1), (outs[k].shape[-1],))
                                   for k in (5, 8, 11)], 1)
                else:
                    f = model((x.expand(-1, 3, -1, -1) - mean) / std)
            out.append(f.float().cpu())
    return torch.cat(out).numpy()


def embed_tta(model, imgs, kind, mean, std, device, bs=32):
    # mirroring across the sun axis is the one flip that preserves the normalised geometry
    a = embed(model, imgs, kind, mean, std, device, bs)
    b = embed(model, imgs[:, :, ::-1].copy(), kind, mean, std, device, bs)
    return 0.5 * (a + b)


def lfm_branch(nrm, patch_size, tta, args):
    torch, device = _device()
    m = load_lunar_fm(patch_size, args.lfm_repo, args.lfm_weights, device)
    mean = torch.tensor(LFM_MEAN, device=device).view(1, 1, 1, 1)
    std = torch.tensor(LFM_STD, device=device).view(1, 1, 1, 1)
    fn = embed_tta if tta else embed
    f = fn(m, nrm, "lfm", mean, std, device, args.batch_size)
    del m
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return f


def timm_branch(nrm, name, tta, args, **create):
    import timm
    torch, device = _device()
    m = timm.create_model(name, pretrained=True, num_classes=0, **create).to(device).eval()
    dc = timm.data.resolve_model_data_config(m)
    mean = torch.tensor(dc["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(dc["std"], device=device).view(1, 3, 1, 1)
    fn = embed_tta if tta else embed
    f = fn(m, nrm, "timm", mean, std, device, args.batch_size)
    del m
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return f


# raw crops, azimuth-normalised crops and the parsed args in; feature matrix out
BRANCHES = {
    "hog_norm":   lambda raw, nrm, a: hog_features(nrm),
    "hog_raw":    lambda raw, nrm, a: hog_features(raw),
    "hog_fine":   lambda raw, nrm, a: hog_features(nrm, ppc=8, size=128),
    "hog_coarse": lambda raw, nrm, a: hog_features(nrm, ppc=32, size=224),
    "hog_centre": lambda raw, nrm, a: hog_features(nrm[:, 64:192, 64:192]),
    "sunrel":     lambda raw, nrm, a: sun_relative(nrm),
    "lfm":        lambda raw, nrm, a: lfm_branch(nrm, 16, False, a),
    "lfm_ps8":    lambda raw, nrm, a: lfm_branch(nrm, 8, True, a),
    "dino":       lambda raw, nrm, a: timm_branch(nrm, "convnext_small.dinov3_lvd1689m", False, a),
    "dino_vit":   lambda raw, nrm, a: timm_branch(nrm, "vit_base_patch16_dinov3.lvd1689m", True, a,
                                                  img_size=IMG_SIZE),
}


def branch_probability(model, X):
    """Apply one branch's fitted scaler, PCA and logistic head."""
    Xs = model["scaler"].transform(X)
    if model.get("pca") is not None:
        Xs = model["pca"].transform(Xs)
    return model["clf"].predict_proba(Xs)[:, 1]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="folder holding the crops")
    p.add_argument("--metadata", required=True, help="CSV with image_id and sun_azimuth_angle")
    p.add_argument("--weights", default="weights/stack_weights.joblib")
    p.add_argument("--out", default="predictions.csv")
    p.add_argument("--threshold", type=float, default=None,
                   help="decision threshold (default %s, tuned out-of-fold)" % DEFAULT_THRESHOLD)
    p.add_argument("--branches", nargs="+", default=None,
                   help="subset of branches to run; fewer than all skips the stack meta-model")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lfm-repo", default="lfm_repo")
    p.add_argument("--lfm-weights", default="lfm_weights")
    args = p.parse_args()

    models = joblib.load(args.weights)
    meta = models["_meta"]
    order = models["stack_image_only"]["branch_order"]
    wanted = args.branches or order
    unknown = [b for b in wanted if b not in BRANCHES]
    if unknown:
        p.error("unknown branches %s; available: %s" % (unknown, sorted(BRANCHES)))

    df = pd.read_csv(args.metadata)
    for col in ("image_id", "sun_azimuth_angle"):
        if col not in df.columns:
            p.error("--metadata is missing the column '%s'" % col)
    az = df.sun_azimuth_angle.values.astype(np.float32)

    t0 = time.time()
    raw = load_images(args.images, df.image_id)
    nrm = normalise(raw, az)
    print("%d crops loaded and azimuth-normalised in %.0fs" % (len(df), time.time() - t0), flush=True)

    probs = {}
    for name in wanted:
        t = time.time()
        feats = BRANCHES[name](raw, nrm, args)
        expected = models[name]["scaler"].n_features_in_
        if feats.shape[1] != expected:
            raise RuntimeError("branch '%s' produced %d features, the model expects %d"
                               % (name, feats.shape[1], expected))
        probs[name] = branch_probability(models[name], feats)
        print("  %-10s %s in %.0fs" % (name, feats.shape, time.time() - t), flush=True)

    if set(wanted) == set(order):
        P = np.stack([probs[n] for n in order], 1)
        stack = models["stack_image_only"]
        prob = stack["clf"].predict_proba(stack["scaler"].transform(P))[:, 1]
        source = "image-only stack over all %d branches" % len(order)
    else:
        prob = np.mean([probs[n] for n in wanted], 0)
        source = "mean of " + ", ".join(wanted) + " (the stack meta-model needs every branch)"

    thr = args.threshold if args.threshold is not None else DEFAULT_THRESHOLD
    out = pd.DataFrame({"image_id": df.image_id,
                        "label": (prob >= thr).astype(int),
                        "prob_rise": np.round(prob, 6)})
    out.to_csv(args.out, index=False)

    print("\n%s | threshold %s" % (source, thr))
    print("predicted %d Depth / %d Rise" % (int((out.label == 0).sum()), int((out.label == 1).sum())))
    print("normalisation: rotate by %s, crop %d, resize %d"
          % (meta["rotation"], meta["rot_crop"], meta["img_size"]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
