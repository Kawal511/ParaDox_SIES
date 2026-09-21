"""The evidence in METHODOLOGY.md section 3: do the images carry any label signal?

Four tests on the training set (no deep learning required for 1, 2 and 4):
  1. byte-identical duplicate images and their labels
  2. nearest-neighbour label agreement among non-duplicate images
  3. pixels -> azimuth regression (do the renders reflect the metadata sun angle?)
  4. azimuth-only vs azimuth+image cross-validation

    python scripts/evidence_image_signal.py --train-dir Train/train_images --train-csv Train/train_metadata.csv
"""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default="Train/train_images")
    ap.add_argument("--train-csv", default="Train/train_metadata.csv")
    ap.add_argument("--thumb", type=int, default=32)
    args = ap.parse_args()

    tr = pd.read_csv(args.train_csv)
    y = tr.label.values
    az = tr.sun_azimuth_angle.values

    def md5(name: str) -> str:
        with open(f"{args.train_dir}/{name}", "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def thumb(name: str) -> np.ndarray:
        im = cv2.imread(f"{args.train_dir}/{name}", cv2.IMREAD_GRAYSCALE)
        return cv2.resize(im, (args.thumb, args.thumb), interpolation=cv2.INTER_AREA)

    with ThreadPoolExecutor(8) as ex:
        tr["h"] = list(ex.map(md5, tr.image_id))
        X = np.stack(list(ex.map(thumb, tr.image_id))).reshape(len(tr), -1).astype(np.float32)
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-6)

    # 1. duplicates
    counts = tr.h.map(tr.h.value_counts()).values
    groups = tr[counts > 1].groupby("h").label.nunique()
    print(f"1. duplicate image groups: {len(groups)} | with OPPOSITE labels: {(groups > 1).sum()}")

    # 2. nearest neighbour among unique images
    u = counts == 1
    Xu = X[u] / np.linalg.norm(X[u], axis=1, keepdims=True)
    sims = Xu @ Xu.T
    np.fill_diagonal(sims, -1)
    yu = y[u]
    chance = yu.mean() ** 2 + (1 - yu.mean()) ** 2
    print(f"2. NN label agreement (unique images): {np.mean(yu[sims.argmax(1)] == yu):.3f} vs chance {chance:.3f}")
    del sims

    # 3. pixels -> azimuth
    rad = np.deg2rad(az)
    for name, target in (("sin", np.sin(rad)), ("cos", np.cos(rad))):
        oof = np.zeros(len(target))
        for a, b in KFold(5, shuffle=True, random_state=42).split(X):
            oof[b] = RidgeCV(alphas=np.logspace(0, 4, 9)).fit(X[a], target[a]).predict(X[b])
        print(f"3. pixels -> {name}(azimuth): CV R2 = {r2_score(target, oof):.3f}")

    # 4. azimuth only vs azimuth + image
    A = np.concatenate([np.stack([np.sin(k * rad), np.cos(k * rad)], 1) for k in (1, 2, 3)], 1)
    Z = PCA(64, random_state=42).fit_transform(X)
    w = (len(y) / (2.0 * np.bincount(y)))[y]
    for name, feats in (("azimuth only", A),
                        ("azimuth + image + image x azimuth", np.concatenate([A, Z, Z * A[:, :1], Z * A[:, 1:2]], 1))):
        oof = np.zeros(len(y))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(feats, y):
            sc = StandardScaler().fit(feats[a])
            model = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(feats[a]), y[a], sample_weight=w[a])
            oof[b] = model.predict_proba(sc.transform(feats[b]))[:, 1]
        print(f"4. {name:34s} CV AUC {roc_auc_score(y, oof):.4f}")


if __name__ == "__main__":
    main()
