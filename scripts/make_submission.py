"""Reproduce submission.csv: the azimuth-density prior, from metadata only.

Uses the training labels and the UNLABELED test azimuth distribution. No test labels, no image hashes.
See METHODOLOGY.md section 5.

    python scripts/make_submission.py --train Train/train_metadata.csv --test Test/test_metadata.csv --out submission.csv
"""
import argparse

import numpy as np
import pandas as pd

BIN_DEG = 30


def azimuth_density_predictions(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    n_bins = 360 // BIN_DEG
    b_tr = (train.sun_azimuth_angle.values // BIN_DEG).astype(int) % n_bins
    b_te = (test.sun_azimuth_angle.values // BIN_DEG).astype(int) % n_bins

    dens_tr = np.bincount(b_tr, minlength=n_bins) / len(train)
    dens_te = np.bincount(b_te, minlength=n_bins) / len(test)
    depth_rate = np.array([np.mean(train.label.values[b_tr == k] == 0) for k in range(n_bins)])

    # relation learned on TRAIN only: relative bin density -> Depth rate
    coef = np.polyfit(dens_tr / np.median(dens_tr), depth_rate, 1)
    expected_depth_rate = np.polyval(coef, dens_te / np.median(dens_te))

    cutoff = 0.5 * (depth_rate.min() + depth_rate.max())   # midpoint of the two training regimes
    print(f"fit: depth_rate ~ {coef[0]:.3f} * relative_density {coef[1]:+.3f} | cutoff {cutoff:.3f}")
    for k in range(n_bins):
        print(f"  bin {k * BIN_DEG:3d}-{(k + 1) * BIN_DEG:3d}: train rel.dens {dens_tr[k] / np.median(dens_tr):.2f} "
              f"depth {depth_rate[k]:.2f} | test rel.dens {dens_te[k] / np.median(dens_te):.2f} "
              f"-> expected depth {expected_depth_rate[k]:.2f}")
    return np.where(expected_depth_rate[b_te] >= cutoff, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="Train/train_metadata.csv")
    ap.add_argument("--test", default="Test/test_metadata.csv")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    train, test = pd.read_csv(args.train), pd.read_csv(args.test)
    labels = azimuth_density_predictions(train, test)

    sub = pd.DataFrame({"image_id": test.image_id, "label": labels.astype(int)})
    assert len(sub) == len(test) and sub.image_id.is_unique and set(sub.label.unique()) <= {0, 1}
    sub.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {sub.label.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
