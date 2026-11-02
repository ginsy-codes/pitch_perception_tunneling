"""
Perception Stuff+ model.

Trains a regression on path-integral features (total variation of polar
differences vs. the pitcher's average primary fastball) to predict whiff rate.
Outputs a standardized Stuff+ score (100 = league average for the pitch type).

Train on 2025 aggregates; evaluate on 2026 once path integrals are available.

Usage:
    python model/stuff_plus.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INTEGRALS_PATH = ROOT / "datamining" / "outputs" / "path_integrals_2025.csv"
PITCHES_PATH = ROOT / "datamining" / "outputs" / "pitches_2025.csv"
OUT_PATH = ROOT / "results" / "stuff_plus_2025.csv"

# Map old column names to current nomenclature
COL_MAP = {
    "pre_tv_r": "pre_dp_r",
    "pre_tv_theta": "pre_dp_theta",
    "pre_tv_phi": "pre_dp_phi",
    "post_tv_r": "post_dp_r",
    "post_tv_theta": "post_dp_theta",
    "post_tv_phi": "post_dp_phi",
}

FEATURES = ["pre_dp_r", "pre_dp_theta", "pre_dp_phi", "post_dp_r", "post_dp_theta", "post_dp_phi"]

SWING_DESCS = {
    "foul", "foul_tip", "swinging_strike", "swinging_strike_blocked",
    "hit_into_play", "hit_into_play_score", "hit_into_play_no_out",
    "hit_into_play_line_drive", "hit_into_play_ground_ball", "hit_into_play_fly_ball",
}
WHIFF_DESCS = {"swinging_strike", "swinging_strike_blocked"}


def load_outcomes() -> pd.DataFrame:
    pitches = pd.read_csv(
        PITCHES_PATH,
        usecols=["pitcher", "pitch_type", "description", "stand"],
    )
    pitches = pitches.dropna(subset=["pitch_type", "stand"])
    pitches["is_swing"] = pitches["description"].isin(SWING_DESCS)
    pitches["is_whiff"] = pitches["description"].isin(WHIFF_DESCS)

    agg = pitches.groupby(["pitcher", "pitch_type", "stand"]).agg(
        n_pitches=("description", "size"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
    ).reset_index()
    agg["whiff_rate"] = np.where(agg["swings"] > 0, agg["whiffs"] / agg["swings"], np.nan)
    return agg[agg["n_pitches"] >= 30].copy()


def load_features() -> pd.DataFrame:
    df = pd.read_csv(INTEGRALS_PATH)
    for old, new in COL_MAP.items():
        if old in df.columns:
            df[new] = df[old]
    return df


def prepare(features: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    merged = features.merge(
        outcomes,
        left_on=["pitcher", "target", "handedness"],
        right_on=["pitcher", "pitch_type", "stand"],
        how="inner",
    )
    return merged.dropna(subset=FEATURES + ["whiff_rate"]).copy()


def train_and_score(df: pd.DataFrame) -> pd.DataFrame:
    """Train a ridge model per pitch type; produce Stuff+ (100 = mean)."""
    results = []

    for ptype, group in df.groupby("pitch_type"):
        if len(group) < 40:
            continue

        X = group[FEATURES].to_numpy()
        y = group["whiff_rate"].to_numpy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)

        pred = model.predict(X_scaled)
        # Stuff+: 100 = mean predicted whiff rate for this pitch type
        # Higher = better (more expected whiffs from path features)
        mean_pred = pred.mean()
        std_pred = pred.std()
        if std_pred < 1e-6:
            stuff = np.full_like(pred, 100.0)
        else:
            stuff = 100 + 15 * (pred - mean_pred) / std_pred

        out = group[["pitcher", "pitch_type", "handedness", "n_pitches", "whiff_rate"]].copy()
        out["pred_whiff"] = pred
        out["stuff_plus"] = stuff
        out["train_r2"] = r2_score(y, pred)
        out["train_mae"] = mean_absolute_error(y, pred)
        results.append(out)

        print(f"{ptype}: n={len(group)}, train r²={r2_score(y, pred):.3f}, "
              f"MAE={mean_absolute_error(y, pred):.3f}")

    return pd.concat(results, ignore_index=True)


def main():
    print("Loading...")
    features = load_features()
    outcomes = load_outcomes()
    df = prepare(features, outcomes)
    print(f"Training rows: {len(df)}")

    scored = train_and_score(df)
    scored = scored.sort_values("stuff_plus", ascending=False)
    scored.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
    print("\nTop 10 Stuff+:")
    print(scored.head(10)[["pitcher", "pitch_type", "handedness", "whiff_rate", "pred_whiff", "stuff_plus"]].to_string(index=False))
    print("\nBottom 5 Stuff+:")
    print(scored.tail(5)[["pitcher", "pitch_type", "handedness", "whiff_rate", "pred_whiff", "stuff_plus"]].to_string(index=False))


if __name__ == "__main__":
    main()
