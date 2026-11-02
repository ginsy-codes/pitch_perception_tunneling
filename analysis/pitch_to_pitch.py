"""
Pitch-to-pitch tunneling analysis.

For each pitch, compute path-integral features relative to the immediately
preceding pitch in the same plate appearance (or the most recent pitch by
the same pitcher). Correlate those features with outcomes and compare to
the season-average-fastball reference.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PITCHES_PATH = Path("/tmp/polar_tunneling/datamining/outputs/pitches_2025.csv")
INTEGRALS_PATH = Path("/tmp/polar_tunneling/datamining/outputs/path_integrals_2025.csv")
OUT_PATH = ROOT / "results" / "pitch_to_pitch_correlations.csv"

SWING_DESCS = {
    "foul", "foul_tip", "swinging_strike", "swinging_strike_blocked",
    "hit_into_play", "hit_into_play_score", "hit_into_play_no_out",
    "hit_into_play_line_drive", "hit_into_play_ground_ball", "hit_into_play_fly_ball",
}
WHIFF_DESCS = {"swinging_strike", "swinging_strike_blocked"}
IN_PLAY = {
    "hit_into_play", "hit_into_play_score", "hit_into_play_no_out",
    "hit_into_play_line_drive", "hit_into_play_ground_ball", "hit_into_play_fly_ball",
}


def load_and_prepare():
    pitches = pd.read_csv(
        PITCHES_PATH,
        usecols=["pitcher", "pitch_type", "description", "stand", "at_bat_number",
                 "pitch_number", "launch_speed", "estimated_woba_using_speedangle",
                 "release_speed_mph", "plate_x_ft", "plate_z_ft"],
    )
    pitches = pitches.dropna(subset=["pitch_type", "stand", "at_bat_number"])
    pitches = pitches.sort_values(["pitcher", "at_bat_number", "pitch_number"])

    pitches["is_swing"] = pitches["description"].isin(SWING_DESCS)
    pitches["is_whiff"] = pitches["description"].isin(WHIFF_DESCS)
    pitches["is_in_play"] = pitches["description"].isin(IN_PLAY)
    pitches["is_hard_hit"] = pitches["is_in_play"] & (pitches["launch_speed"] >= 95)

    # Prior pitch within the same plate appearance
    pitches["prior_pitch_type"] = pitches.groupby(["pitcher", "at_bat_number"])["pitch_type"].shift(1)
    pitches["prior_release_speed_mph"] = pitches.groupby(["pitcher", "at_bat_number"])["release_speed_mph"].shift(1)
    pitches["prior_plate_x"] = pitches.groupby(["pitcher", "at_bat_number"])["plate_x_ft"].shift(1)
    pitches["prior_plate_z"] = pitches.groupby(["pitcher", "at_bat_number"])["plate_z_ft"].shift(1)

    # Simple proxy for path difference: absolute differences in plate location and speed
    # (full polar reconstruction of every consecutive pair is expensive; this is a first-order proxy)
    pitches["delta_plate_x"] = (pitches["plate_x_ft"] - pitches["prior_plate_x"]).abs()
    pitches["delta_plate_z"] = (pitches["plate_z_ft"] - pitches["prior_plate_z"]).abs()
    pitches["delta_speed"] = (pitches["release_speed_mph"] - pitches["prior_release_speed_mph"]).abs()
    pitches["delta_location"] = np.sqrt(pitches["delta_plate_x"]**2 + pitches["delta_plate_z"]**2)

    return pitches


def aggregate_outcomes(pitches: pd.DataFrame) -> pd.DataFrame:
    # Only pitches that have a prior pitch
    has_prior = pitches.dropna(subset=["prior_pitch_type"])

    agg = has_prior.groupby(["pitcher", "pitch_type", "stand"]).agg(
        n_pitches=("description", "size"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        in_play=("is_in_play", "sum"),
        hard_hits=("is_hard_hit", "sum"),
        mean_delta_location=("delta_location", "mean"),
        mean_delta_speed=("delta_speed", "mean"),
        mean_delta_plate_x=("delta_plate_x", "mean"),
        mean_delta_plate_z=("delta_plate_z", "mean"),
    ).reset_index()

    agg["whiff_rate"] = np.where(agg["swings"] > 0, agg["whiffs"] / agg["swings"], np.nan)
    agg["hard_hit_rate"] = np.where(agg["in_play"] > 0, agg["hard_hits"] / agg["in_play"], np.nan)

    # xwOBACON
    def xwoba(g):
        ip = g[g["is_in_play"] & g["estimated_woba_using_speedangle"].notna()]
        return ip["estimated_woba_using_speedangle"].mean() if len(ip) else np.nan

    xw = has_prior.groupby(["pitcher", "pitch_type", "stand"]).apply(
        xwoba, include_groups=False
    ).reset_index(name="xwobacon")
    agg = agg.merge(xw, on=["pitcher", "pitch_type", "stand"])
    return agg[agg["n_pitches"] >= 30].copy()


def run_correlations(agg: pd.DataFrame) -> pd.DataFrame:
    features = ["mean_delta_location", "mean_delta_speed", "mean_delta_plate_x", "mean_delta_plate_z"]
    metrics = ["whiff_rate", "hard_hit_rate", "xwobacon"]

    common = agg["pitch_type"].value_counts()
    common = common[common >= 50].index.tolist()

    rows = []
    for ptype in common:
        subset = agg[agg["pitch_type"] == ptype]
        for feat in features:
            for met in metrics:
                sub = subset[[feat, met]].dropna()
                if len(sub) < 40:
                    continue
                x, y = sub[feat].to_numpy(), sub[met].to_numpy()
                slope, intercept, r, p, _ = stats.linregress(x, y)
                rows.append({
                    "pitch_type": ptype,
                    "feature": feat,
                    "metric": met,
                    "n": len(sub),
                    "r": r,
                    "r2": r**2,
                    "p_value": p,
                })

    return pd.DataFrame(rows).sort_values("r2", ascending=False)


def main():
    print("Loading pitches...")
    pitches = load_and_prepare()
    print(f"Pitches with prior: {pitches['prior_pitch_type'].notna().sum()}")

    print("Aggregating...")
    agg = aggregate_outcomes(pitches)
    print(f"Aggregated rows: {len(agg)}")

    print("Correlating...")
    results = run_correlations(agg)
    results.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")
    print("\nTop results:")
    print(results.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
