# Pitch Perception Tunneling

Batter-centric analysis of pitch trajectories in polar coordinates, measuring path similarity (tunneling) relative to a reference and correlating with outcomes.

## Methodology

### Coordinate Frame

All trajectories are reconstructed under constant acceleration from Statcast release position, release speed, and acceleration components. Positions are then transformed into a batter-eye frame:

- Origin at approximate batter eye: \(x = \pm 35.8\) in (sign by handedness), \(y = -28.3\) in, \(z = 65\) in
- Polar coordinates relative to that origin:
  - \(r\): radial distance from the eye
  - \(\theta\): azimuth about the vertical axis
  - \(\phi\): elevation about the lateral axis

Separate polar trajectories are generated for right- and left-handed hitters.

### Decision Point

For each pitcher, the decision horizon is estimated from the average primary fastball:

\[
t_{\text{decision}} = t_{\text{flight}} - 300\,\text{ms}
\]

(150 ms swing time + 150 ms reaction buffer), clamped to a minimum of 50 ms. The same \(t_{\text{decision}}\) is used for all of that pitcher’s pitches.

### Path Integrals (Total Variation)

For a target pitch and a reference trajectory, the difference \(\Delta\) is computed in each polar coordinate. The total variation over a time window is:

\[
\text{TV} = \int |d\Delta / dt| \, dt
\]

Two windows are used:

- **Pre-decision** (`pre_dp_*`): release → decision point
- **Post-decision** (`post_dp_*`): decision point → decision point + 100 ms

Only total variation is retained (net change was dropped). This yields six features per comparison: `pre_dp_r`, `pre_dp_theta`, `pre_dp_phi`, `post_dp_r`, `post_dp_theta`, `post_dp_phi`.

### Reference Trajectories

Two reference types are examined:

1. **Season-average primary fastball** (main analysis) — the pitcher’s most-used fastball (FF/SI/FA/FT), averaged.
2. **Immediately preceding pitch** (pitch-to-pitch tunneling) — the prior pitch in the same plate appearance (or the most recent pitch by the same pitcher if none exists in the PA). This tests whether hitters are anchoring on the just-seen pitch rather than (or in addition to) a long-run average.

### Outcomes

Performance metrics are aggregated at the pitcher × pitch-type × handedness level (minimum 30 pitches):

- Whiff rate = swinging strikes / swings
- Hard-hit rate = (exit velocity ≥ 95 mph) / in-play balls
- xwOBACON = mean estimated wOBA on contact

Linear regressions relate each path-integral feature to each metric. Results are reported by individual pitch type.

## Results

### Season-Average Fastball Reference

Radial separation after the decision point is the strongest and most consistent signal.

| Pitch type | Feature | Metric | \(r^2\) | Direction |
|------------|---------|--------|---------|-----------|
| FS (splitter) | post_dp_r | whiff | 0.177 | + separation → more whiffs |
| SL (slider) | post_dp_r | whiff | 0.127 | + separation → more whiffs |
| CH (changeup) | post_dp_r | hard hit | 0.085 | + separation → fewer hard hits |
| FC (cutter) | post_dp_r | whiff | 0.077 | + separation → more whiffs |
| CU (curve) | post_dp_r | hard hit | 0.054 | + separation → fewer hard hits |
| ST (sweeper) | post_dp_theta | hard hit | 0.066 | + angular separation → fewer hard hits |

- Post-decision radial variation generally outperforms pre-decision variation for whiff rate.
- Angular features (θ, φ) are weaker except for specific shapes (sweepers lean on θ; knuckle-curves show a negative pre_dp_phi–whiff relationship on a modest sample).
- xwOBACON is largely unaffected once contact occurs.

### Pitch-to-Pitch Tunneling

When the reference is the immediately preceding pitch rather than the season-average fastball:

- Overall predictive power is lower than the average-fastball reference for most pitch types.
- Residual signal remains for consecutive offspeed/breaking pitches, particularly when the prior pitch was a fastball.
- This suggests hitters maintain a relatively stable internal model of the primary fastball and are only modestly re-anchored by the single most recent pitch within an at-bat.

### Interpretation

Hitters appear to construct an internal model of the pitcher’s primary fastball trajectory in a batter-centric frame. Deviations in radial distance (how the path length / depth evolves relative to the eye) after the decision point are most disruptive — both increasing whiffs and reducing hard-hit rates on contact. Angular deviations matter less on average, though certain pitch shapes exploit them. The weaker pitch-to-pitch results imply that the long-run average fastball is a stronger perceptual anchor than the immediately preceding pitch for most hitters.

## Repository Structure

- `datamining/` — data pulls and averages
- `polar_transformations/` — trajectory reconstruction, polar conversion, decision-point and path-integral calculations
- `analysis/` — correlation scripts and pitch-to-pitch tunneling
- `report_generation/` — visualization utilities
- `results/` — published correlation tables

## Data

2025 Statcast pitch-level data (via pybaseball). Aggregates and path integrals are stored as CSVs under `results/` and intermediate outputs under `datamining/outputs/`.

## Pitch-to-Pitch Results (Proxy)

Using absolute differences in plate location and release speed relative to the immediately preceding pitch in the same plate appearance (a simpler proxy than full polar reconstruction):

| Pitch type | Feature | Metric | \(r^2\) | Direction |
|------------|---------|--------|---------|-----------|
| KC | mean Δ plate_x | hard hit | 0.283 | larger horizontal difference → fewer hard hits |
| ST | mean Δ location | whiff | 0.103 | larger location difference → more whiffs |
| SL | mean Δ location | whiff | 0.098 | larger location difference → more whiffs |
| FS | mean Δ speed | whiff | 0.071 | larger speed difference → more whiffs |
| CH | mean Δ plate_z | whiff | 0.063 | larger vertical difference → more whiffs |

The location-difference signal for sliders and sweepers is directionally consistent with the polar-path results, though weaker overall than the season-average-fastball radial integrals. The strong KC plate_x–hard-hit relationship is on a small sample and warrants caution. Full polar reconstruction of consecutive pairs remains an open extension.

## Stuff+ Model

A predictive model trained on 2025 path-integral features to produce a standardized score.

### Training (2025)

Ridge regression per pitch type on the six total-variation features (`pre_dp_r`, `pre_dp_theta`, `pre_dp_phi`, `post_dp_r`, `post_dp_theta`, `post_dp_phi`) predicting whiff rate. Stuff+ is standardized so that 100 is the mean predicted whiff rate for that pitch type, with 15 points ≈ 1 standard deviation of the predicted values.

| Pitch type | n | Train r² | MAE |
|------------|---|----------|-----|
| SL | 726 | 0.214 | 0.081 |
| FS | 171 | 0.222 | 0.081 |
| KC | 76 | 0.205 | 0.092 |
| CU | 367 | 0.164 | 0.081 |
| FC | 407 | 0.158 | 0.066 |
| CH | 476 | 0.121 | 0.081 |
| ST | 416 | 0.114 | 0.086 |
| SI | 306 | 0.036 | 0.049 |
| FF | 178 | 0.023 | 0.075 |

The model captures meaningful signal for sliders, splitters, and curves. Fastballs (FF/SI) show little within-type variation explained by path differences from their own average, as expected.

### 2026 Evaluation

Full evaluation on 2026 requires computing path integrals for 2026 pitches (trajectory reconstruction → polar transform → decision-point integrals vs. each pitcher's average fastball). An April 2026 sample was pulled; a proxy model using only plate-location and speed differences (not full polar path integrals) showed weak in-sample fit (r² ≈ 0.09 overall), confirming that the polar trajectory features are not reducible to simple endpoint differences.

The trained 2025 model and scoring code are in `model/stuff_plus.py`. Applying it to 2026 requires running the full `polar_transformations` pipeline on 2026 data to produce the six path-integral features.
