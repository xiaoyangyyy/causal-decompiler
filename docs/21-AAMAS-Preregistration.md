# AAMAS pre-registration

Written before the registered grid in `papers/results/aamas_preregistered.json`. Endpoints are not chosen after seeing that file.

## Endpoints

Seed-mean of `y_private`, `y_public`, and `y_action` from `three_channel_y`. No other outcome may be promoted to a primary endpoint after the run.

## Contrasts

1. Provider contrast. Two providers, same scenarios, seeds, and anchor fraction. Movement means the absolute difference of seed-means on at least one endpoint is at least 0.02.
2. If that contrast is unavailable or does not move, the follow-up is anchor halving: `anchor_keep_fraction` 1.0 versus 0.5, even-indexed anchors after sorting by round and event id. The same 0.02 rule applies. A difference below 0.02 is reported as no movement. It is not a reason to switch endpoints.

## Design

- Scenarios and lengths: LabWars 60, CrisisGrid 24, ReleaseOps 16.
- Seeds: 0, 1, 2.
- Sampling: `cognitive_sampling_top_k` = 1.
- LabWars uses the full cast (`mvp` false). The other packs have no MVP subset.

## Decision rule

- Both providers present and at least one endpoint moves: claim `model_moves_registered_channel`.
- Second provider missing: claim `cross_model_not_identified`. Do not impute the missing model from the scripted arm.
- Both providers present and no endpoint moves: claim `anchors_dominate_under_registered_threshold`, then report the anchor-half contrast under the same threshold.
