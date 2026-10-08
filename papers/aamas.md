# Anchor density, not the language model, moves the registered channels

Pre-registered endpoints and the decision rule are in `docs/21-AAMAS-Preregistration.md`. The completed grid is `papers/results/aamas_preregistered.json`. This note reports that grid. It does not reuse the old 200-world F1 table.

## Question

On LabWars (60 rounds, full cast), CrisisGrid (24), and ReleaseOps (16), do the seed-means of `y_private`, `y_public`, and `y_action` move when the provider changes, or only when the anchor schedule is cut in half?

Movement means an absolute difference of at least 0.02. Seeds are 0, 1, and 2. `cognitive_sampling_top_k` is 1. LabWars is not the MVP subset.

## What was run

Eighteen scripted cells: 3 scenarios × 3 seeds × anchor fractions {1.0, 0.5}. Half-anchors keep even positions after sorting by round and event id. Errors: 0.

A full-length ReleaseOps cell with local `llama3.2`, seed 0, and all anchors returned after 3038 seconds (`run_id` `8ecbad0a`). All three channels were 0.625, the same values as every scripted ReleaseOps cell. Those channels were the presence score: one remaining event of each fault type sets the switch, and all three channels copy that task value. The live pack score counts events instead of switching them; it does not rewrite this grid. That single cell is not part of the eighteen-row grid, and LabWars and CrisisGrid still have no second-model rows. The pre-registered provider contrast therefore still has one completed provider.

## Decision

The registered rule returns `cross_model_not_identified`. No second provider is in the eighteen cells, so no channel is declared model-sensitive.

The registered anchor contrast, pooled across scenarios, does move:

| Channel | Half anchors minus full anchors |
|---|---:|
| y_private | −0.040 |
| y_public | −0.025 |
| y_action | −0.028 |

All three exceed 0.02. The same contrast split by scenario does not:

| Scenario | y_private | y_public | y_action | Moves at 0.02 |
|---|---:|---:|---:|---|
| LabWars | −0.112 | −0.074 | −0.083 | yes |
| CrisisGrid | −0.008 | 0.000 | 0.000 | no |
| ReleaseOps | 0.000 | 0.000 | 0.000 | no |

ReleaseOps stays at 0.625 on every seed and both anchor fractions. The pooled “yes” is the LabWars shift averaged with two near-zero scenarios. The per-scenario split was not a second primary endpoint; it is how the pooled number is made readable.

## What this does not say

It does not say that a second language model would leave the channels unchanged. That cell was not in the completed grid. It does not say that halving anchors changes CrisisGrid or ReleaseOps. Under the scripted dynamics those two packs sit on schedules whose registered channels do not respond to dropping every other anchor.

Search ranking for the companion causal procedure no longer adds a bonus for event ids E003 or E052. That change is not what produced the table above. The table is the simulator’s three-channel outcomes under the two anchor fractions.
