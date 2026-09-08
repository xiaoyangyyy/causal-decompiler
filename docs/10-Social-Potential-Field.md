# Social Potential Field

> Social Potential Field is the v0.2 theory object of LabWars: a compact, auditable state vector that explains how organizational pressure becomes action tendency.

## 1. Motivation

LabWars should not be read as a scripted lab-drama simulator. Its central object is the latent social pressure acting on each agent at each event. The Social Potential Field makes that object explicit.

For agent `i` at round `t`:

```text
S_i(t) = [R_i, T_i, P_i, C_i, U_i, M_i]
```

Where:

| Symbol | Field | Meaning |
|---|---|---|
| R | reputation_pressure | Fear of losing credit, visibility, or career standing |
| T | trust_deficit | Lack of trust, resentment, anger, and low team morale |
| P | power_constraint | Dependence on PI/institution and low access to alternatives |
| C | contribution_entitlement | Gap between contribution and expected authorship recognition |
| U | uncertainty | Deadline, rival, feasibility, and epistemic uncertainty |
| M | memory_pressure | Emotionally salient recalled memory pressure |

The action tendency becomes:

```text
a_i(t) = f(S_i(t), S_j(t), E_t, H_i(t))
```

where `E_t` is the current event and `H_i(t)` is the agent memory/history state.

## 2. Design constraints

1. Social Potential is derived from existing state; it does not replace beliefs, memory, or relationships.
2. It is observable in action logs and aggregate outcomes.
3. It supports lesion-style ablations by zeroing dimensions.
4. The first implementation is diagnostic, not a hard policy rewrite.

## 3. Current implementation

Code lives in [`src/cognition/social_potential.py`](../src/cognition/social_potential.py).

Each action log now records:

- `social_potential`
- `selected_social_pressure`
- `selected_social_pressure_decomposition`
- `social_potential_ablation`

Final outcomes also include mean field dimensions:

- `social_potential_reputation_pressure_mean`
- `social_potential_trust_deficit_mean`
- `social_potential_power_constraint_mean`
- `social_potential_contribution_entitlement_mean`
- `social_potential_uncertainty_mean`
- `social_potential_memory_pressure_mean`
- `selected_social_pressure_mean`
- `selected_social_pressure_max`

## 4. Lesion protocol

A lesion asks a counterfactual question over the same selected action:

```text
How much selected-action pressure remains if dimension D is removed?
```

Each action log records the field on the selected action. MRI reports read those traces; there is no separate ablation matrix.

## 5. Why post-hoc first?

The first v0.2 implementation records the field without strongly changing behavior. This preserves comparability with the existing dual-engine action policy. Once the field is validated, future versions can feed selected Social Potential dimensions back into candidate generation as calibrated priors.

## 6. Next implementation step

The next step is to connect the field to explicit pressure-specific reports:

- AuthorshipPressureField
- TrustCollapseField
- AuthorityComplianceField
- IntegrityRiskField

Those can be implemented as named projections of Social Potential rather than separate unrelated systems.

## 7. Relation to Agent Social State

Social Potential Field is the implemented projection of the formal Agent Social State described in [`docs/11-Agent-Social-State-Model.md`](11-Agent-Social-State-Model.md).
