# Agent Social State Model

> This document turns LabWars from a simulation into a model: every agent has a latent social state, that state evolves under events and actions, and policies are generated from a fusion of social physics and LLM cognitive scoring.

## 1. Unified mathematical object

For each agent `i` at round `t`, define the Agent Social State:

```text
z_i(t) = [r_i(t), tau_i(t), p_i(t), c_i(t), u_i(t), m_i(t)]
```

Where:

| Term | Code projection | Meaning |
|---|---|---|
| `r_i(t)` | reputation_pressure | Status and credit-loss pressure |
| `tau_i(t)` | trust_deficit | Trust erosion, resentment, anger, low morale |
| `p_i(t)` | power_constraint | Dependency on PI/institution and lack of alternatives |
| `c_i(t)` | contribution_entitlement | Perceived contribution-authorship gap |
| `u_i(t)` | uncertainty | Deadline, rival, feasibility, and epistemic uncertainty |
| `m_i(t)` | memory_pressure | Salient affective pressure from recall/history |

The current implementation exposes this through `SocialPotentialField` in `src/cognition/social_potential.py`.

## 2. State transition

The social state evolves as:

```text
z_i(t+1) = F(z_i(t), z_j(t), e_t, a_t, mu_i(t))
```

Where:

- `z_j(t)` is the relevant counterpart or organization context;
- `e_t` is the current event;
- `a_t` is the selected action profile;
- `mu_i(t)` is the agent memory state, including recalled and newly consolidated memories.

LabWars implements `F` through structured updates in memory, beliefs, emotions, relationships, authorship ledgers, and project state.

## 3. Candidate-action policy

The social field produces candidate actions:

```text
A_i(t) = {a_i1, a_i2, ..., a_ik}
```

Each candidate receives a structural score:

```text
s_field(a_ik) = phi(a_ik, z_i(t), e_t, H_i(t))
```

Where `H_i(t)` is the history available to the agent.

## 4. LLM cognitive scoring

The LLM does not own the action. It scores candidate plausibility:

```text
s_llm(a_ik) = G_theta(a_ik, z_i(t), e_t, H_i(t))
```

This makes the LLM a cognitive interpretation layer rather than a sovereign actor.

## 5. Fusion policy

The selected action is sampled from fused scores:

```text
s_fused(a_ik) = (1 - lambda) * s_field(a_ik) + lambda * s_llm(a_ik)
```

Then:

```text
a_i(t) ~ softmax(s_fused(A_i(t)))
```

Important limits:

- `lambda = 0`: Social Physics only
- `0 < lambda < 1`: dual-engine policy
- `lambda = 1`: LLM cognitive scoring dominates candidate ranking
- `llm_native`: LLM proposes candidates directly and is mapped back into the action schema

## 6. Falsifiable predictions

The model is useful only if interventions change trajectories in predictable ways.

| Lesion | Removed mechanism | Prediction |
|---|---|---|
| memory_lesion | `m_i(t)` / recall channel | short-term cooperation can remain, but long-horizon grievance and stable coalition formation weaken |
| hierarchy_lesion | `p_i(t)` / authority dependency | authority compliance and public-private divergence should drop |
| social_physics_only | LLM candidate scoring | trajectories remain structurally coherent but language-mediated plausibility shifts disappear |
| llm_native | structured candidate generator | trajectories become more language-native and less anchored to calibrated social pressure |
| llm_scoring_off | LLM plausibility scoring | field candidates remain, but LLM override pressure goes to zero |

## 7. Why this matters

Without `z_i(t)`, LabWars is merely a simulator with many variables. With `z_i(t)`, it becomes an Agent Organization MRI model: a compact state representation, a transition function, a policy decomposition, and an intervention protocol.

## 8. Benchmark layer

The formal state object is read by Causal MRI on the canonical 14-agent trajectory. See [`docs/18-Causal-Decompiler-Paper-Protocol.md`](18-Causal-Decompiler-Paper-Protocol.md).
