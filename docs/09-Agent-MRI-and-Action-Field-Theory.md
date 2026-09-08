# Agent MRI and Action Field Theory

> LabWars treats the academic lab not as a story stage, but as a social pressure chamber: controlled stimuli, behavioral trajectories, and ablation experiments are used to infer the mechanisms behind long-horizon LLM-agent social behavior.

## 1. Core positioning

LabWars is not designed to prove that agents can complete research tasks. It asks a more diagnostic question: when agents face long-term collaboration, ambiguous authorship, unequal authority, contested contribution memory, and career dependency, which mechanisms push cooperation into conflict?

This is the meaning of **Agent MRI**:

```text
social stimulus -> behavioral trajectory -> state and memory change -> mechanism decompilation
```

Medical MRI uses external signals to infer hidden structure. Agent MRI uses social events, counterfactual interventions, and auditable action logs to infer the cognitive-social structure of an agent system.

## 2. Why this is not a normal multi-agent benchmark

Many multi-agent benchmarks ask:

- How can agents cooperate?
- How can agents finish tasks?
- How can success rate improve?

LabWars asks the inverse questions:

- How does a promise become grievance through long-term memory?
- How does contribution become perceived authorship entitlement?
- Why can public compliance coexist with private sabotage?
- How does authority pressure reshape integrity risk?

That places LabWars closer to computational sociology, cognitive architecture, and causal experimentation than to role-play simulation.

## 3. The LLM is an interpretation layer, not the sovereign actor

LabWars deliberately separates social pressure from language-model judgment:

```text
Continuous latent action field
        |
        v
Candidate actions
        |
        v
LLM subjective plausibility scoring
        |
        v
Field/LLM fusion
        |
        v
Actual action
        |
        v
Memory, emotion, belief, relationship update
```

The LLM does not freely overwrite the primary action. It mainly:

- scores the subjective plausibility of candidate actions;
- writes public positions, private intents, and constrained utterances after an action is selected;
- interprets events into subjective memories.

The structured action field provides the social-pressure prior. This lets reports separate three sources of behavior:

| Source | Meaning | Observable evidence |
|---|---|---|
| Social Physics | Structural pressure from power, contribution, dependency, relationships, and memory | field_score / decomposition |
| LLM Cognitive Layer | LLM plausibility judgment over candidate actions | llm_score / override pressure |
| Fusion Policy | The sampled action after combining field and LLM scores | fused_score / selected action |

## 4. What an action field should be

An action field should not be a hard-coded trigger system.

Weak design:

```python
if contribution > x:
    claim_authorship()
```

Preferred design:

```text
authorship_pressure =
    entitlement_gap
  + promise_violation_memory
  + contribution_visibility
  + perceived_exploitation
  + coalition_support
  - career_risk
  - dependency_on_PI
```

Authorship conflict should emerge from continuous pressure. The same principle applies to trust collapse, authority compliance, and integrity risk.

## 5. Four priority pressure fields

### 5.1 AuthorshipPressureField

Explains why an agent shifts from cooperation to authorship claims, protest, withdrawal threats, or coalition behavior.

Candidate decomposition terms:

- perceived contribution share
- expected authorship entitlement
- author-order fairness gap
- promise-violation memory
- credit visibility
- coalition support
- retaliation risk
- career dependency

### 5.2 TrustCollapseField

Explains how local events propagate into network-level trust fragmentation.

Candidate decomposition terms:

- betrayal salience
- repeated ambiguity
- third-party rumor propagation
- memory reconsolidation
- perceived apology sincerity
- dependency asymmetry

### 5.3 AuthorityComplianceField

Explains why an agent publicly complies with a PI, reviewer, funder, or institutional superior.

Candidate decomposition terms:

- career hostage index
- visa/funding dependency
- retaliation probability
- alternative opportunities
- coalition protection
- private resentment

### 5.4 IntegrityRiskField

Explains why academic integrity risk rises under pressure.

Candidate decomposition terms:

- deadline pressure
- authorship threat
- PI control pressure
- external competition
- reproducibility confidence
- public/private divergence

## 6. Lesion-style mechanism experiments

LabWars experiments should be organized as lesions, interventions, and ablations.

| Experiment | Question | Core conditions | Key metrics |
|---|---|---|---|
| Memory Lesion | Does conflict disappear without long-term memory? | full / no / shuffled / delayed / false memory | memory_causal_impact, dispute index |
| Hierarchy Ablation | What happens when PI authority is weakened or removed? | normal / flat / weak / authoritarian hierarchy | authority_compliance, coalition_strength |
| Credit Visibility | How does visible or noisy contribution tracking affect credit conflict? | transparent / private / noisy / falsified ledger | credit_threat, protest_authorship |
| Cognition Mode | Does conflict come from structure or LLM interpretation? | social_physics / dual_engine / llm_native / lambda sweep | override pressure, trajectory divergence |
| False Evidence | How does misinformation reshape trust networks? | insert / correct / delay / source-status manipulation | trust_recovery_rate, belief persistence |
| Public-Private Split | When do agents comply publicly but defect privately? | surveillance / anonymity / retaliation risk | public_private_divergence |

## 7. Hierarchical academic society roadmap

Do not scale by simply increasing 14 agents to 100 agents. First introduce hierarchy:

```text
University
`-- Department
    |-- Lab A
    |   |-- PI
    |   |-- Postdoc
    |   |-- PhD
    |   `-- RA
    |-- Rival Lab
    |-- Reviewers / Editor
    |-- Funding Agency
    `-- Industry Partner
```

Cross-layer events matter more than raw population size:

- rival lab scoops result
- reviewer conflict of interest
- editor delays review
- funder pressures PI
- alumni gives private warning
- department chair protects PI
- industry partner wants patent priority

## 8. Paper/report narrative

A strong framing is:

1. LabWars is a long-horizon Agent MRI environment for academic-organization conflict.
2. It separates Social Physics from LLM cognitive scoring through a dual-engine policy.
3. It uses memory IRF, CRN contrasts, and optional λ lesions to decompile agent social behavior.

The conference experiment is the Causal Decompiler protocol in [`docs/18-Causal-Decompiler-Paper-Protocol.md`](18-Causal-Decompiler-Paper-Protocol.md): MRI of one frozen 14-agent / 60-round trajectory.
