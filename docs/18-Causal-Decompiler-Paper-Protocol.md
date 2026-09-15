# Causal Decompiler Paper Protocol (v2)

The paper answers three questions only:

- **RQ1** Can we automatically find the true causes of Y (slice → Top-k → C*_α)?
- **RQ2** Why does single-event knockout fail (AND / OR / redundant / suppressor)?
- **RQ3** Path and timing: event → seen → remembered → internal state → public/action?

LabWars is one long social scene. CrisisGrid (spatial comms) and ReleaseOps (workflow DAG) are independent scenario packs. The main numerical result is **200 parameterized mechanism worlds**, not the old six planted-world table.

## Method (four blocks)

1. **Five-layer IR** — E (event) → I (visibility, memory) → S (belief, relationship, goal) → B (action, public expression) → Y. Institution is IR context, not a layer.
2. **Automatic slice** — ancestors of Y, then layer → cluster → item. Expensive twins only on Top-5 (LLM) / Top-8 (scripted).
3. **Shared-noise Gumbel-CRN** — \(A_{i,t}=\arg\max_a[\log P(a\mid S_{i,t})+G_{i,t,a}]\) with event-keyed `STREAM_ACTION_GUMBEL`. Independent-resample baseline changes the salt.
4. **Bidirectional search + Minimal Effect-Recovery Set** — deletion (necessity) and restoration from empty (sufficiency). \(C_\alpha^*=\arg\min|S|\) s.t. \(v(S)\ge\alpha v(\text{Top-k})\), \(v(S)=Y(do(S))-Y(do(\varnothing))\). Main α=0.9; also report {0.8, 0.9, 0.95}. Harsanyi only on Top-k.

Paper interventions are layer-specific: `do_event`, `do_visibility` (hide/lock, not omniscient whole-world lesion), `do_memory`, `do_behavior`. Information algebra, `do_belief`, omniscient W2, and the six planted worlds are **not** paper tables (the six worlds remain unit tests).

## Outcomes

\(Y=(Y^{private},Y^{public},Y^{action})\), \(PPG=Y^{private}-Y^{public}\), \(PCI=Y^{private}-(Y^{public}+Y^{action})/2\).

- LabWars / CrisisGrid: three channels + PPG/PCI.
- ReleaseOps: objective task Y (failure, rollback, outage) only.

## Commands

Scripted MRI:

```bash
python -m src.experiments paper --rounds 8 --seed 11 --llm-provider scripted
python -m src.experiments paper --scenario crisisgrid --rounds 8 --llm-provider scripted
python -m src.experiments paper --scenario releaseops --rounds 8 --llm-provider scripted
```

200-world benchmark (CI default):

```bash
python -m src.experiments benchmark --n-per-family 50 --output output/reports
```

Replay a frozen LabWars jsonl (new IR/search; do not re-run 14×60 unless the cache is incompatible):

```bash
python -m src.experiments paper --from-jsonl output/runs/run_694c3a50.jsonl --full-cast
```

Paid matrix is explicit, not pytest:

```bash
python -m src.experiments matrix --scenarios labwars,crisisgrid,releaseops --providers ollama,deepseek --seeds 0,1,2,3,4,5,6,7,8,9
```

`--sampled-top-k` defaults to 1. DeepSeek: `DEEPSEEK_API_KEY` in gitignored `.env`. Local 4B: `--llm-provider ollama`.

## Tables

| Table | Content |
|---|---|
| Identity twin | CRN + LLM replay hits/misses |
| Five-layer IR | E/I/S/B/Y counts |
| Slice compression | graph nodes → ancestors → Top-k |
| Three-channel Y | private / public / action, PPG, PCI (task Y on ReleaseOps) |
| Bidirectional search | deletion + restoration on Top-k |
| C*_α | Minimal Effect-Recovery Set + α ∈ {0.8, 0.9, 0.95} |
| Layer dos | do_event / do_visibility / do_memory / do_behavior |
| 200 worlds | cause-set F1, interaction-sign accuracy, budget, false attribution |
| Baselines / ablations | recency, text sim, LLM-direct, knockout; no-IR / no-slice / knockout-only; shared vs independent Gumbel |

## Scenario packs

`load_world(scenario)` reads `config/scenarios/{name}/` when present. LabWars stays in `config/` as the compatibility entry.

| Pack | Topology | Mechanism | Y |
|---|---|---|---|
| LabWars | hierarchy, 60 rounds | delayed memory, AND, performative compliance | authorship / PPG |
| CrisisGrid | spatial comms, 20–30 rounds | visibility chain + OR reports | evac / stranded / resources |
| ReleaseOps | workflow DAG, staged | AND faults + alert→rollback suppressor | deploy fail / rollback / outage |
