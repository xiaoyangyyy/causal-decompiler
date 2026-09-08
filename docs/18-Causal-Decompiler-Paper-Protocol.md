# Causal Decompiler Paper Protocol

> LabWars' conference-facing experiment is not a 630-cell ATE grid. It is an **MRI of one factual trajectory**: freeze exogenous noise and LLM text, patch the unrolled SCM, and read a vector of public vs private outcomes.

## 1. What is supposed to surprise a reviewer

Most LLM-agent papers report "treatment X changed success rate." LabWars reports four objects that usually do not exist in those papers:

1. **Identity twin.** A no-op counterfactual must reproduce the factual actions. If this fails, later ATEs are theatre.
2. **Split-Y.** Public protest can sit at 0.03 while private divergence is 0.5. Same `do()`, two transcripts.
3. **Memory IRF.** Deleting the same cluster at t=3 vs t=45 vs t=52 is an interventional curve of **public protest vs private PPD**, not a mediation fraction `|ΔM/ΔY|`.
4. **AND-cause lie.** Skipping E003 (promise) and skipping E052 (draft) each look decisive; Shapley splits them. The planted SCM (Y = promise AND draft) is the oracle: knockout sums to 2, Shapley sums to 1.
5. **Fork moment.** After a patch, the first round where action / public stance / private intent leaves the factual transcript.

Three-worlds is pinned to the draft beat (E052 / R52): W0 factual, W1 `do(skip draft)`, W2 `do(skip draft)` + omniscient observation. If private divergence moves and public compliance does not, the effect lived in the hidden transcript.

## 2. Commands

Scripted smoke (no API):

```bash
python -m src.experiments paper --rounds 8 --seed 11 --llm-provider scripted
```

`--sampled-top-k` defaults to 1. DeepSeek config sets `thinking: disabled`.

Full 60-round MRI after a persisted factual run (replays the LLM sidecar; twins re-derive keyed noise from the seed, they do not inject persisted draws):

```bash
python -m src.experiments paper --from-jsonl output/runs/run_XXXX.jsonl --full-cast
```

A/B/C/D as CRN twins of one control:

```bash
python -m src.experiments paper --rounds 60 --full-cast --contrasts A --contrast-seeds 0,1,2 --llm-provider scripted
```

Library surface — one MRI entry, optional CRN pairs:

```python
from src.experiments import run_paper_protocol, run_paper_contrasts, run_crn_pair

run_paper_protocol(cfg)                 # identity, split-Y, IRF, Shapley, three-worlds
run_paper_protocol(cfg, contrasts=["A"])  # plus experiment-A CRN twins
```

CLI `decompile` is an alias of `paper`.

## 3. Tables the protocol writes

| Table | Estimand |
|---|---|
| Identity twin | CRN + LLM replay hits/misses |
| Split-Y | protest, PPD, R52 comply, cluster, promise broken/honored, trust logged |
| Memory IRF | Δ protest / Δ PPD / Δ R52 comply over delete-time |
| Fork moment | first round+channel where the twin leaves the factual transcript |
| Shapley vs skip | planted oracle + E003 × E052 |
| Three worlds | skip draft beat; total / omniscient / gated / hypocrisy |
| CRN contrasts | A1→A2 honor, A1→A5 delete, C3→C2 false memory, … |

Outputs: `output/reports/paper_protocol_{run_id}.md` and `.json`.

## 4. What not to run for the paper

The 19×60 independent-seed DeepSeek matrix is a cost accident and is no longer in the repo. Do not recreate it. Validity is the CRN pair `V6→V2`.

## 5. Identification caveats (write these in the paper)

- Memory IRF is an interventional analogue of an indirect effect, not a natural indirect effect.
- LLM replay identifies twins only when the prompt is unchanged. `λ` lesions rewrite prompts and will miss the cache.
- N=1 bootstrap CIs are vacuous; multi-seed is for the CRN contrast table, not for the MRI listing of one machine.
