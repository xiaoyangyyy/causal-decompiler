# When a budgeted set query recovers a minimal sufficient set

Numbers are from `papers/results/uai_oracle.json`, produced by `evaluate_benchmark`, `evaluate_baselines`, `evaluate_heldout`, and `identification_report`. The search procedure is `budgeted_recover` in `src/engine/causal/oracle.py`. It receives factor names and `query(active_set) -> Y`. It does not read planted labels.

## Statement

P1. On a deterministic monotone set-oracle, with budget at least \(2n+3\), the procedure that tests singletons and then deletes from the full set returns one minimal sufficient set. For a redundant OR it also returns every size-1 alternative. Families whose outcome is on only for a non-monotone count, such as exactly two of four causes, are outside P1.

A1–A5 in `src/engine/causal/identification.py` stay assumptions. Memory deletion along a trajectory is an interventional analogue, not a natural indirect effect. Certificates store these strings with that status (`src/engine/causal/certificate.py`).

## Four monotone or timed families

Two hundred worlds, fifty in each family, seed 11. Cause-set F1 is 1.000. False attribution is 0. The mean number of unique queries is 7.74. Alpha in {0.8, 0.9, 0.95} leaves F1 and the interaction-sign accuracy at 1.000 on a smaller draw of four worlds per family.

The interaction sign for an AND is the k-order Harsanyi dividend of the cause set. For three causes that dividend is the total effect, while the dividend of the first pair is 0. Pairwise sign accuracy on the AND family is 0.400. That is the quantity that used to be reported as if it were the interaction sign. It is now a separate column.

Delayed mediation is not a static coalition. The early cause writes a mediator at time 0. Deleting the early cause after that write leaves Y unchanged. Deleting the mediator at the horizon sets Y to 0. All fifty timed worlds recover that pattern (F1 1.000 on the timing test, four queries).

## Baselines that see the same oracle

On the non-timed worlds in an eight-per-family draw:

| Procedure | F1 | False attribution |
|---|---:|---:|
| budgeted recovery | 1.000 | 0.000 |
| mediation (best positive pair dividend) | 0.550 | 0.333 |
| single-factor knockout | 0.508 | 0.333 |

Character overlap and a hash of the world id are not baselines. They remain in the result file under `not_baselines` so the exclusion is checkable. Knockout’s false attributions include the suppressor counterexample: from the full set, removing the suppressor changes Y and removing the cause does not, so knockout returns the suppressor. Recovery returns the cause.

## Counterexamples required by the statement

- Suppressor. Knockout returns `suppressor`. Recovery returns `cause`.
- Redundant OR of three causes. Recovery returns `c0` and lists alternatives `c0`, `c1`, `c2`. The first name is not a unique cause.
- AND of order 3. Pairwise dividend 0. k-order dividend 1. The sum of single deletions is 3, against a total effect of 1. Recovery returns `{c0, c1, c2}`.
- Exactly-two of four causes, twenty worlds, held out of the search module. Default budget 16, and the same procedure at budgets 32 and 64: F1 0, failure rate 1. Extra budget does not help, because the procedure never proposes the size-2 queries this family needs.

## Scope

P1 is a claim about this procedure on monotone deterministic oracles and about the timed write/mediator contrast. It is not a claim that the same queries identify a language-model policy, and it is not the LabWars table.
