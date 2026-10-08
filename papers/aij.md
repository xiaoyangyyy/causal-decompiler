# A decompiler for frozen trajectories, and where the guarantee stops

This is the journal draft that uses the UAI statement and one external log. It does not reprint the AAMAS grid as a theorem. LabWars appears only as the case the guarantee does not cover.

## What is identified

On a deterministic monotone set-oracle, `budgeted_recover` returns a minimal sufficient set and, for an OR, the alternative singletons. The proof obligation is the procedure itself plus the four counterexamples in `papers/uai.md`: suppressor versus knockout, non-unique OR, k-order versus pairwise AND, and the exactly-two family that the procedure misses at every budget it was given. Pairwise dividends are the wrong sign statistic for an AND of order greater than 2 (accuracy 0.400 on that family). The k-order dividend matches the planted sign on the two hundred monotone and timed worlds (accuracy 1.000).

Assumptions A1–A5 are not theorems. In particular, deleting a memory at a chosen time is not a natural indirect effect, and the search never ranges over a complete do-algebra. Certificates emit the statements with those labels.

## Pair queries stop at order 2

`budgeted_recover` is unchanged. Exactly-two of four causes remains outside it (F1 0 at budgets 16, 32, and 64).

`budgeted_recover_pairs` is a second procedure. After the monotone pass fails, it spends the remaining budget on size-2 sets. It is not folded into P1. Its main table is the exactly-three family (`src/engine/causal/heldout_order3.py`): every pair has the wrong count, so recovery fails. That failure is the boundary of the pair procedure. Triple queries are not added in order to clear it.

The registration that keeps these claims separate from the completed eighteen-cell grid is `docs/22-AIJ-Preregistration.md`.

## External log

`external_traces/agent_trace.json` was written for this adapter. It is an interface check, not a third-party agent log, and the journal claim does not treat it as one. `src/engine/causal/external_adapter.py` replays it and does not import `src.cognition`, `src.world`, or the lab simulator.

Identity replay returns Y = 1.0, matching the logged outcome. Each of the four layer operations drops Y by 1:

| Layer | ATE |
|---|---:|
| do_event on e1 | −1 |
| do_visibility, hide e1 from a | −1 |
| do_memory, delete e1 from a | −1 |
| do_behavior, set b to idle | −1 |

All four layers are necessary for this program. The minimal set has size 4. That is an AND across layers on a log the lab’s state equations do not implement. It is one program, not a sample of third-party agents. A journal claim that the same recovery holds for an untouched multi-agent codebase still needs a log whose rules were not written for this adapter.

## What the lab scenarios add, and what they do not

The pre-registered three-channel grid (`papers/aamas.md`) was scored with the presence formulas and is not recomputed. Under that score, halving anchors moves LabWars and leaves ReleaseOps at 0.625. One full-length llama3.2 LabWars cell (seed 0, all anchors, `fdad2b93`) differs from scripted seed 0 by 0 on `y_private`, +0.0177 on `y_public`, and 0 on `y_action` under the recorded endpoints (potential without the action impulse; protest count divided by 4). The same seed with every other anchor deleted (`f1875e74`) differs from scripted seed 0 at that fraction by 0 on all three channels. The live LabWars endpoints are the escalation score and the mean protest intensity. The live anchor fraction scales salience and does not delete events. Both recorded cells stay outside the eighteen-row file. CrisisGrid has no second-model row. One full-length llama3.2 ReleaseOps cell (seed 0) matched the scripted presence score at 0.625 after 3038 seconds and is not a result under the live `task_y`. Live CrisisGrid coverage cancels one report per forward, reroute, or remembered report. Live ReleaseOps `task_y` is the graded fault left after alert coverage and rollback coverage. `defer_test` enters only the skip term; `restore_release` enters only rollback.

## Reading the two papers together

Use the UAI note for P1 and the exactly-two boundary. Use this draft for the pair-query procedure and its exactly-three boundary, and for the interface check on the local external log. CrisisGrid and ReleaseOps model runs, if started, use `stranded` and `task_y`. Do not add the old cause-set F1 of 1.000 from the full-coalition enumerator. The F1 of 1.000 in the current oracle file is the monotone procedure on monotone worlds.
