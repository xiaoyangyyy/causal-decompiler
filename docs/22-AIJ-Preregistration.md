# Artificial Intelligence pre-registration

This file does not replace `docs/21-AAMAS-Preregistration.md` and does not change its endpoints. The eighteen-cell grid stays as recorded.

## What the journal paper is allowed to claim

- P1 applies only to `budgeted_recover` on deterministic monotone set-oracles, plus the timed write/mediator contrast. The exactly-two family stays outside P1. Its F1 of 0 under that procedure is a boundary, not a cell in the main table.
- `budgeted_recover_pairs` is a second procedure. It may be reported only as the size-2 extension. The main table for that procedure is the exactly-three family in `src/engine/causal/heldout_order3.py`. A failure there is not followed by adding triple queries before submission.
- `external_traces/agent_trace.json` was written for this adapter. It is not a third-party agent log. The journal text may report its four layer effects only as an interface check.

## Endpoints that have not been used yet

Contrasts are differences of seed-means. A cutoff is not applied.

The live process is the recurrence of the state under the event field. Beliefs, emotions, and action tendencies update by smooth maps (logistic gates, softplus, saturating impulses, softmax). The integers 60, 24, and 18 are the lengths of the exogenous anchor tapes already written for LabWars, CrisisGrid, and the live ReleaseOps pack. They are not a stopping rule on the state, and a contrast is not accepted or rejected by how many rounds were played. The completed grid remains the recorded tape, including its 16-round ReleaseOps rows, and is not recomputed.

- LabWars model contrast, recorded cells, used the previous endpoints: `y_private` was authorship potential without the action impulse, and `y_action` was the protest count divided by 4. Those cells stay as recorded. Seed 0, llama3.2, all anchors (`fdad2b93`) differs from scripted seed 0 by 0 on `y_private`, +0.0177 on `y_public`, and 0 on `y_action`. The same seed with half the anchors deleted (`f1875e74`) differs from scripted seed 0 at that fraction by 0 on all three channels. The live endpoints are different: `y_private` is the escalation score, which includes the action impulse, and `y_action` is mean protest intensity over the draft window. Live anchor fraction scales salience of every anchor. It does not delete every other event. Do not recompute the recorded cells.
- CrisisGrid and ReleaseOps in the completed grid were scored by event-type presence. That score is only in `papers/results/aamas_preregistered.json`. Halving anchors left ReleaseOps at 0.625 under that score. Those rows are not recomputed.
- The live ReleaseOps pack is 18 rounds, three cycles of the six-stage pipeline. The completed grid remains the 16-round run and is not recomputed.
- Live pack scores are `crisisgrid_channels` and `releaseops_channels`. The reported private channel is `stranded` or `task_y`. Rates divide by the declared round horizon. CrisisGrid cancels one report for each forward, reroute, or remembered report. ReleaseOps fault is the minimum of the stale rate and the skip rate. Suppression is the product of how much of the stale count is covered by alerts and by rollbacks. `task_y` is the fault that remains after that suppression. CrisisGrid actions are `share_result` and `reroute`. ReleaseOps delay is `defer_test` and enters the skip term only; repair is `restore_release` and enters rollback only. A memory counts only when `event_ref` points at a report or a stale-config event and the write is not the automatic same-round observation. A later model run on these packs uses the live scores and is not added to the eighteen-cell file. No such run is started in this registration.
