# Theory-Grounded Agent Variables

LabWars should not justify its internal variables by saying that they are convenient engineering features. The variables are meant to be operational proxies for known constructs in social science, organizational studies, and cognitive science.

## 1. Variable-to-theory map

| LabWars variable | Theory source | Operational meaning in LabWars |
|---|---|---|
| trust | Social exchange theory | agents update expected reciprocity and cooperation value after actions, memories, and events |
| status / reputation | Status characteristics theory; evolutionary reputation games | perceived credit, first-author probability, and reputation pressure shape public claims and private strategies |
| alliance | Structural balance theory; coalition formation | repeated positive ties and shared antagonism create clustered relationship structure |
| memory | Bounded rationality; memory reconsolidation | agents act from salient recalled episodes rather than full omniscient history |
| belief | BDI-style cognitive architecture | agents maintain subjective estimates about fairness, publishability, contribution, and threat |
| emotion | Affective appraisal theory | anxiety, anger, resentment, loyalty, and burnout modulate action tendencies |
| power / hierarchy | Authority and dependency theory | PI access, funding dependence, recommendation control, and authorship veto create compliance pressure |
| contribution ledger | Credit attribution and distributive justice | uncertain contribution shares create entitlement and authorship conflict |
| uncertainty | Bounded rationality / organizational ambiguity | ambiguous deadlines, rival threat, and unstable publishability raise exploratory and defensive behavior |

## 2. Why these variables are sufficient for v1

The v1 goal is not to model all human social cognition. It targets organizational conflict under pressure. For that narrower target, the model needs variables for:

1. who depends on whom: power and dependency;
2. who believes whom: trust and belief;
3. who thinks they deserve credit: contribution and status;
4. who remembers what: memory and reconsolidation;
5. who can form coalitions: alliance and relationship graph;
6. who is under stress: emotion and uncertainty.

These six channels are enough to generate falsifiable intervention predictions:

- deleting memory should reduce delayed grievance persistence;
- deleting trust should reduce stable coalition structure;
- deleting status should reduce credit inequality and authorship conflict;
- deleting hierarchy should reduce public/private divergence and career-hostage pressure.

## 3. Lesion definitions

| Lesion | Removed mechanism | Expected effect |
|---|---|---|
| no_memory | pre-decision recall and memory writes | weaker delayed conflict and less grievance persistence |
| no_status | credit sensitivity, authorship entitlement, reputation pressure | lower credit gap and authorship dispute |
| no_trust | trust, resentment, alliance dynamics | weaker coalition formation and lower trust fragmentation |
| no_hierarchy | PI-centered dependency and authority pressure | lower power concentration and compliance pressure |

These lesions are not arbitrary scenario changes. They are mechanism removals: the event stream and action space remain available, but a specific social-cognitive channel is neutralized.

## 4. Scientific protocol

The reviewer-facing protocol is Causal MRI of one 14-agent / 60-round trajectory:

```text
identity twin
split-Y
memory IRF at story beats
Shapley vs skip
three-worlds
optional λ lesion
A/B/C/D/V as CRN twins of one control
```

See [`docs/18-Causal-Decompiler-Paper-Protocol.md`](18-Causal-Decompiler-Paper-Protocol.md).

## 5. Claim boundary

A successful LabWars run does not prove that LLM agents are human social actors. It supports a narrower claim: under a formal social-state transition model, artificial agents can exhibit measurable organization-level dynamics, and those dynamics change under theoretically motivated mechanism lesions.