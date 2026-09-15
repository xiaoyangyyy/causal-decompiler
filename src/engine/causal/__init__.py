"""Causal Decompiler package.

Public MRI entry is CausalDecompiler. Twins and CausalOps are the algebra.
Paired ATE lives on estimands.paired_effect, not a multi-seed grid.
"""

from .algebra import (
    CausalOp,
    delete_memory,
    do_behavior,
    do_belief,
    do_event,
    do_memory,
    do_presence,
    do_private_public,
    do_visibility,
    lesion,
    observe_lock,
    override_event,
    resample,
    set_policy_lambda,
    skip_event,
)
from .certificate import certificates_from_report
from .decompiler import CausalDecompiler, CausalMRIReport
from .ir import SocialCausalIR, extract_ir
from .mechanisms import evaluate_benchmark, parameterized_worlds
from .search import candidates_for_outcome, harsanyi_from_shapley, minimal_effect_recovery_set, minimal_sufficient_set
from .twin import identity_holds, load_factual, run_factual, run_replay, run_twin, sim_config_from_log
from .worlds import evaluate_planted_worlds, planted_social_worlds

__all__ = [
    "CausalDecompiler",
    "CausalMRIReport",
    "CausalOp",
    "SocialCausalIR",
    "candidates_for_outcome",
    "certificates_from_report",
    "delete_memory",
    "do_behavior",
    "do_belief",
    "do_event",
    "do_memory",
    "do_presence",
    "do_private_public",
    "do_visibility",
    "evaluate_benchmark",
    "evaluate_planted_worlds",
    "extract_ir",
    "harsanyi_from_shapley",
    "identity_holds",
    "lesion",
    "minimal_effect_recovery_set",
    "minimal_sufficient_set",
    "observe_lock",
    "override_event",
    "parameterized_worlds",
    "planted_social_worlds",
    "resample",
    "run_factual",
    "run_replay",
    "run_twin",
    "sim_config_from_log",
    "load_factual",
    "set_policy_lambda",
    "skip_event",
]
