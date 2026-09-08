"""Causal Decompiler package.

Public MRI entry is CausalDecompiler. Twins and CausalOps are the algebra.
Paired ATE lives on estimands.paired_effect, not a multi-seed grid.
"""

from .algebra import (
    CausalOp,
    delete_memory,
    lesion,
    observe_lock,
    override_event,
    resample,
    set_policy_lambda,
    skip_event,
)
from .decompiler import CausalDecompiler, CausalMRIReport
from .twin import identity_holds, load_factual, run_factual, run_replay, run_twin, sim_config_from_log

__all__ = [
    "CausalDecompiler",
    "CausalMRIReport",
    "CausalOp",
    "delete_memory",
    "identity_holds",
    "lesion",
    "observe_lock",
    "override_event",
    "resample",
    "run_factual",
    "run_replay",
    "run_twin",
    "sim_config_from_log",
    "load_factual",
    "set_policy_lambda",
    "skip_event",
]
