"""LabWars experiment runners: paper MRI, CRN contrasts, A–D/V conditions."""

from .conditions import EXPERIMENT_MATRIX, build_sim_config, get_condition
from .paper_contrasts import run_crn_pair, run_paper_contrasts
from .paper_protocol import PaperProtocolResult, run_paper_protocol
from .report import generate_report
from .runner import run_single

__all__ = [
    "EXPERIMENT_MATRIX",
    "PaperProtocolResult",
    "build_sim_config",
    "generate_report",
    "get_condition",
    "run_crn_pair",
    "run_paper_contrasts",
    "run_paper_protocol",
    "run_single",
]
