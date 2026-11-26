"""Optimization building blocks for ETROYL openEMS designs."""
from .evaluator import CandidateEvaluator, EvaluationResult
from .evaluation_store import EvaluationStore
from .candidate_selection import discrete_candidate_pool, select_next_candidates
from .engine import DiscreteSurrogateOptimizer, OptimizationSummary
from .optuna_engine import OptunaTPEOptimizer
from .parameter_space import DiscreteParameterSpace, ParameterDefinition, ParameterSpaceError
from .surrogate import GaussianProcessSurrogate, Prediction, expected_improvement

__all__ = [
    "CandidateEvaluator", "EvaluationResult", "EvaluationStore",
    "DiscreteParameterSpace", "ParameterDefinition", "ParameterSpaceError",
    "GaussianProcessSurrogate", "Prediction", "expected_improvement",
    "discrete_candidate_pool", "select_next_candidates",
    "DiscreteSurrogateOptimizer", "OptunaTPEOptimizer", "OptimizationSummary",
]
