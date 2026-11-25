"""Discrete optimization building blocks for ETROYL openEMS designs."""
from .evaluator import CandidateEvaluator, EvaluationResult
from .evaluation_store import EvaluationStore
from .parameter_space import DiscreteParameterSpace, ParameterDefinition, ParameterSpaceError
from .surrogate import GaussianProcessSurrogate, Prediction, expected_improvement

__all__ = [
    "CandidateEvaluator", "EvaluationResult", "EvaluationStore",
    "DiscreteParameterSpace", "ParameterDefinition", "ParameterSpaceError",
    "GaussianProcessSurrogate", "Prediction", "expected_improvement",
]
