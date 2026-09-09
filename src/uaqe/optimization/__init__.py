"""UAQE Autonomous Optimization Package.

Exports the core multi-objective autonomous optimization framework:
- OptimizationController: Central autonomous controller
- CandidateGenerator, OptimizationCandidate: Capability-driven candidate synthesis
- CandidateEvaluator, CandidateEvaluationResult: Real empirical candidate execution
- ObjectiveFunction, ObjectiveWeights: Profile-driven multi-objective scoring
- StoppingPolicy, StoppingReason: Autonomous search termination policies
- SearchManager: History, candidate ranking, and Pareto frontier tracking
- AccuracySafetyPolicy, AccuracyClassification: Hard accuracy safety boundaries
"""

from .accuracy_safety_policy import AccuracySafetyPolicy, AccuracyClassification
from .objective_function import ObjectiveFunction, ObjectiveWeights
from .stopping_policy import StoppingPolicy, StoppingReason
from .candidate_generator import CandidateGenerator, OptimizationCandidate
from .candidate_evaluator import CandidateEvaluator, CandidateEvaluationResult
from .search_manager import SearchManager
from .optimization_controller import OptimizationController

__all__ = [
    "AccuracySafetyPolicy",
    "AccuracyClassification",
    "ObjectiveFunction",
    "ObjectiveWeights",
    "StoppingPolicy",
    "StoppingReason",
    "CandidateGenerator",
    "OptimizationCandidate",
    "CandidateEvaluator",
    "CandidateEvaluationResult",
    "SearchManager",
    "OptimizationController"
]
