"""Per-trial execution engine — real hardware or analytical simulation.

Per ``11_Implementation_Rules.md`` §7.4 ("Benchmark Strategy"),
``Benchmarker`` delegates to a real-hardware execution path or a
``HardwareProfile``-derived analytical simulation path per
``benchmark.json``'s ``use_real_hardware_if_available``/
``simulation_fallback_enabled`` flags, and "both paths implement the
same internal strategy interface so ``Benchmarker``'s aggregation
logic is identical regardless of which path ran." :class:`TrialRunner`
is that internal strategy interface; :class:`RealHardwareTrialRunner`
and :class:`SimulationTrialRunner` are its two implementations, and
:class:`RuntimeBenchmark` is the collaborator that selects between them
and drives the warmup/trial loop, exactly mirroring how ``*Engine``
classes elsewhere in this codebase (e.g.
:class:`~uaqe.compression.compression_engine.CompressionEngine`)
delegate to a resolved strategy without containing strategy-specific
branching logic themselves.

Per ``10_Module_Development_Guide.md`` §10 ("``Benchmarker``
device-detection mechanism is infrastructure-specific per backend
[...] the domain-level ``Benchmarker`` calls an injected capability, it
does not itself probe hardware"), :class:`RealHardwareTrialRunner`
never implements device probing/execution itself: it accepts two
optional injected callables (``device_probe``, ``device_execute``),
supplied by the ``CompositionRoot`` from whichever infrastructure
backend implements them. With no callables injected (the default in
this codebase today, since no such infrastructure backend is
implemented yet — the same scope note
:mod:`uaqe.hardware.latency_estimator` and
:mod:`uaqe.hardware.power_estimator` already carry), this runner is
simply never available, and every benchmark run falls back to
:class:`SimulationTrialRunner`.

Simulation model: :class:`SimulationTrialRunner` wraps
:class:`~uaqe.hardware.latency_estimator.LatencyEstimator` for a single
point estimate, then applies a small, seeded per-trial jitter (mirroring
the seeded-``random.Random`` precedent
:class:`~uaqe.compression.weight_cluster.WeightClusterCompressor`
already sets for reproducible randomized behavior in this codebase) so
that repeated trials produce a non-degenerate distribution for
:class:`~uaqe.benchmark.latency_benchmark.LatencyBenchmark` to compute
meaningful ``latency_ms_p50``/``latency_ms_p99`` percentiles from,
rather than an identical point value on every trial.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from uaqe.benchmark.benchmark_result import RuntimeBenchmarkScore
from uaqe.common.exceptions import BenchmarkError
from uaqe.common.imr import IMR
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.hardware.latency_estimator import LatencyEstimator

if TYPE_CHECKING:
    from uaqe.domain.hardware_manager import HardwareProfile
    from uaqe.exporter.exporter import DeploymentArtifact

#: Seed for :class:`SimulationTrialRunner`'s per-trial jitter, mirroring
#: :mod:`uaqe.compression.weight_cluster`'s
#: ``random.Random(_KMEANS_RANDOM_SEED)`` precedent for reproducible
#: randomized behavior.
_SIMULATION_JITTER_SEED: int = 20260727

#: The simulated per-trial latency's jitter half-width, as a fraction of
#: the point estimate (e.g. ``0.10`` => each trial is uniformly
#: distributed within +/-10% of the point estimate).
_SIMULATION_JITTER_FRACTION: float = 0.10


class TrialRunner(ABC):
    """The internal strategy interface both execution paths implement,
    per ``11_Implementation_Rules.md`` §7.4.
    """

    @abstractmethod
    def is_available(self, profile: "HardwareProfile") -> bool:
        """Report whether this runner can execute a trial for
        ``profile`` right now.

        Args:
            profile: The candidate deployment target.

        Returns:
            ``True`` if :meth:`run_trial` can be called for this
            profile.
        """
        raise NotImplementedError

    @abstractmethod
    def run_trial(
        self,
        artifact: "DeploymentArtifact",
        profile: "HardwareProfile",
        imr: Optional[IMR],
    ) -> float:
        """Execute (or simulate) one inference and return its latency.

        Args:
            artifact: The ``DeploymentArtifact`` under benchmark.
            profile: The resolved deployment target.
            imr: The ``IMR`` the artifact was exported from, if
                available; only the simulation path needs it.

        Returns:
            The trial's latency, in milliseconds.
        """
        raise NotImplementedError


class RealHardwareTrialRunner(TrialRunner):
    """Executes a trial on a physical device via an injected,
    infrastructure-specific capability.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            runner operates silently.
    """

    def __init__(
        self,
        logger: Optional[ILogger] = None,
        *,
        device_probe: Optional[Callable[["HardwareProfile"], bool]] = None,
        device_execute: Optional[
            Callable[["DeploymentArtifact", "HardwareProfile"], float]
        ] = None,
    ) -> None:
        """Initialize a ``RealHardwareTrialRunner``.

        Args:
            logger: Optional structured logging sink.
            device_probe: An injected capability reporting whether a
                physical device matching ``profile`` is currently
                attached. ``None`` (the default) means no such
                capability is wired in, so :meth:`is_available` always
                reports ``False``.
            device_execute: An injected capability that runs one
                inference on the attached device and returns its
                latency, in milliseconds. Required whenever
                ``device_probe`` is supplied.
        """
        self.logger: Optional[ILogger] = logger
        self._device_probe = device_probe
        self._device_execute = device_execute

    def is_available(self, profile: "HardwareProfile") -> bool:
        """Report whether an injected ``device_probe`` finds a physical
        device attached for ``profile``.

        Args:
            profile: The candidate deployment target.

        Returns:
            ``False`` if no ``device_probe`` was injected; otherwise
            ``device_probe(profile)``.
        """
        if self._device_probe is None:
            return False
        return self._device_probe(profile)

    def run_trial(
        self,
        artifact: "DeploymentArtifact",
        profile: "HardwareProfile",
        imr: Optional[IMR],
    ) -> float:
        """Run one on-device inference via the injected
        ``device_execute`` capability.

        Args:
            artifact: The ``DeploymentArtifact`` under benchmark.
            profile: The resolved deployment target.
            imr: Unused by this runner; on-device execution measures
                the artifact directly.

        Returns:
            The trial's latency, in milliseconds.

        Raises:
            BenchmarkError: If no ``device_execute`` capability was
                injected despite :meth:`is_available` reporting
                ``True``.
        """
        if self._device_execute is None:
            raise BenchmarkError(
                "RealHardwareTrialRunner.is_available reported True but "
                "no device_execute capability was injected.",
                code="BENCHMARK_MISSING_DEVICE_EXECUTOR",
            )
        return self._device_execute(artifact, profile)


class SimulationTrialRunner(TrialRunner):
    """Simulates a trial from a ``HardwareProfile``-derived analytical
    latency estimate, with seeded per-trial jitter.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            runner operates silently.
    """

    def __init__(
        self,
        logger: Optional[ILogger] = None,
        estimator: Optional[LatencyEstimator] = None,
    ) -> None:
        """Initialize a ``SimulationTrialRunner``.

        Args:
            logger: Optional structured logging sink.
            estimator: The underlying estimator to delegate the point
                estimate to; a fresh
                :class:`~uaqe.hardware.latency_estimator.
                LatencyEstimator` is constructed if omitted.
        """
        self.logger: Optional[ILogger] = logger
        self._estimator = estimator or LatencyEstimator(logger)
        self._rng = random.Random(_SIMULATION_JITTER_SEED)

    def is_available(self, profile: "HardwareProfile") -> bool:
        """Report whether this runner can simulate a trial for
        ``profile``.

        Args:
            profile: The candidate deployment target; unused, since the
                simulation path is always available given an ``IMR``
                (checked by the caller, per :meth:`run_trial`).

        Returns:
            Always ``True``. Whether an ``IMR`` is actually available
            is checked by :class:`RuntimeBenchmark` before selecting
            this runner, not by this method.
        """
        return True

    def run_trial(
        self,
        artifact: "DeploymentArtifact",
        profile: "HardwareProfile",
        imr: Optional[IMR],
    ) -> float:
        """Simulate one trial's latency from ``imr``'s analytical
        estimate, jittered by a fixed, seeded fraction.

        Args:
            artifact: Unused by this runner; the simulation is derived
                from ``imr``, not the exported artifact.
            profile: The candidate deployment target.
            imr: The model to estimate; required for this runner.

        Returns:
            The simulated trial's latency, in milliseconds.

        Raises:
            BenchmarkError: If ``imr`` is ``None``.
        """
        if imr is None:
            raise BenchmarkError(
                "SimulationTrialRunner.run_trial requires an IMR; none "
                "was supplied for this run.",
                code="BENCHMARK_NO_IMR_FOR_SIMULATION",
            )
        point_estimate_ms = self._estimator.estimate(imr, profile).total_latency_ms
        jitter = 1.0 + self._rng.uniform(
            -_SIMULATION_JITTER_FRACTION, _SIMULATION_JITTER_FRACTION
        )
        return max(point_estimate_ms * jitter, 0.0)


class RuntimeBenchmark:
    """Drives the warmup/trial loop against whichever
    :class:`TrialRunner` is selected for a run.

    Attributes:
        logger: Optional structured logging sink; if omitted, this
            benchmark operates silently.
    """

    def __init__(
        self,
        logger: Optional[ILogger] = None,
        real_runner: Optional[RealHardwareTrialRunner] = None,
        simulation_runner: Optional[SimulationTrialRunner] = None,
    ) -> None:
        """Initialize a ``RuntimeBenchmark``.

        Args:
            logger: Optional structured logging sink.
            real_runner: The real-hardware execution strategy; a fresh
                :class:`RealHardwareTrialRunner` (with no injected
                device capability, so never available) is constructed
                if omitted.
            simulation_runner: The analytical simulation strategy; a
                fresh :class:`SimulationTrialRunner` is constructed if
                omitted.
        """
        self.logger: Optional[ILogger] = logger
        self._real_runner = real_runner or RealHardwareTrialRunner(logger)
        self._simulation_runner = simulation_runner or SimulationTrialRunner(logger)

    def run(
        self,
        artifact: "DeploymentArtifact",
        profile: "HardwareProfile",
        imr: Optional[IMR],
        *,
        trials: int,
        warmup_trials: int,
        timeout_seconds_per_trial: int,
        attempt_real_hardware: bool,
        allow_simulation_fallback: bool,
    ) -> RuntimeBenchmarkScore:
        """Run ``warmup_trials`` (discarded) then ``trials`` (recorded)
        against the selected execution path.

        Args:
            artifact: The ``DeploymentArtifact`` under benchmark.
            profile: The resolved deployment target.
            imr: The ``IMR`` the artifact was exported from, if
                available.
            trials: The number of recorded trials to run.
            warmup_trials: The number of warmup trials to run and
                discard before recording begins.
            timeout_seconds_per_trial: The maximum latency, in seconds,
                a single trial may report before this method raises
                ``BenchmarkError``.
            attempt_real_hardware: Whether to first check
                :attr:`_real_runner`'s availability for ``profile``.
            allow_simulation_fallback: Whether :attr:`_simulation_runner`
                may be used when the real-hardware path is unavailable
                or was not attempted.

        Returns:
            The finalized :class:`~uaqe.benchmark.benchmark_result.
            RuntimeBenchmarkScore`.

        Raises:
            BenchmarkError: If no execution path is available/allowed,
                or if any trial's latency exceeds
                ``timeout_seconds_per_trial``.
        """
        runner, used_real_hardware, assumptions = self._select_runner(
            profile,
            attempt_real_hardware=attempt_real_hardware,
            allow_simulation_fallback=allow_simulation_fallback,
        )

        timeout_ms = timeout_seconds_per_trial * 1000.0

        for trial_index in range(warmup_trials):
            self._run_one_trial(
                runner, artifact, profile, imr, timeout_ms, trial_index, warmup=True
            )

        per_trial_latency_ms = []
        for trial_index in range(trials):
            latency_ms = self._run_one_trial(
                runner, artifact, profile, imr, timeout_ms, trial_index, warmup=False
            )
            per_trial_latency_ms.append(latency_ms)

        if self.logger is not None:
            self.logger.info(
                "Benchmark trials complete.",
                hardware_profile_id=profile.profile_id,
                trials=trials,
                warmup_trials=warmup_trials,
                used_real_hardware=used_real_hardware,
            )

        return RuntimeBenchmarkScore(
            used_real_hardware=used_real_hardware,
            trials_run=trials,
            warmup_trials=warmup_trials,
            per_trial_latency_ms=per_trial_latency_ms,
            assumptions=assumptions,
        )

    def _select_runner(
        self,
        profile: "HardwareProfile",
        *,
        attempt_real_hardware: bool,
        allow_simulation_fallback: bool,
    ) -> Tuple[TrialRunner, bool, List[str]]:
        """Select which :class:`TrialRunner` this run should use.

        Args:
            profile: The candidate deployment target.
            attempt_real_hardware: Whether to first check
                :attr:`_real_runner`'s availability.
            allow_simulation_fallback: Whether the simulation path may
                be used as a fallback.

        Returns:
            A tuple of the selected runner, whether it is the
            real-hardware runner, and the assumptions list to carry
            into the result.

        Raises:
            BenchmarkError: If neither path is available/allowed.
        """
        if attempt_real_hardware and self._real_runner.is_available(profile):
            return (
                self._real_runner,
                True,
                [f"Benchmarked on physical device for profile {profile.profile_id!r}."],
            )

        if allow_simulation_fallback:
            return (
                self._simulation_runner,
                False,
                [
                    "No physical device attached (or use_real_hardware_if_"
                    "available=False); used the HardwareProfile-derived "
                    "analytical simulation path with seeded per-trial "
                    "jitter, not a measured on-device result.",
                ],
            )

        raise BenchmarkError(
            f"No benchmark execution path available for profile "
            f"{profile.profile_id!r}: no physical device attached and "
            "simulation_fallback_enabled is False (or no IMR was "
            "available for simulation).",
            code="BENCHMARK_NO_EXECUTION_PATH",
        )

    def _run_one_trial(
        self,
        runner: TrialRunner,
        artifact: "DeploymentArtifact",
        profile: "HardwareProfile",
        imr: Optional[IMR],
        timeout_ms: float,
        trial_index: int,
        *,
        warmup: bool,
    ) -> float:
        """Run a single trial and enforce ``timeout_ms``.

        Args:
            runner: The selected execution strategy.
            artifact: The ``DeploymentArtifact`` under benchmark.
            profile: The resolved deployment target.
            imr: The ``IMR`` the artifact was exported from, if
                available.
            timeout_ms: The maximum allowed latency, in milliseconds.
            trial_index: This trial's 0-indexed position, for logging.
            warmup: Whether this is a discarded warmup trial.

        Returns:
            The trial's latency, in milliseconds.

        Raises:
            BenchmarkError: If the trial's latency exceeds
                ``timeout_ms``.
        """
        wall_start = time.monotonic()
        latency_ms = runner.run_trial(artifact, profile, imr)
        wall_ms = (time.monotonic() - wall_start) * 1000.0

        if latency_ms > timeout_ms:
            raise BenchmarkError(
                f"Trial {trial_index} latency of {latency_ms:.2f}ms exceeds "
                f"timeout_seconds_per_trial ({timeout_ms:.0f}ms) for "
                f"profile {profile.profile_id!r}.",
                code="BENCHMARK_TRIAL_TIMEOUT",
            )

        if self.logger is not None:
            self.logger.debug(
                "Benchmark trial complete.",
                trial_index=trial_index,
                warmup=warmup,
                latency_ms=latency_ms,
                wall_ms=wall_ms,
            )

        return latency_ms
