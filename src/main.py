#!/usr/bin/env python3
"""CLI entry point for the Universal AI Quantization Engine (UAQE).

Usage::

    python src/main.py --model models/model.onnx --hardware esp32 --output output/

Responsibilities (nothing else):

1. Parse command-line arguments with ``argparse``.
2. Build a ``WorkflowController`` via
   :class:`~uaqe.interface.composition_root.CompositionRoot`.
3. Build a ``WorkflowConfig`` from the parsed arguments, using the
   project's existing configuration/value-object classes.
4. Execute the pipeline via ``WorkflowController.execute``.
5. Print a clean execution summary.
6. Handle exceptions and return the appropriate process exit code.

No path is hardcoded: ``--model``, ``--hardware``, and ``--output`` are
all supplied by the caller. ``--model`` and ``--hardware`` map directly
onto ``WorkflowConfig.model_path``/``hardware_profile_id``.

Known gap regarding ``--output``: ``WorkflowConfig`` has no
``output_dir`` field, and this codebase snapshot's ``StageFactory``
constructs ``Exporter``/``ReportGenerator`` without ever calling
``WorkflowConfig.override(...)`` for an output path — each exporter
backend and ``ReportGenerator`` instead falls back to its own
constructor default (e.g. ``Exporter``'s ``"outputs/exports"``,
``ReportGenerator``'s own default). There is therefore no existing
run-level hook this file can invoke to actually route ``--output``'s
value into where artifacts get written, without adding a new field to
a frozen, locked ``WorkflowConfig`` dataclass or changing
``StageFactory``'s stage-construction lambdas — both out of scope here
per this task's "do not invent code / do not redesign" constraint.
``--output`` is accepted and echoed in the printed summary so the
requested CLI contract (``--model``/``--hardware``/``--output``) is
honored, but it does not currently change where the run's
``Exporter``/``ReportGenerator`` write their files; those still use
their own built-in defaults. Wiring ``--output`` all the way through
is a ``StageFactory``/``WorkflowConfig`` change for a separate,
explicitly scoped pass.

Exit codes:

- ``0``: The run completed with every stage succeeding.
- ``1``: The run completed but at least one stage failed under its
  ``PipelineErrorPolicy`` (a "clean" failure — the pipeline itself
  did not crash).
- ``2``: A ``UAQEError`` was raised before or during the run (e.g.
  invalid arguments, missing config files, a stage construction
  failure) — the pipeline could not complete at all.
- ``130``: Interrupted by the user (``Ctrl+C``).
"""

from __future__ import annotations

import argparse
import sys

from uaqe.application.execution_summary import ExecutionSummary
from uaqe.application.workflow_config import WorkflowConfig
from uaqe.common.exceptions import UAQEError
from uaqe.interface.composition_root import CompositionRoot


def parse_args(argv) -> argparse.Namespace:
    """Parse command-line arguments for a single UAQE run.

    Args:
        argv: The argument list to parse (excluding the program name),
            e.g. ``sys.argv[1:]``.

    Returns:
        The parsed ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        prog="uaqe",
        description=(
            "Universal AI Quantization Engine — quantize, compress, "
            "optimize, and export a model for a target embedded/FPGA/"
            "Raspberry Pi hardware profile."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        dest="model_path",
        help="Filesystem path to the source model file to load.",
    )
    parser.add_argument(
        "--hardware",
        required=True,
        dest="hardware_profile_id",
        help=(
            "HardwareProfile.profile_id to target for this run, e.g. "
            "'esp32', 'artix7', 'raspberrypi4'."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        dest="output_dir",
        help="Directory exported artifacts and reports should be written under.",
    )
    parser.add_argument(
        "--calibration-dataset",
        dest="calibration_dataset_path",
        default=None,
        help=(
            "Optional filesystem path to a calibration dataset. If "
            "omitted, calibration runs from layer weights alone."
        ),
    )
    parser.add_argument(
        "--evaluation-dataset",
        dest="evaluation_dataset_path",
        default=None,
        help=(
            "Optional filesystem path to a labeled evaluation dataset. "
            "If omitted, accuracy-delta evaluation is disabled for this run."
        ),
    )
    parser.add_argument(
        "--run-id-prefix",
        dest="run_id_prefix",
        default=None,
        help="Optional override for this run's minted run_id prefix.",
    )
    parser.add_argument(
        "--runtime",
        dest="runtime",
        default=None,
        help="Target runtime override, e.g. 'onnxruntime', 'tflite-runtime', 'torch'.",
    )
    parser.add_argument(
        "--config-overrides",
        dest="config_overrides",
        default=None,
        help="JSON or Python dict string for run overrides.",
    )
    return parser.parse_args(argv)


def build_workflow_config(args: argparse.Namespace) -> WorkflowConfig:
    """Build this run's ``WorkflowConfig`` from parsed CLI arguments.

    Args:
        args: The ``argparse.Namespace`` returned by :func:`parse_args`.

    Returns:
        The assembled ``WorkflowConfig``. Not yet validated — validation
        happens inside ``WorkflowBuilder.build`` when the run is executed.
    """
    import json
    overrides = {}
    if getattr(args, "config_overrides", None):
        try:
            # Replace single quotes with double quotes for valid JSON
            raw_overrides = args.config_overrides.replace("'", '"')
            overrides = json.loads(raw_overrides)
        except Exception:
            import ast
            try:
                overrides = ast.literal_eval(args.config_overrides)
            except Exception as e:
                raise UAQEError(
                    f"Failed to parse config-overrides: {e}. Expected a JSON or dict string.",
                    code="CONFIG_PARSING_FAILED"
                )

    return WorkflowConfig(
        model_path=args.model_path,
        hardware_profile_id=args.hardware_profile_id,
        runtime=args.runtime,
        calibration_dataset_path=args.calibration_dataset_path,
        evaluation_dataset_path=args.evaluation_dataset_path,
        config_overrides=overrides,
        run_id_prefix=args.run_id_prefix,
    )


def main(argv=None) -> int:
    """Run one end-to-end UAQE workflow from CLI arguments.

    Args:
        argv: The argument list to parse (excluding the program name).
            Defaults to ``sys.argv[1:]`` when ``None``.

    Returns:
        The process exit code (see module docstring for the mapping).
    """
    args = parse_args(sys.argv[1:] if argv is None else argv)
    config = build_workflow_config(args)

    try:
        controller = CompositionRoot.build_workflow_controller()
        result = controller.execute(config)
    except UAQEError as exc:
        print(f"error: {exc.code}: {exc.message}", file=sys.stderr)
        if exc.remediation_hint:
            print(f"hint: {exc.remediation_hint}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130

    summary = ExecutionSummary.from_result(result)
    print(summary.describe())
    print(f"requested output directory: {args.output_dir} (see module docstring)")
    if summary.failed_stage_names:
        print(f"failed stages: {', '.join(summary.failed_stage_names)}")
    if summary.error_codes:
        print(f"error codes: {', '.join(summary.error_codes)}")
    if summary.output_paths:
        print("outputs:")
        for path in summary.output_paths:
            print(f"  {path}")

    return 0 if summary.success else 1


if __name__ == "__main__":
    sys.exit(main())
