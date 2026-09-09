"""``uaqe.exporter`` — turns a final, fully-optimized ``IMR`` into a
deployment artifact for its resolved ``HardwareProfile``, for the
Universal AI Quantization Engine.

This package covers the same responsibilities as the locked
``uaqe.domain.export`` module described in ``03_API_Specification.md``
§9 and ``09_Architecture_Lock.md`` §12 (plan format -> validate ->
select backend -> export -> checksum -> package), decomposed into
eleven single-responsibility files rather than the doc's one —
following the same precedent ``uaqe.quantization``, ``uaqe.compression``,
and ``uaqe.optimizer`` each already set for their own locked
counterparts (see those packages' ``__init__`` docstrings).

Pipeline (run in this order, entirely within one ``PipelineStage``):

1. :class:`~uaqe.exporter.exporter.Exporter` — the sole orchestrating
   stage, registered under context key ``"exporter"``. Resolves the
   run's final ``IMR`` and target ``HardwareProfile``, then runs steps
   2-6 below and returns the finalized
   :class:`~uaqe.exporter.exporter.DeploymentArtifact`.
2. :class:`~uaqe.exporter.export_planner.ExportPlanner` — decides which
   single ``ExportFormat`` to target, primarily from
   ``HardwareProfile.preferred_export_formats``.
3. :class:`~uaqe.exporter.export_validator.ExportValidator` — the last
   compatibility gate before anything is written: precision and
   final-size checks against the target.
4. One ``IExporterBackend`` per ``ExportFormat`` (resolved by
   :meth:`~uaqe.exporter.exporter.Exporter.select_backend` — never
   invoked directly outside of this package):

   - :class:`~uaqe.exporter.binary_exporter.BinaryExporter` — ``BIN``:
     one flat file of raw, concatenated parameter bytes.
   - :class:`~uaqe.exporter.hex_exporter.HexExporter` — ``HEX``: an
     Intel HEX text encoding of the same bytes.
   - :class:`~uaqe.exporter.mem_exporter.MemExporter` — ``MEM``: a
     ``$readmemh``-compatible text encoding of the same bytes, for FPGA
     block-RAM initialization.
   - :class:`~uaqe.exporter.tflite_exporter.TFLiteExporter` /
     :class:`~uaqe.exporter.onnx_exporter.OnnxExporter` — ``TFLITE`` /
     ``ONNX``: a self-contained, versioned interchange container (magic
     header + JSON graph metadata + parameter blob) for a downstream
     conversion step to turn into the official FlatBuffer/protobuf
     format; see each module's docstring for why.

5. :class:`~uaqe.exporter.artifact_manifest.ArtifactManifestBuilder` —
   checksums every written file into a ``manifest.json``.
6. :class:`~uaqe.exporter.package_builder.PackageBuilder` — bundles the
   artifact files and manifest into one deployment archive.

Reporting:

- :class:`~uaqe.exporter.export_report.ExportReportRenderer` summarizes
  a completed export into a Markdown/dict
  :class:`~uaqe.exporter.export_report.ExportReportDocument`, called
  separately by a reporting stage — never by ``Exporter`` itself,
  mirroring :class:`~uaqe.compression.compression_report.
  CompressionReportRenderer`'s own relationship to
  ``CompressionPlanner``.

Per ``09_Architecture_Lock.md`` §8, this package depends only on
``uaqe.common`` and ``uaqe.domain`` (for ``PipelineStage``,
``PipelineContext``, ``HardwareProfile``) — never on
``uaqe.infrastructure`` or ``uaqe.interface``. The five first-party
``IExporterBackend`` implementations are registered with
``uaqe.infrastructure.plugins.PluginRegistry`` the same way a
third-party plugin backend would be
(``10_Module_Development_Guide.md`` §18), so ``Exporter`` depends on
them only through the ``IExporterBackend`` port passed to its
constructor, never by importing a concrete backend class directly.
"""

from uaqe.exporter.artifact_manifest import (
    CHECKSUM_ALGORITHM,
    MANIFEST_SCHEMA_VERSION,
    ArtifactManifest,
    ArtifactManifestBuilder,
    ManifestFileEntry,
)
from uaqe.exporter.binary_exporter import (
    BinaryExporter,
    serialize_parameters,
    write_artifact_file,
)
from uaqe.exporter.export_planner import ExportPlan, ExportPlanner
from uaqe.exporter.export_report import ExportReportDocument, ExportReportRenderer
from uaqe.exporter.export_validator import ExportValidator
from uaqe.exporter.exporter import DeploymentArtifact, Exporter
from uaqe.exporter.hex_exporter import HexExporter
from uaqe.exporter.mem_exporter import MemExporter
from uaqe.exporter.onnx_exporter import OnnxExporter
from uaqe.exporter.package_builder import PackageBuilder
from uaqe.exporter.tflite_exporter import TFLiteExporter, build_container
from uaqe.exporter.binary_exporter import BinaryExporter
from uaqe.exporter.export_planner import ExportPlanner, ExportPlan
__all__ = [
    # exporter.py
    "Exporter",
    "DeploymentArtifact",
    # export_planner.py
    "ExportPlanner",
    "ExportPlan",
    # export_validator.py
    "ExportValidator",
    # binary_exporter.py
    "BinaryExporter",
    "serialize_parameters",
    "write_artifact_file",
    # hex_exporter.py
    "HexExporter",
    # mem_exporter.py
    "MemExporter",
    # tflite_exporter.py
    "TFLiteExporter",
    "build_container",
    # onnx_exporter.py
    "OnnxExporter",
    # artifact_manifest.py
    "ArtifactManifest",
    "ArtifactManifestBuilder",
    "ManifestFileEntry",
    "MANIFEST_SCHEMA_VERSION",
    "CHECKSUM_ALGORITHM",
    # package_builder.py
    "PackageBuilder",
    # export_report.py
    "ExportReportRenderer",
    "ExportReportDocument",
]
