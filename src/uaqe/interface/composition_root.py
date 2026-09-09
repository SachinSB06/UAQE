"""``CompositionRoot`` — the sole Dependency Injection / Composition
Root of the Universal AI Quantization Engine.

Per ``09_Architecture_Lock.md`` §8's locked layering
(``interface -> application, infrastructure, common``), this is the one
module in the whole codebase permitted to import concrete
``uaqe.infrastructure`` classes *and* wire them into
``uaqe.application``'s constructor-injected collaborators
(``StageFactory``, ``WorkflowBuilder``, ``PipelineExecutor``,
``WorkflowController``). No other module does this; every class those
four take is accepted purely as an interface (``uaqe.common.interfaces``)
or frozen config value object, per the constructor-injection rule
already followed throughout ``uaqe.application`` (see e.g.
``StageFactoryDependencies``'s own module docstring).

This module uses **only constructor injection**: every collaborator is
built bottom-up (repositories and adapters first, then the services
that depend on them, then the application-layer objects that depend on
those) and handed to the next constructor explicitly. There is no
global state, no singleton registry, and no third-party DI framework —
``build_workflow_controller`` is a plain function that returns a new,
independent object graph on every call.

Runtime data prerequisites
--------------------------
``ConfigRepository`` and ``HardwareProfileRepository`` are lazy: no
file is read until their first ``load_*``/``get``/``list_all`` call,
which happens the first time :meth:`CompositionRoot.build_workflow_controller`
runs. Per ``02_Folder_Structure.md`` §§7-8, this expects the following
to exist, relative to the current working directory the process is
started from (this snapshot of the project does not ship them):

- ``config/config.json``, ``config/settings.yaml``,
  ``config/quantization.json``, ``config/compression.json``,
  ``config/optimization.json`` (``06_Config_Spec.md``)
- ``hardware_profiles/**/*.json`` (``05_Hardware_Profile_Spec.md``)

If any is missing, the corresponding repository call raises a
``ConfigurationError`` — this module does not invent defaults for
them, per ``11_Implementation_Rules.md`` §5.1/§5.2's contract that
config/hardware-profile validation belongs to the repositories
themselves.

Logging bootstrap note
----------------------
``LoggerFactory.build_logger_params`` needs a ``run_id``,
``min_level``, and ``sink`` (``06_Config_Spec.md`` §2's
``settings.yaml`` ``logging`` section). ``IConfigRepository`` has no
``load_logging_config()`` method to source ``min_level``/``sink``
from, so this module reads ``settings.yaml`` directly via the same
``YamlConfigLoader`` internal collaborator ``ConfigRepository`` itself
uses — not a new class, and not a violation of
``10_Module_Development_Guide.md`` §0 rule 7 ("no module other than an
``IConfigRepository`` implementation reads a config file directly"),
since ``CompositionRoot`` is the composition boundary that rule's
surrounding text is scoped against, exactly as it is already licensed
to construct ``StructuredLogger`` directly (``11_Implementation_Rules.md``
§3.4: "No class instantiates ``StructuredLogger`` directly except
``CompositionRoot``.").

Because ``WorkflowController``/``WorkflowBuilder``/``PipelineExecutor``
share a single injected ``ILogger`` for their entire lifetime — not one
freshly minted per run — this bootstrap logger is tagged with a fixed
sentinel ``run_id`` (:data:`_BOOTSTRAP_RUN_ID`) rather than any
individual run's own ``run_id`` (which does not exist yet at wiring
time; each run's actual ``run_id`` is minted later by
``WorkflowContext``). This is a property of the existing, locked
``WorkflowController`` constructor shape (a single ``logger`` for the
controller's whole lifetime), not something invented here.
"""

from __future__ import annotations

from pathlib import Path

from uaqe.application.pipeline_executor import PipelineExecutor
from uaqe.application.stage_factory import StageFactory, StageFactoryDependencies
from uaqe.application.workflow_builder import WorkflowBuilder
from uaqe.application.workflow_controller import WorkflowController
from uaqe.common.interfaces.i_logger import ILogger
from uaqe.compression.huffman_compressor import HuffmanCompressor
from uaqe.compression.magnitude_pruner import MagnitudePruner
from uaqe.compression.rle_compressor import RLECompressor
from uaqe.compression.structured_pruner import StructuredPruner
from uaqe.compression.weight_cluster import WeightClusterCompressor
from uaqe.compression.sensitivity_aware_structured_pruner import SensitivityAwareStructuredPruner
from uaqe.compression.calibration_guided_reconstruction_pruner import CalibrationGuidedReconstructionPruner
from uaqe.exporter.binary_exporter import BinaryExporter
from uaqe.exporter.hex_exporter import HexExporter
from uaqe.exporter.mem_exporter import MemExporter
from uaqe.exporter.onnx_exporter import OnnxExporter
from uaqe.exporter.tflite_exporter import TFLiteExporter
from uaqe.infrastructure.framework_adapters import (
    KerasAdapter,
    OnnxAdapter,
    TensorFlowAdapter,
    TFLiteAdapter,
    TorchAdapter,
)
from uaqe.infrastructure.logging.logger_factory import LoggerFactory
from uaqe.infrastructure.logging.structured_logger import StructuredLogger
from uaqe.infrastructure.configuration.yaml_loader import YamlConfigLoader
from uaqe.infrastructure.repositories import ConfigRepository, HardwareProfileRepository
from uaqe.quantization.int4_quantizer import Int4Quantizer
from uaqe.quantization.int8_quantizer import Int8Quantizer
from uaqe.quantization.mixed_precision_quantizer import MixedPrecisionQuantizer
from uaqe.reports.csv_report import CsvReportRenderer
from uaqe.reports.html_report import HtmlReportRenderer
from uaqe.reports.json_report import JsonReportRenderer
from uaqe.reports.markdown_report import MarkdownReportRenderer
from uaqe.reports.pdf_report import PdfReportRenderer
from uaqe.reports.summary_report import SummaryReportRenderer

#: Default location of ``config/*.json``/``settings.yaml``, relative to
#: the process's current working directory (``02_Folder_Structure.md``
#: §7).
DEFAULT_CONFIG_DIR = "config"

#: Default location of ``hardware_profiles/**/*.json``
#: (``02_Folder_Structure.md`` §8).
DEFAULT_HARDWARE_PROFILES_DIR = "hardware_profiles"

#: Default root under which ``StructuredLogger`` addresses one
#: ``<run_id>/run.log`` per run that logs to a file
#: (``02_Folder_Structure.md`` §13).
DEFAULT_LOGS_DIR = "logs"

#: Fallback ``settings.yaml`` ``logging.min_level``/``logging.sink``
#: values used only if ``settings.yaml`` itself cannot be read (e.g. it
#: does not exist yet) — see the module docstring's "Logging bootstrap
#: note".
_DEFAULT_MIN_LEVEL = "INFO"
_DEFAULT_SINK = "stdout"

#: Sentinel ``run_id`` tag for the single ``ILogger`` shared by
#: ``WorkflowController``/``WorkflowBuilder``/``PipelineExecutor`` for
#: their whole lifetime, minted before any individual run's own
#: ``run_id`` exists — see the module docstring's "Logging bootstrap
#: note".
_BOOTSTRAP_RUN_ID = "composition-root"


class CompositionRoot:
    """Builds a fully wired :class:`~uaqe.application.workflow_controller.
    WorkflowController` from concrete infrastructure implementations.

    Every method here is a ``@staticmethod``: this class holds no
    instance state of its own and is never instantiated — it exists
    purely as a namespace for the wiring logic below.
    """

    @staticmethod
    def build_workflow_controller() -> WorkflowController:
        """Build and return a fully configured ``WorkflowController``.

        Constructs, in dependency order:

        1. The shared ``ILogger`` (:meth:`_build_logger`).
        2. ``ConfigRepository`` and ``HardwareProfileRepository``, and
           every ``*Config`` value object the stages need
           (:meth:`_load_configs`).
        3. Every concrete ``IFrameworkAdapter``, ``IQuantizationStrategy``,
           ``ICompressionStrategy``, ``IExporterBackend``, and
           ``IReportRenderer`` implementation
           (:meth:`_build_framework_adapters`,
           :meth:`_build_quantization_strategies`,
           :meth:`_build_compression_strategies`,
           :meth:`_build_exporter_backends`,
           :meth:`_build_report_renderers`).
        4. A ``StageFactoryDependencies`` bundle from all of the above,
           then ``StageFactory``, ``WorkflowBuilder``, and
           ``PipelineExecutor`` in turn.
        5. The ``WorkflowController`` composing steps 2-4's builder and
           executor with the logger from step 1.

        Returns:
            A ``WorkflowController`` ready for
            :meth:`~uaqe.application.workflow_controller.
            WorkflowController.execute` to be called against it, e.g.
            by ``src/main.py``.

        Raises:
            ConfigurationError: If any required ``config/*.json``,
                ``config/settings.yaml``, or
                ``hardware_profiles/**/*.json`` file is missing,
                malformed, or fails validation once its owning
                repository is first read from (per those repositories'
                own lazy-loading contracts).
        """
        logger = CompositionRoot._build_logger()

        config_repository = ConfigRepository(DEFAULT_CONFIG_DIR)
        hardware_repository = HardwareProfileRepository(DEFAULT_HARDWARE_PROFILES_DIR)

        quantization_config, compression_config, optimization_config, execution_config = (
            CompositionRoot._load_configs(config_repository)
        )

        dependencies = StageFactoryDependencies(
            framework_adapters=CompositionRoot._build_framework_adapters(),
            hardware_repository=hardware_repository,
            quantization_config=quantization_config,
            quantization_strategies=CompositionRoot._build_quantization_strategies(logger),
            compression_config=compression_config,
            compression_strategies=CompositionRoot._build_compression_strategies(logger),
            optimization_config=optimization_config,
            exporter_backends=CompositionRoot._build_exporter_backends(logger),
            report_renderers=CompositionRoot._build_report_renderers(),
        )

        stage_factory = StageFactory(logger, dependencies)
        builder = WorkflowBuilder(stage_factory, logger)
        executor = PipelineExecutor(logger, execution_config.on_error)

        return WorkflowController(builder, executor, logger)

    @staticmethod
    def _build_logger() -> ILogger:
        """Build the single ``ILogger`` shared by the returned
        ``WorkflowController`` and every collaborator it drives.

        See the module docstring's "Logging bootstrap note" for why
        ``settings.yaml`` is read directly here via ``YamlConfigLoader``
        rather than through ``IConfigRepository``, and why the logger
        is tagged with :data:`_BOOTSTRAP_RUN_ID` rather than a
        per-run ``run_id``.

        Returns:
            The constructed ``StructuredLogger``, as an ``ILogger``.
        """
        logger_factory = LoggerFactory(DEFAULT_LOGS_DIR)
        yaml_loader = YamlConfigLoader()
        settings_path = Path(DEFAULT_CONFIG_DIR) / "settings.yaml"

        try:
            settings = yaml_loader.load(settings_path)
        except Exception:
            settings = {}

        logging_settings = settings.get("logging", {}) if settings else {}
        min_level = logging_settings.get("min_level", _DEFAULT_MIN_LEVEL)
        sink = logging_settings.get("sink", _DEFAULT_SINK)

        params = logger_factory.build_logger_params(
            run_id=_BOOTSTRAP_RUN_ID, min_level=min_level, sink=sink
        )
        return StructuredLogger(
            run_id=params.run_id,
            min_level=params.min_level,
            sink_paths=params.sink_paths,
        )

    @staticmethod
    def _load_configs(config_repository: ConfigRepository):
        """Load every subsystem ``*Config`` value object needed to
        build a ``StageFactoryDependencies`` bundle and a
        ``PipelineExecutor``.

        Args:
            config_repository: The ``ConfigRepository`` to load from.

        Returns:
            A 4-tuple of ``(QuantizationConfig, CompressionConfig,
            OptimizationConfig, ExecutionConfig)``.
        """
        return (
            config_repository.load_quantization_config(),
            config_repository.load_compression_config(),
            config_repository.load_optimization_config(),
            config_repository.load_execution_config(),
        )

    @staticmethod
    def _build_framework_adapters():
        """Construct every concrete ``IFrameworkAdapter`` implementation.

        Returns:
            One instance each of ``OnnxAdapter``, ``TorchAdapter``,
            ``TensorFlowAdapter``, ``KerasAdapter``, and
            ``TFLiteAdapter``, for ``ModelLoader`` to select from by
            file extension.
        """
        adapters = [
            OnnxAdapter(),
            TorchAdapter(),
        ]

        if TensorFlowAdapter is not None:
            adapters.append(TensorFlowAdapter())

        if KerasAdapter is not None:
            adapters.append(KerasAdapter())

        if TFLiteAdapter is not None:
            adapters.append(TFLiteAdapter())

        return adapters

    @staticmethod
    def _build_quantization_strategies(logger: ILogger):
        """Construct every concrete ``IQuantizationStrategy`` implementation.

        Args:
            logger: Structured logging sink threaded into each strategy.

        Returns:
            Every registered ``IQuantizationStrategy``, keyed by its
            own ``name()`` — the mapping shape
            ``StageFactoryDependencies.quantization_strategies`` expects.
        """
        strategies = (
            Int8Quantizer(logger),
            Int4Quantizer(logger),
            MixedPrecisionQuantizer(logger),
        )
        return {strategy.name(): strategy for strategy in strategies}

    @staticmethod
    def _build_compression_strategies(logger: ILogger):
        """Construct every concrete ``ICompressionStrategy`` implementation.

        Args:
            logger: Structured logging sink threaded into each strategy.

        Returns:
            Every registered ``ICompressionStrategy``, keyed by its own
            ``name()`` — the mapping shape
            ``StageFactoryDependencies.compression_strategies`` expects.
        """
        strategies = (
            MagnitudePruner(logger),
            StructuredPruner(logger),
            WeightClusterCompressor(logger),
            HuffmanCompressor(logger),
            RLECompressor(logger),
            SensitivityAwareStructuredPruner(logger),
            CalibrationGuidedReconstructionPruner(logger),
        )
        return {strategy.name(): strategy for strategy in strategies}

    @staticmethod
    def _build_exporter_backends(logger: ILogger):
        """Construct every concrete ``IExporterBackend`` implementation.

        Args:
            logger: Structured logging sink threaded into each backend.

        Returns:
            One instance each of ``BinaryExporter``, ``HexExporter``,
            ``MemExporter``, ``OnnxExporter``, and ``TFLiteExporter``,
            for ``Exporter`` to select from by resolved
            ``HardwareProfile``.
        """
        return [
            BinaryExporter(logger),
            HexExporter(logger),
            MemExporter(logger),
            OnnxExporter(logger),
            TFLiteExporter(logger),
        ]

    @staticmethod
    def _build_report_renderers():
        """Construct every concrete ``IReportRenderer`` implementation.

        Returns:
            One instance each of ``CsvReportRenderer``,
            ``HtmlReportRenderer``, ``JsonReportRenderer``,
            ``MarkdownReportRenderer``, ``PdfReportRenderer``, and
            ``SummaryReportRenderer``, for ``ReportGenerator`` to render
            one ``ReportDocument`` per renderer from.
        """
        return [
            CsvReportRenderer(),
            HtmlReportRenderer(),
            JsonReportRenderer(),
            MarkdownReportRenderer(),
            PdfReportRenderer(),
            SummaryReportRenderer(),
        ]
