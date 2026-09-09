from __future__ import annotations
from typing import List, Dict
from uaqe.common.interfaces.i_exporter_backend import IExporterBackend
from uaqe.common.types import ExportFormat
from uaqe.common.exceptions import ExportError

class ExporterFactory:
    """Resolves the concrete IExporterBackend implementation corresponding to a selected target runtime."""
    
    def __init__(self, backends: List[IExporterBackend]) -> None:
        self._backends: Dict[ExportFormat, IExporterBackend] = {
            backend.EXPORT_FORMAT: backend for backend in backends  # type: ignore[attr-defined]
        }
        
    def get_exporter(self, runtime: str) -> IExporterBackend:
        if runtime == "onnxruntime":
            fmt = ExportFormat.ONNX
        elif runtime in ("tflite-runtime", "tflite-micro"):
            fmt = ExportFormat.TFLITE
        elif runtime == "torch":
            raise ExportError("Runtime 'torch' is not yet implemented.", code="RUNTIME_NOT_IMPLEMENTED")
        elif runtime == "bare-metal-hdl":
            # FPGA bare-metal targets fallback to BIN format.
            fmt = ExportFormat.BIN
        else:
            raise ExportError(f"Unknown runtime '{runtime}'.", code="UNKNOWN_RUNTIME")
            
        backend = self._backends.get(fmt)
        if not backend:
            raise ExportError(
                f"No registered exporter backend for ExportFormat.{fmt.name}. "
                f"Available formats: {[f.name for f in self._backends]}.",
                code="EXPORT_MISSING_BACKEND"
            )
        return backend
