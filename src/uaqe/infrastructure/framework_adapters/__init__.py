"""Concrete ``IFrameworkAdapter`` implementations, one per supported
source model format.

Per ``02_Folder_Structure.md`` §5, each adapter module owns exactly
one third-party ML framework import:

- ``torch_adapter.py``      -- ``.pth``, ``.pt``   (PyTorch)
- ``onnx_adapter.py``       -- ``.onnx``            (ONNX)
- ``tensorflow_adapter.py`` -- ``.pb``              (TensorFlow frozen graph)
- ``keras_adapter.py``      -- ``.h5``, ``.keras``  (Keras saved model)
- ``tflite_adapter.py``     -- ``.tflite``          (TensorFlow Lite)

No other module in ``uaqe`` may import these third-party libraries
directly (``07_Coding_Standards.md`` §14 rule 4). Adapters are wired
into the application by ``uaqe.interface.composition_root`` only
(``03_API_Specification.md`` §22 rule 1).
"""

from __future__ import annotations

from uaqe.infrastructure.framework_adapters.onnx_adapter import OnnxAdapter
from uaqe.infrastructure.framework_adapters.torch_adapter import TorchAdapter

# Optional TensorFlow / Keras support
try:
    from uaqe.infrastructure.framework_adapters.tensorflow_adapter import TensorFlowAdapter
except ImportError:
    TensorFlowAdapter = None

try:
    from uaqe.infrastructure.framework_adapters.keras_adapter import KerasAdapter
except ImportError:
    KerasAdapter = None

try:
    from uaqe.infrastructure.framework_adapters.tflite_adapter import TFLiteAdapter
except ImportError:
    TFLiteAdapter = None

__all__ = [
    "TorchAdapter",
    "OnnxAdapter",
    "TensorFlowAdapter",
    "KerasAdapter",
    "TFLiteAdapter",
]
