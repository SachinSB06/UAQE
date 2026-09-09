import os
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class RealDatasetAdapter:
    """A model-agnostic, reusable dataset adapter that loads raw images,
    applies configurable preprocessing, and maps class folders to indices.
    """

    def __init__(
        self,
        dataset_path: str,
        class_mapping: Dict[str, int],
        preprocessing_mode: str = "rgb_0_1",
        input_shape: Tuple[int, ...] = (1, 3, 128, 128)
    ) -> None:
        """Initialize the dataset adapter.

        Args:
            dataset_path: Path to the root directory of the test dataset.
            class_mapping: Dictionary mapping directory names to class index integers.
            preprocessing_mode: Type of preprocessing to apply ('rgb_0_1', 'rgb_1_1', 'imagenet').
            input_shape: Expected model input shape e.g. [batch, channels, height, width] or [batch, height, width, channels].
        """
        self.dataset_path = dataset_path
        self.class_mapping = class_mapping
        self.preprocessing_mode = preprocessing_mode
        self.input_shape = input_shape

        # Parse layout: NCHW vs NHWC
        # For batch size 1: shape is e.g. [1, 3, 128, 128] (NCHW) or [1, 128, 128, 3] (NHWC)
        if len(input_shape) == 4:
            if input_shape[1] == 3 or input_shape[1] == 1:
                self.layout = "NCHW"
                self.channels = input_shape[1]
                self.height = input_shape[2]
                self.width = input_shape[3]
            elif input_shape[3] == 3 or input_shape[3] == 1:
                self.layout = "NHWC"
                self.channels = input_shape[3]
                self.height = input_shape[1]
                self.width = input_shape[2]
            else:
                self.layout = "NCHW"
                self.channels = 3
                self.height = input_shape[2]
                self.width = input_shape[3]
        else:
            # Default to NCHW
            self.layout = "NCHW"
            self.channels = 3
            self.height = 128
            self.width = 128

        self.samples: List[Tuple[str, int, str]] = [] # list of (file_path, class_index, class_name)
        self.corrupt_files: List[str] = []
        self.skipped_files: List[str] = []
        self.class_counts: Dict[str, int] = {}

        self._discover_dataset()

    def _discover_dataset(self) -> None:
        """Scan the dataset directory and discover valid image files."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_path}")

        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        
        # Scan folders in deterministic order
        for folder_name in sorted(os.listdir(self.dataset_path)):
            folder_path = os.path.join(self.dataset_path, folder_name)
            if not os.path.isdir(folder_path):
                continue
            
            if folder_name not in self.class_mapping:
                # Document skipped class folder
                self.class_counts[folder_name] = 0
                for f in sorted(os.listdir(folder_path)):
                    if f.lower().endswith(valid_extensions):
                        self.skipped_files.append(os.path.join(folder_path, f))
                continue
                
            class_idx = self.class_mapping[folder_name]
            self.class_counts[folder_name] = 0
            
            for file_name in sorted(os.listdir(folder_path)):
                if not file_name.lower().endswith(valid_extensions):
                    continue
                file_path = os.path.join(folder_path, file_name)
                
                # Check for basic file corruption by attempting a quick open
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    self.samples.append((file_path, class_idx, folder_name))
                    self.class_counts[folder_name] += 1
                except Exception:
                    self.corrupt_files.append(file_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Load and preprocess a single sample.

        Returns:
            Dict containing:
                - 'tensor': preprocessed numpy float32 array
                - 'label': ground truth index
                - 'class_name': name of the class folder
                - 'path': absolute image path
        """
        file_path, label, class_name = self.samples[index]
        
        try:
            img = Image.open(file_path)
            # Standard conversion: replicate single channels to RGB 3-channels
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # Bilinear resize is project standard
            img_resized = img.resize((self.width, self.height), Image.Resampling.BILINEAR)
            arr = np.array(img_resized, dtype=np.float32) # (H, W, C)
            
            # Preprocess based on selected mode
            if self.preprocessing_mode == "rgb_0_1":
                processed = arr / 255.0
            elif self.preprocessing_mode == "rgb_1_1":
                processed = (arr / 127.5) - 1.0
            elif self.preprocessing_mode == "imagenet":
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                processed = ((arr / 255.0) - mean) / std
            else:
                raise ValueError(f"Unsupported preprocessing mode: {self.preprocessing_mode}")

            # Transpose layout if model is NCHW
            if self.layout == "NCHW":
                processed = processed.transpose(2, 0, 1) # (C, H, W)
                
            return {
                "tensor": processed,
                "label": label,
                "class_name": class_name,
                "path": file_path
            }
        except Exception as e:
            # Re-raise with location details
            raise ValueError(f"Failed to process image {file_path}: {e}") from e
