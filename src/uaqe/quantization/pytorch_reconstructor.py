import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Tuple, Optional, List
import onnx
from onnx import numpy_helper

class MobileNetV3Reconstructor:
    """Reconstructs PyTorch MobileNetV3-Small models from existing ONNX initializers,
    supporting both 10-class reference models for numerical verification and
    9-class models for training and fine-tuning on semiconductor defect datasets.
    """

    DEFAULT_9CLASS_MAPPING = {
        "0": "bridge",
        "1": "clean",
        "2": "cmp",
        "3": "crack",
        "4": "opens",
        "5": "other",
        "6": "particle",
        "7": "scratch",
        "8": "vias"
    }

    def __init__(self, onnx_model_path: str = "src/models/mobilenetv3_sem.onnx") -> None:
        """Initialize reconstructor with path to reference ONNX model."""
        self.onnx_model_path = onnx_model_path
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"Reference ONNX model not found: {onnx_model_path}")
        
        self.onnx_model = onnx.load(onnx_model_path)
        self.initializers: Dict[str, np.ndarray] = {
            init.name: numpy_helper.to_array(init) for init in self.onnx_model.graph.initializer
        }
        self.conv_nodes = [n for n in self.onnx_model.graph.node if n.op_type == "Conv"]

    def _get_module_by_name(self, model: nn.Module, module_name: str) -> nn.Module:
        parts = module_name.split(".")
        parent = model
        for p in parts:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        return parent

    def build_10class_reference_model(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Constructs a 10-class PyTorch MobileNetV3-Small reference model and transfers
        all compatible weights and biases from the ONNX graph for numerical equivalence testing.
        """
        py_model = models.mobilenet_v3_small(num_classes=10)
        py_conv_named = [(name, m) for name, m in py_model.named_modules() if isinstance(m, nn.Conv2d)]
        
        transfer_log: List[Dict[str, Any]] = []
        
        # 1. Map 52 Conv layers
        for i, (c_node, (py_name, py_conv)) in enumerate(zip(self.conv_nodes, py_conv_named)):
            w_name = c_node.input[1]
            b_name = c_node.input[2]
            w = torch.from_numpy(self.initializers[w_name].copy())
            b = torch.from_numpy(self.initializers[b_name].copy())
            
            if py_conv.bias is not None:
                py_conv.weight.data.copy_(w)
                py_conv.bias.data.copy_(b)
                transfer_log.append({
                    "layer_index": i,
                    "onnx_node": c_node.name,
                    "onnx_w_name": w_name,
                    "onnx_b_name": b_name,
                    "py_module": py_name,
                    "w_shape": list(w.shape),
                    "b_shape": list(b.shape),
                    "type": "Conv2d_with_bias"
                })
            else:
                py_conv.weight.data.copy_(w)
                # Find corresponding BatchNorm2d right after Conv2d in the sequential container
                parts = py_name.split(".")
                parent = py_model
                for p in parts[:-1]:
                    parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
                idx = int(parts[-1])
                bn = parent[idx + 1]
                if isinstance(bn, nn.BatchNorm2d):
                    bn.weight.data.fill_(1.0)
                    bn.bias.data.copy_(b)
                    bn.running_mean.data.zero_()
                    bn.running_var.data.fill_(1.0)
                    bn.eps = 0.0

                transfer_log.append({
                    "layer_index": i,
                    "onnx_node": c_node.name,
                    "onnx_w_name": w_name,
                    "onnx_b_name": b_name,
                    "py_module": py_name,
                    "w_shape": list(w.shape),
                    "b_shape": list(b.shape),
                    "type": "Conv2d_fused_bn"
                })

        # 2. Map Classifier layers
        py_model.classifier[0].weight.data.copy_(torch.from_numpy(self.initializers["classifier.0.weight"].copy()))
        py_model.classifier[0].bias.data.copy_(torch.from_numpy(self.initializers["classifier.0.bias"].copy()))
        py_model.classifier[3].weight.data.copy_(torch.from_numpy(self.initializers["classifier.3.weight"].copy()))
        py_model.classifier[3].bias.data.copy_(torch.from_numpy(self.initializers["classifier.3.bias"].copy()))
        
        py_model.eval()
        metadata = {
            "num_classes": 10,
            "transferred_conv_layers": len(self.conv_nodes),
            "classifier_head_transferred": True,
            "transfer_log": transfer_log
        }
        return py_model, metadata

    def build_9class_trainable_model(
        self,
        transfer_backbone: bool = True,
        transfer_classifier_0: bool = True,
        reinit_classifier_head: bool = True
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """Constructs a 9-class PyTorch MobileNetV3-Small model matching the full semiconductor dataset.
        Transfers backbone and classifier.0 weights from ONNX while initializing the final 9-class
        linear classifier.
        """
        model = models.mobilenet_v3_small(num_classes=9)
        py_conv_named = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
        
        transfer_stats = {
            "transferred_tensors": 0,
            "transformed_tensors": 0,
            "newly_initialized_tensors": 0,
            "unmatched_tensors": 0,
            "tensor_records": []
        }

        if transfer_backbone:
            for i, (c_node, (py_name, py_conv)) in enumerate(zip(self.conv_nodes, py_conv_named)):
                w_name = c_node.input[1]
                b_name = c_node.input[2]
                w = torch.from_numpy(self.initializers[w_name].copy())
                b = torch.from_numpy(self.initializers[b_name].copy())
                
                py_conv.weight.data.copy_(w)
                transfer_stats["transferred_tensors"] += 1
                transfer_stats["tensor_records"].append({
                    "onnx_name": w_name,
                    "target": f"{py_name}.weight",
                    "onnx_shape": list(w.shape),
                    "py_shape": list(py_conv.weight.shape),
                    "status": "TRANSFERRED"
                })
                
                if py_conv.bias is not None:
                    py_conv.bias.data.copy_(b)
                    transfer_stats["transferred_tensors"] += 1
                    transfer_stats["tensor_records"].append({
                        "onnx_name": b_name,
                        "target": f"{py_name}.bias",
                        "onnx_shape": list(b.shape),
                        "py_shape": list(py_conv.bias.shape),
                        "status": "TRANSFERRED"
                    })
                else:
                    # BatchNorm folding: initialize BN running stats and biases
                    parts = py_name.split(".")
                    parent = model
                    for p in parts[:-1]:
                        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
                    idx = int(parts[-1])
                    bn = parent[idx + 1]
                    if isinstance(bn, nn.BatchNorm2d):
                        bn.weight.data.fill_(1.0)
                        bn.bias.data.copy_(b)
                        bn.running_mean.data.zero_()
                        bn.running_var.data.fill_(1.0)
                        bn.eps = 1e-5
                        transfer_stats["transferred_tensors"] += 1
                        transfer_stats["tensor_records"].append({
                            "onnx_name": b_name,
                            "target": f"{py_name}_bn.bias",
                            "onnx_shape": list(b.shape),
                            "py_shape": list(bn.bias.shape),
                            "status": "TRANSFERRED_TO_BN"
                        })

        if transfer_classifier_0:
            model.classifier[0].weight.data.copy_(torch.from_numpy(self.initializers["classifier.0.weight"].copy()))
            model.classifier[0].bias.data.copy_(torch.from_numpy(self.initializers["classifier.0.bias"].copy()))
            transfer_stats["transferred_tensors"] += 2
            transfer_stats["tensor_records"].append({
                "onnx_name": "classifier.0.weight",
                "target": "classifier.0.weight",
                "onnx_shape": list(self.initializers["classifier.0.weight"].shape),
                "py_shape": list(model.classifier[0].weight.shape),
                "status": "TRANSFERRED"
            })
            transfer_stats["tensor_records"].append({
                "onnx_name": "classifier.0.bias",
                "target": "classifier.0.bias",
                "onnx_shape": list(self.initializers["classifier.0.bias"].shape),
                "py_shape": list(model.classifier[0].bias.shape),
                "status": "TRANSFERRED"
            })

        if reinit_classifier_head:
            nn.init.kaiming_normal_(model.classifier[3].weight, nonlinearity="linear")
            nn.init.zeros_(model.classifier[3].bias)
            transfer_stats["newly_initialized_tensors"] += 2
            transfer_stats["tensor_records"].append({
                "onnx_name": "classifier.3.weight (10-class)",
                "target": "classifier.3.weight (9-class)",
                "onnx_shape": [10, 1024],
                "py_shape": [9, 1024],
                "status": "NEWLY_INITIALIZED"
            })
            transfer_stats["tensor_records"].append({
                "onnx_name": "classifier.3.bias (10-class)",
                "target": "classifier.3.bias (9-class)",
                "onnx_shape": [10],
                "py_shape": [9],
                "status": "NEWLY_INITIALIZED"
            })

        metadata = {
            "num_classes": 9,
            "architecture": "MobileNetV3-Small",
            "class_mapping": self.DEFAULT_9CLASS_MAPPING,
            "transfer_stats": transfer_stats
        }
        return model, metadata

    def test_10class_numerical_equivalence(
        self,
        num_test_samples: int = 5,
        tolerance_mae: float = 1e-4,
        tolerance_cosine: float = 0.9999
    ) -> Dict[str, Any]:
        """Tests numerical equivalence between the 10-class PyTorch reference model
        and the source ONNX Runtime model.
        """
        import onnxruntime as ort
        
        py_model, _ = self.build_10class_reference_model()
        ort_sess = ort.InferenceSession(self.onnx_model_path)
        
        np.random.seed(42)
        test_inputs = [
            np.random.randn(1, 3, 128, 128).astype(np.float32),
            np.random.uniform(0.0, 1.0, (1, 3, 128, 128)).astype(np.float32),
            np.random.uniform(0.0, 1.0, (4, 3, 128, 128)).astype(np.float32),
            np.random.randn(2, 3, 128, 128).astype(np.float32),
            np.random.uniform(-1.0, 1.0, (1, 3, 128, 128)).astype(np.float32)
        ][:num_test_samples]
        
        cosines, maes, rmses, max_errors = [], [], [], []
        
        for x in test_inputs:
            ort_out = ort_sess.run(None, {"input": x})[0]
            with torch.no_grad():
                py_out = py_model(torch.from_numpy(x)).numpy()
            
            cos_sim = float(np.dot(ort_out.flatten(), py_out.flatten()) / (np.linalg.norm(ort_out) * np.linalg.norm(py_out) + 1e-12))
            mae = float(np.mean(np.abs(ort_out - py_out)))
            rmse = float(np.sqrt(np.mean((ort_out - py_out) ** 2)))
            max_err = float(np.max(np.abs(ort_out - py_out)))
            
            cosines.append(cos_sim)
            maes.append(mae)
            rmses.append(rmse)
            max_errors.append(max_err)

        mean_cosine = float(np.mean(cosines))
        mean_mae = float(np.mean(maes))
        mean_rmse = float(np.mean(rmses))
        max_abs_error = float(np.max(max_errors))
        
        is_verified = (mean_cosine >= tolerance_cosine) and (mean_mae <= tolerance_mae)
        
        return {
            "status": "VERIFIED" if is_verified else "FAILED",
            "cosine_similarity": mean_cosine,
            "mae": mean_mae,
            "rmse": mean_rmse,
            "max_absolute_error": max_abs_error,
            "sample_count": len(test_inputs),
            "per_sample_cosines": cosines,
            "per_sample_maes": maes
        }
