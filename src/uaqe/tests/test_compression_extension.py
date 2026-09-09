import unittest
import numpy as np
import array
import math
from uaqe.common.imr import IMRTensor, IMRLayer
from uaqe.common.types import Precision
from uaqe.compression.sparse_encoder import SparseEncoder
from uaqe.compression.pruning import PruningApplier, PruningMask
from uaqe.compression.weight_cluster import WeightClusterCompressor, ClusteringCodebook
from uaqe.common.exceptions import CompressionError

class TestCompressionExtension(unittest.TestCase):
    def test_sparse_encoding_fp16_and_int8(self):
        encoder = SparseEncoder()
        
        # Test FP16 sparse encoding / decoding
        fp16_data = np.array([1.0, 0.0, 2.0, 0.0, 3.0], dtype=np.float16)
        tensor_fp16 = IMRTensor(shape=(5,), dtype="float16", data=fp16_data.tobytes())
        
        # Sparsity is high, so kept count is low (keep only 1.0, 2.0, 3.0 at indices 0, 2, 4)
        sparse_fp16 = encoder.encode_tensor(tensor_fp16, [0, 2, 4])
        self.assertEqual(sparse_fp16.dtype, "sparse_coo_float16")
        # Header (4) + Indices (3 * 4) + Values (3 * 2) = 4 + 12 + 6 = 22 bytes
        self.assertEqual(len(sparse_fp16.data), 22)
        
        decoded_fp16 = encoder.decode_tensor(sparse_fp16)
        self.assertEqual(decoded_fp16.dtype, "float16")
        decoded_arr = np.frombuffer(decoded_fp16.data, dtype=np.float16)
        np.testing.assert_array_equal(decoded_arr, fp16_data)
        
        # Test INT8 sparse encoding / decoding
        int8_data = np.array([5, 0, -10, 0, 15], dtype=np.int8)
        tensor_int8 = IMRTensor(shape=(5,), dtype="int8", data=int8_data.tobytes())
        
        sparse_int8 = encoder.encode_tensor(tensor_int8, [0, 2, 4])
        self.assertEqual(sparse_int8.dtype, "sparse_coo_int8")
        # Header (4) + Indices (3 * 4) + Values (3 * 1) = 4 + 12 + 3 = 19 bytes
        self.assertEqual(len(sparse_int8.data), 19)
        
        decoded_int8 = encoder.decode_tensor(sparse_int8)
        self.assertEqual(decoded_int8.dtype, "int8")
        decoded_arr_int8 = np.frombuffer(decoded_int8.data, dtype=np.int8)
        np.testing.assert_array_equal(decoded_arr_int8, int8_data)

    def test_dense_fallback_pruning(self):
        applier = PruningApplier()
        
        # Scenario A: Sparsity = 0.2 (keep 8 elements) -> dense fallback
        fp16_data = np.array([1.0] * 10, dtype=np.float16)
        tensor = IMRTensor(shape=(10,), dtype="float16", data=fp16_data.tobytes())
        layer = IMRLayer(name="layer", op_type="Conv", inputs=[], outputs=[], parameters={"weight": tensor}, attributes={}, precision=Precision.FP16)
        
        # Compute magnitude mask for sparsity 0.2
        mask = applier.compute_magnitude_mask(tensor, "weight", 0.2)
        self.assertEqual(len(mask.kept_indices), 8)
        
        new_layer, stats = applier.apply_mask(layer, {"weight": mask})
        # Check that it fell back to dense float16
        self.assertEqual(new_layer.parameters["weight"].dtype, "float16")
        self.assertEqual(len(new_layer.parameters["weight"].data), 20)
        
        # Scenario B: Sparsity = 0.8 (keep 2 elements) -> sparse encoding
        mask_sparse = applier.compute_magnitude_mask(tensor, "weight", 0.8)
        self.assertEqual(len(mask_sparse.kept_indices), 2)
        
        new_layer_sparse, stats_sparse = applier.apply_mask(layer, {"weight": mask_sparse})
        # Check that it used sparse encoding
        self.assertEqual(new_layer_sparse.parameters["weight"].dtype, "sparse_coo_float16")
        self.assertEqual(len(new_layer_sparse.parameters["weight"].data), 16)

    def test_weight_clustering_fp16_and_int8(self):
        class FakeLogger:
            def info(self, *args, **kwargs): pass
            def warning(self, *args, **kwargs): pass
            
        clusterer = WeightClusterCompressor(logger=FakeLogger(), num_clusters=4)
        
        # Test FP16 clustering
        # Dense size = 100 * 2 = 200 bytes.
        # Clustered size = 100 (uint8) + 4 (centroids) * 2 = 108 bytes < 200.
        # So clustering must execute and reduce size!
        fp16_vals = np.array([float(i % 4) for i in range(100)], dtype=np.float16)
        tensor_fp16 = IMRTensor(shape=(100,), dtype="float16", data=fp16_vals.tobytes())
        layer_fp16 = IMRLayer(name="layer", op_type="Conv", inputs=[], outputs=[], parameters={"weight": tensor_fp16}, attributes={}, precision=Precision.FP16)
        
        new_layer_fp16 = clusterer._cluster_layer(layer_fp16)
        self.assertEqual(new_layer_fp16.parameters["weight"].dtype, "clustered_uint8")
        self.assertEqual(len(new_layer_fp16.parameters["weight"].data), 100)
        self.assertIn("weight_codebook", new_layer_fp16.attributes)
        
        # Test INT8 clustering fallback
        # Dense size = 100 * 1 = 100 bytes.
        # Clustered size = 100 + 4 * 1 = 104 bytes > 100.
        # Clustering must fallback (do not cluster) because clustered_size >= dense_size.
        int8_vals = np.array([i % 4 for i in range(100)], dtype=np.int8)
        tensor_int8 = IMRTensor(shape=(100,), dtype="int8", data=int8_vals.tobytes())
        layer_int8 = IMRLayer(name="layer", op_type="Conv", inputs=[], outputs=[], parameters={"weight": tensor_int8}, attributes={}, precision=Precision.INT8)
        
        new_layer_int8 = clusterer._cluster_layer(layer_int8)
        # Verify it fell back to unclustered int8
        self.assertEqual(new_layer_int8.parameters["weight"].dtype, "int8")
        self.assertEqual(len(new_layer_int8.parameters["weight"].data), 100)
        self.assertNotIn("weight_codebook", new_layer_int8.attributes)

    def test_clustered_tensor_reconstruction(self):
        class FakeLogger:
            def info(self, *args, **kwargs): pass
            def warning(self, *args, **kwargs): pass
            
        clusterer = WeightClusterCompressor(logger=FakeLogger(), num_clusters=8)
        
        # Original float16 array containing some pattern
        orig_vals = np.array([float(i % 8) for i in range(100)], dtype=np.float16)
        tensor = IMRTensor(shape=(10, 10), dtype="float16", data=orig_vals.tobytes())
        layer = IMRLayer(name="layer", op_type="Conv", inputs=[], outputs=[], parameters={"weight": tensor}, attributes={}, precision=Precision.FP16)
        
        new_layer = clusterer._cluster_layer(layer)
        c_tensor = new_layer.parameters["weight"]
        codebook = new_layer.attributes["weight_codebook"]
        
        # Reconstruct the tensor
        indices = np.frombuffer(c_tensor.data, dtype=np.uint8)
        centroids = np.array(codebook.centroids, dtype=np.float32)
        recon_vals = centroids[indices]
        
        # Verify shape preservation and element count
        self.assertEqual(c_tensor.shape, (10, 10))
        self.assertEqual(len(recon_vals), 100)
        
        # Verify codebook centroids size
        self.assertEqual(len(centroids), 8)
        
        # Verify index range is valid
        self.assertTrue(np.all(indices < 8))
        self.assertTrue(np.all(indices >= 0))
        
        # Calculate errors
        abs_errors = np.abs(orig_vals.astype(np.float32) - recon_vals)
        max_err = np.max(abs_errors)
        mean_err = np.mean(abs_errors)
        
        # Since we clustered 8 unique values into 8 centroids, reconstruction should be exact
        self.assertAlmostEqual(max_err, 0.0, places=5)
        self.assertAlmostEqual(mean_err, 0.0, places=5)

    def test_runtime_selector_and_exporter_factory(self):
        from uaqe.domain.hardware_manager import HardwareProfile
        from uaqe.common.types import HardwareClass, ExportFormat
        from uaqe.runtime.runtime_selector import RuntimeSelector
        from uaqe.exporter.exporter_factory import ExporterFactory
        from uaqe.common.exceptions import ExportError
        
        # 1. Create a dummy profile
        profile = HardwareProfile(
            profile_id="test_pi5",
            display_name="Test Pi 5",
            hardware_class=HardwareClass.RASPBERRY_PI,
            ram_bytes=1024,
            flash_bytes=None,
            storage_bytes=1024,
            tensor_memory_bytes=512,
            runtime="tflite-runtime",
            default_runtime="tflite-runtime",
            supported_runtimes=["tflite-runtime", "onnxruntime", "torch"],
            max_model_size_bytes=1024,
            schema_version="1.0"
        )
        
        selector = RuntimeSelector()
        
        # Test default runtime selection
        runtime = selector.select_runtime(profile, requested_runtime=None)
        self.assertEqual(runtime, "tflite-runtime")
        
        # Test override runtime selection
        runtime_override = selector.select_runtime(profile, requested_runtime="onnxruntime")
        self.assertEqual(runtime_override, "onnxruntime")
        
        # Test unsupported runtime validation
        with self.assertRaises(ExportError) as ctx:
            selector.select_runtime(profile, requested_runtime="bare-metal-hdl")
        self.assertEqual(ctx.exception.code, "UNSUPPORTED_RUNTIME")
        
        # 2. Test ExporterFactory mapping
        class FakeBackend:
            def __init__(self, fmt):
                self.EXPORT_FORMAT = fmt
                
        backends = [FakeBackend(ExportFormat.ONNX), FakeBackend(ExportFormat.TFLITE)]
        factory = ExporterFactory(backends)
        
        # Test mapping
        self.assertEqual(factory.get_exporter("onnxruntime").EXPORT_FORMAT, ExportFormat.ONNX)
        self.assertEqual(factory.get_exporter("tflite-runtime").EXPORT_FORMAT, ExportFormat.TFLITE)
        
        # Test torch not implemented
        with self.assertRaises(ExportError) as ctx:
            factory.get_exporter("torch")
        self.assertEqual(ctx.exception.code, "RUNTIME_NOT_IMPLEMENTED")

    def test_tflite_conversion_blocker(self):
        from uaqe.exporter.tflite_exporter import TFLiteExporter
        from uaqe.domain.hardware_manager import HardwareProfile
        from uaqe.common.types import HardwareClass
        from uaqe.common.imr import IMR, IMRMetadata
        from uaqe.common.exceptions import ExportError
        
        profile = HardwareProfile(
            profile_id="test_pi5",
            display_name="Test Pi 5",
            hardware_class=HardwareClass.RASPBERRY_PI,
            ram_bytes=1024,
            flash_bytes=None,
            storage_bytes=1024,
            tensor_memory_bytes=512,
            runtime="tflite-runtime",
            default_runtime="tflite-runtime",
            supported_runtimes=["tflite-runtime"],
            max_model_size_bytes=1024,
            schema_version="1.0"
        )
        
        class FakeLogger:
            def info(self, *args, **kwargs): pass
            def warning(self, *args, **kwargs): pass
            
        exporter = TFLiteExporter(logger=FakeLogger())
        imr = IMR(layers=[], metadata=IMMRMetadata(source_framework="onnx", source_format=".onnx", original_input_shapes={}, op_count=0, total_parameters=0) if 'IMMRMetadata' in globals() else IMRMetadata(source_framework="onnx", source_format=".onnx", original_input_shapes={}, op_count=0, total_parameters=0))
        
        with self.assertRaises(ExportError) as ctx:
            exporter.export(imr, profile)
        self.assertIn("requires a genuine TFLite FlatBuffer", str(ctx.exception))

    def test_fp16_onnx_export_details(self):
        import onnx
        from onnx import numpy_helper
        from uaqe.common.imr import IMR, IMRLayer, IMRTensor, IMRMetadata
        from uaqe.exporter.onnx_exporter import OnnxExporter, consumer_supports_fp16
        from uaqe.domain.hardware_manager import HardwareProfile
        from uaqe.common.types import HardwareClass, ExportFormat
        import numpy as np
        import os
        
        # 1. Create a dummy original ONNX model to load
        model = onnx.ModelProto()
        model.ir_version = 7
        opset = model.opset_import.add()
        opset.domain = ""
        opset.version = 13
        
        # Add input and output to graph
        graph = model.graph
        graph.name = "test_graph"
        inp = graph.input.add()
        inp.name = "input"
        inp.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        inp.type.tensor_type.shape.dim.add().dim_value = 1
        inp.type.tensor_type.shape.dim.add().dim_value = 10
        
        out = graph.output.add()
        out.name = "output"
        out.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        out.type.tensor_type.shape.dim.add().dim_value = 1
        out.type.tensor_type.shape.dim.add().dim_value = 10
        
        # Add weights initializers
        w1_data = np.random.randn(10, 10).astype(np.float32)
        w1_init = numpy_helper.from_array(w1_data, name="w1")
        graph.initializer.append(w1_init)
        
        w2_data = np.random.randn(10, 10).astype(np.float32)
        w2_init = numpy_helper.from_array(w2_data, name="w2")
        graph.initializer.append(w2_init)
        
        # Add nodes
        node1 = onnx.helper.make_node(
            "Gemm",
            inputs=["input", "w1", "w2"],
            outputs=["output"],
            name="gemm1"
        )
        graph.node.append(node1)
        
        # Write dummy model to disk
        dummy_onnx_path = "scratch/dummy_test_model.onnx"
        os.makedirs("scratch", exist_ok=True)
        onnx.save(model, dummy_onnx_path)
        
        try:
            # 2. Build mock IMR
            # w1 is quantized to float16
            w1_fp16 = w1_data.astype(np.float16)
            tensor_w1 = IMRTensor(shape=(10, 10), dtype="float16", data=w1_fp16.tobytes())
            
            # w2 is clustered_uint8 starting from float16
            # codebook has 8 centroids
            from uaqe.compression.weight_cluster import ClusteringCodebook
            w2_fp16 = w2_data.astype(np.float16)
            centroids = tuple(float(x) for x in np.linspace(-1, 1, 8))
            indices = np.random.randint(0, 8, size=(100,)).astype(np.uint8)
            tensor_w2 = IMRTensor(shape=(10, 10), dtype="clustered_uint8", data=indices.tobytes())
            
            layer = IMRLayer(
                name="gemm_layer",
                op_type="Gemm",
                inputs=["input", "w1", "w2"],
                outputs=["output"],
                parameters={"w1": tensor_w1, "w2": tensor_w2},
                attributes={
                    "w2_codebook": ClusteringCodebook(centroids=centroids),
                    "w2_source_precision": "float16"
                },
                precision=Precision.FP16
            )
            
            imr = IMR(
                layers=[layer],
                metadata=IMRMetadata(
                    source_framework="onnx",
                    source_format=".onnx",
                    original_input_shapes={"input": (1, 10)},
                    op_count=1,
                    total_parameters=200
                )
            )
            
            profile = HardwareProfile(
                profile_id="test_pi5",
                display_name="Test Pi 5",
                hardware_class=HardwareClass.RASPBERRY_PI,
                ram_bytes=1024,
                flash_bytes=None,
                storage_bytes=1024,
                tensor_memory_bytes=512,
                runtime="onnxruntime",
                default_runtime="onnxruntime",
                supported_runtimes=["onnxruntime"],
                max_model_size_bytes=1024,
                schema_version="1.0"
            )
            
            class FakeLogger:
                def info(self, *args, **kwargs): pass
                def warning(self, *args, **kwargs): pass
                
            # 3. Export using OnnxExporter
            exporter = OnnxExporter(logger=FakeLogger(), output_dir="scratch/outputs")
            exporter.bind_source_model_path(dummy_onnx_path)
            artifact = exporter.export(imr, profile)
            
            # 4. Load the generated model and verify assertions
            exported_model = onnx.load(artifact.file_paths[0])
            
            # Assertions:
            onnx.checker.check_model(exported_model)
            
            # Verify initializers in exported model
            inits = {i.name: i for i in exported_model.graph.initializer}
            self.assertIn("w1", inits)
            self.assertIn("w2", inits)
            
            # test_fp16_initializer_dtype
            self.assertEqual(inits["w1"].data_type, onnx.TensorProto.FLOAT16)
            self.assertEqual(inits["w2"].data_type, onnx.TensorProto.FLOAT16)
            
            # test_fp16_initializer_size: bytes == 2 * element_count
            w1_bytes = len(inits["w1"].raw_data) if inits["w1"].raw_data else numpy_helper.to_array(inits["w1"]).nbytes
            w2_bytes = len(inits["w2"].raw_data) if inits["w2"].raw_data else numpy_helper.to_array(inits["w2"]).nbytes
            self.assertEqual(w1_bytes, 200)
            self.assertEqual(w2_bytes, 200)
            
            # test_fp16_shape_preservation
            self.assertEqual(list(inits["w1"].dims), [10, 10])
            self.assertEqual(list(inits["w2"].dims), [10, 10])
            
            # test_fp32_consumer_cast_insertion
            # Consumers of w1 and w2 (Gemm) do not support fp16 directly in our compatibility logic,
            # so there must be Cast nodes inserted.
            nodes = {n.name: n for n in exported_model.graph.node}
            self.assertIn("__fp16_cast__w1", nodes)
            self.assertIn("__fp16_cast__w2", nodes)
            
            # Verify node outputs are used as inputs to Gemm
            self.assertEqual(nodes["__fp16_cast__w1"].output[0], "w1__fp32")
            self.assertEqual(nodes["__fp16_cast__w2"].output[0], "w2__fp32")
            
            gemm_node = nodes["gemm1"]
            self.assertEqual(gemm_node.input[1], "w1__fp32")
            self.assertEqual(gemm_node.input[2], "w2__fp32")
            
            # test_onnxruntime_fp16_load
            import onnxruntime as ort
            sess = ort.InferenceSession(artifact.file_paths[0])
            self.assertIsNotNone(sess)
            
            # Clean up generated files
            if os.path.exists(artifact.file_paths[0]):
                os.remove(artifact.file_paths[0])
                
        finally:
            if os.path.exists(dummy_onnx_path):
                os.remove(dummy_onnx_path)

    def test_int8_onnx_export_details(self):
        import onnx
        from onnx import numpy_helper
        from uaqe.common.imr import IMR, IMRLayer, IMRTensor, IMRMetadata
        from uaqe.exporter.onnx_exporter import OnnxExporter
        from uaqe.domain.hardware_manager import HardwareProfile
        from uaqe.common.types import HardwareClass, ExportFormat
        from uaqe.common.exceptions import ExportError
        from uaqe.common.types import Precision
        import numpy as np
        import os
        import dataclasses
        
        # 1. Create a dummy original ONNX model to load
        model = onnx.ModelProto()
        model.ir_version = 7
        opset = model.opset_import.add()
        opset.domain = ""
        opset.version = 13
        
        graph = model.graph
        graph.name = "test_int8_graph"
        
        inp = graph.input.add()
        inp.name = "input"
        inp.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        inp.type.tensor_type.shape.dim.add().dim_value = 1
        inp.type.tensor_type.shape.dim.add().dim_value = 10
        
        out = graph.output.add()
        out.name = "output"
        out.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        out.type.tensor_type.shape.dim.add().dim_value = 1
        out.type.tensor_type.shape.dim.add().dim_value = 10
        
        # Add weights initializers
        w1_data = np.random.randn(10, 10).astype(np.float32)
        w1_init = numpy_helper.from_array(w1_data, name="w1")
        graph.initializer.append(w1_init)
        
        w2_data = np.random.randn(10, 10).astype(np.float32)
        w2_init = numpy_helper.from_array(w2_data, name="w2")
        graph.initializer.append(w2_init)
        
        # Add node
        node1 = onnx.helper.make_node(
            "Gemm",
            inputs=["input", "w1", "w2"],
            outputs=["output"],
            name="gemm1"
        )
        graph.node.append(node1)
        
        dummy_onnx_path = "scratch/dummy_test_int8_model.onnx"
        os.makedirs("scratch", exist_ok=True)
        onnx.save(model, dummy_onnx_path)
        
        try:
            # 2. Build mock INT8 IMR
            scale_w1 = 0.05
            w1_int8 = np.clip(np.round(w1_data / scale_w1), -128, 127).astype(np.int8)
            tensor_w1 = IMRTensor(shape=(10, 10), dtype="int8", data=w1_int8.tobytes())
            
            # w2 is clustered_uint8 starting from int8
            from uaqe.compression.weight_cluster import ClusteringCodebook
            scale_w2 = 0.1
            w2_int8 = np.clip(np.round(w2_data / scale_w2), -128, 127).astype(np.int8)
            centroids = tuple(float(x) for x in range(-4, 4))  # INT8 centroids in integer range
            indices = np.random.randint(0, 8, size=(100,)).astype(np.uint8)
            tensor_w2 = IMRTensor(shape=(10, 10), dtype="clustered_uint8", data=indices.tobytes())
            
            layer = IMRLayer(
                name="gemm_layer",
                op_type="Gemm",
                inputs=["input", "w1", "w2"],
                outputs=["output"],
                parameters={"w1": tensor_w1, "w2": tensor_w2},
                attributes={
                    "w1_scale": scale_w1,
                    "w1_zero_point": 0,
                    "w2_scale": scale_w2,
                    "w2_zero_point": 0,
                    "w2_codebook": ClusteringCodebook(centroids=centroids),
                    "w2_source_precision": "int8"
                },
                precision=Precision.INT8
            )
            
            imr = IMR(
                layers=[layer],
                metadata=IMRMetadata(
                    source_framework="onnx",
                    source_format=".onnx",
                    original_input_shapes={"input": (1, 10)},
                    op_count=1,
                    total_parameters=200
                )
            )
            
            profile = HardwareProfile(
                profile_id="test_pi5",
                display_name="Test Pi 5",
                hardware_class=HardwareClass.RASPBERRY_PI,
                ram_bytes=1024,
                flash_bytes=None,
                storage_bytes=1024,
                tensor_memory_bytes=512,
                runtime="onnxruntime",
                default_runtime="onnxruntime",
                supported_runtimes=["onnxruntime"],
                max_model_size_bytes=1024,
                schema_version="1.0"
            )
            
            class FakeLogger:
                def info(self, *args, **kwargs): pass
                def warning(self, *args, **kwargs): pass
                
            exporter = OnnxExporter(logger=FakeLogger(), output_dir="scratch/outputs")
            exporter.bind_source_model_path(dummy_onnx_path)
            artifact = exporter.export(imr, profile)
            
            # 3. Verify exported graph properties
            exported_model = onnx.load(artifact.file_paths[0])
            onnx.checker.check_model(exported_model)
            
            inits = {i.name: i for i in exported_model.graph.initializer}
            
            # test_int8_initializer_dtype
            self.assertEqual(inits["w1"].data_type, onnx.TensorProto.INT8)
            self.assertEqual(inits["w2"].data_type, onnx.TensorProto.INT8)
            
            # test_int8_initializer_size: bytes == 1 * element_count
            w1_bytes = len(inits["w1"].raw_data) if inits["w1"].raw_data else numpy_helper.to_array(inits["w1"]).nbytes
            w2_bytes = len(inits["w2"].raw_data) if inits["w2"].raw_data else numpy_helper.to_array(inits["w2"]).nbytes
            self.assertEqual(w1_bytes, 100)
            self.assertEqual(w2_bytes, 100)
            
            # test_int8_shape_preservation
            self.assertEqual(list(inits["w1"].dims), [10, 10])
            self.assertEqual(list(inits["w2"].dims), [10, 10])
            
            # test_scale_serialization & test_zero_point_serialization
            self.assertIn("w1__scale", inits)
            self.assertIn("w1__zero_point", inits)
            self.assertEqual(inits["w1__scale"].data_type, onnx.TensorProto.FLOAT)
            self.assertEqual(inits["w1__zero_point"].data_type, onnx.TensorProto.INT8)
            
            # test_dequantization_reproduces_quantizer_values
            # Reconstruct and compare dequantized weights
            w1_q = numpy_helper.to_array(inits["w1"])
            scale_val = numpy_helper.to_array(inits["w1__scale"])
            zp_val = numpy_helper.to_array(inits["w1__zero_point"])
            dequantized = scale_val * (w1_q.astype(np.float32) - zp_val)
            expected_dequantized = scale_w1 * w1_int8.astype(np.float32)
            np.testing.assert_allclose(dequantized, expected_dequantized, atol=1e-5)
            
            # test_clustered_int8_reconstruction & test_no_double_quantization
            w2_q = numpy_helper.to_array(inits["w2"])
            self.assertEqual(w2_q.dtype, np.int8)
            recon_centroids = np.array(centroids, dtype=np.float32)
            expected_w2_int8 = np.clip(np.round(recon_centroids[indices]), -128, 127).astype(np.int8).reshape(10, 10)
            np.testing.assert_array_equal(w2_q, expected_w2_int8)
            
            # test_dq_graph_generation
            nodes = {n.name: n for n in exported_model.graph.node}
            self.assertIn("__int8_dequant__w1", nodes)
            self.assertIn("__int8_dequant__w2", nodes)
            
            # test_onnxruntime_load
            import onnxruntime as ort
            sess = ort.InferenceSession(artifact.file_paths[0])
            self.assertIsNotNone(sess)
            
            if os.path.exists(artifact.file_paths[0]):
                os.remove(artifact.file_paths[0])
                
            # 4. Negative tests
            # test_per_channel_rejection
            layer_invalid_scale = dataclasses.replace(
                layer,
                attributes={
                    "w1_scale": np.array([0.01, 0.02], dtype=np.float32),  # Vector scale
                    "w1_zero_point": 0,
                    "w2_scale": scale_w2,
                    "w2_zero_point": 0,
                    "w2_codebook": ClusteringCodebook(centroids=centroids),
                    "w2_source_precision": "int8"
                }
            )
            imr_invalid_scale = dataclasses.replace(imr, layers=[layer_invalid_scale])
            with self.assertRaises(ExportError) as ctx:
                exporter.export(imr_invalid_scale, profile)
            self.assertEqual(ctx.exception.code, "UNSUPPORTED_QUANTIZATION")
            
            # test_missing_scale_rejection
            layer_missing_scale = dataclasses.replace(
                layer,
                attributes={
                    "w1_zero_point": 0,
                    "w2_scale": scale_w2,
                    "w2_zero_point": 0,
                    "w2_codebook": ClusteringCodebook(centroids=centroids),
                    "w2_source_precision": "int8"
                }
            )
            imr_missing_scale = dataclasses.replace(imr, layers=[layer_missing_scale])
            with self.assertRaises(ExportError) as ctx:
                exporter.export(imr_missing_scale, profile)
            self.assertEqual(ctx.exception.code, "EXPORT_METADATA_MISSING")

        finally:
            if os.path.exists(dummy_onnx_path):
                os.remove(dummy_onnx_path)

    def test_tflite_export_details(self):
        import onnx
        import tensorflow as tf
        from uaqe.common.imr import IMR, IMRLayer, IMRTensor, IMRMetadata
        from uaqe.exporter.tflite_exporter import TFLiteExporter
        from uaqe.domain.hardware_manager import HardwareProfile
        from uaqe.common.types import HardwareClass, ExportFormat
        from uaqe.common.exceptions import ExportError
        from uaqe.common.types import Precision
        import numpy as np
        import os
        import dataclasses
        from onnx import numpy_helper
        
        # 1. Create a dummy original ONNX model to load
        model = onnx.ModelProto()
        model.ir_version = 7
        opset = model.opset_import.add()
        opset.domain = ""
        opset.version = 13
        
        graph = model.graph
        graph.name = "test_tflite_graph"
        
        inp = graph.input.add()
        inp.name = "input"
        inp.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        inp.type.tensor_type.shape.dim.add().dim_value = 1
        inp.type.tensor_type.shape.dim.add().dim_value = 10
        
        out = graph.output.add()
        out.name = "output"
        out.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        out.type.tensor_type.shape.dim.add().dim_value = 1
        out.type.tensor_type.shape.dim.add().dim_value = 10
        
        # Add weights initializers
        w1_data = np.random.randn(10, 10).astype(np.float32)
        w1_init = numpy_helper.from_array(w1_data, name="w1")
        graph.initializer.append(w1_init)
        
        w2_data = np.random.randn(10, 10).astype(np.float32)
        w2_init = numpy_helper.from_array(w2_data, name="w2")
        graph.initializer.append(w2_init)
        
        # Add node
        node1 = onnx.helper.make_node(
            "Gemm",
            inputs=["input", "w1", "w2"],
            outputs=["output"],
            name="gemm1"
        )
        graph.node.append(node1)
        
        dummy_onnx_path = "scratch/dummy_test_tflite_model.onnx"
        os.makedirs("scratch", exist_ok=True)
        onnx.save(model, dummy_onnx_path)
        
        try:
            # 2. Build mock INT8 IMR
            scale_w1 = 0.05
            w1_int8 = np.clip(np.round(w1_data / scale_w1), -128, 127).astype(np.int8)
            tensor_w1 = IMRTensor(shape=(10, 10), dtype="int8", data=w1_int8.tobytes())
            
            # w2 is clustered_uint8 starting from int8
            from uaqe.compression.weight_cluster import ClusteringCodebook
            scale_w2 = 0.1
            w2_int8 = np.clip(np.round(w2_data / scale_w2), -128, 127).astype(np.int8)
            centroids = tuple(float(x) for x in range(-4, 4))
            indices = np.random.randint(0, 8, size=(100,)).astype(np.uint8)
            tensor_w2 = IMRTensor(shape=(10, 10), dtype="clustered_uint8", data=indices.tobytes())
            
            layer = IMRLayer(
                name="gemm_layer",
                op_type="Gemm",
                inputs=["input", "w1", "w2"],
                outputs=["output"],
                parameters={"w1": tensor_w1, "w2": tensor_w2},
                attributes={
                    "w1_scale": scale_w1,
                    "w1_zero_point": 0,
                    "w2_scale": scale_w2,
                    "w2_zero_point": 0,
                    "w2_codebook": ClusteringCodebook(centroids=centroids),
                    "w2_source_precision": "int8"
                },
                precision=Precision.INT8
            )
            
            imr = IMR(
                layers=[layer],
                metadata=IMRMetadata(
                    source_framework="onnx",
                    source_format=".onnx",
                    original_input_shapes={"input": (1, 10)},
                    op_count=1,
                    total_parameters=200
                )
            )
            
            profile = HardwareProfile(
                profile_id="test_pi5",
                display_name="Test Pi 5",
                hardware_class=HardwareClass.RASPBERRY_PI,
                ram_bytes=1024,
                flash_bytes=None,
                storage_bytes=1024,
                tensor_memory_bytes=512,
                runtime="tflite-runtime",
                default_runtime="tflite-runtime",
                supported_runtimes=["tflite-runtime"],
                max_model_size_bytes=1024,
                schema_version="1.0"
            )
            
            class FakeLogger:
                def info(self, *args, **kwargs): pass
                def warning(self, *args, **kwargs): pass
                
            exporter = TFLiteExporter(logger=FakeLogger(), output_dir="scratch/outputs")
            exporter.bind_source_model_path(dummy_onnx_path)
            artifact = exporter.export(imr, profile)
            
            # 3. Verify exported graph properties
            tflite_path = artifact.file_paths[0]
            self.assertTrue(tflite_path.endswith(".tflite"))
            self.assertTrue(os.path.exists(tflite_path))
            self.assertGreater(os.path.getsize(tflite_path), 0)
            
            # Verify loading with interpreter
            interpreter = tf.lite.Interpreter(tflite_path)
            interpreter.allocate_tensors()
            
            # Verify input/output signatures
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            self.assertEqual(list(input_details[0]["shape"]), [1, 10])
            self.assertEqual(list(output_details[0]["shape"]), [10, 10])
            
            # Clean up generated files
            if os.path.exists(tflite_path):
                os.remove(tflite_path)
                
            # 4. Negative tests
            # Invalid source model path
            exporter_bad = TFLiteExporter(logger=FakeLogger(), output_dir="scratch/outputs")
            exporter_bad.bind_source_model_path("scratch/invalid_path_to_model.onnx")
            with self.assertRaises(ExportError) as ctx:
                exporter_bad.export(imr, profile)
            self.assertEqual(ctx.exception.code, "EXPORT_SOURCE_MODEL_MISSING")

        finally:
            if os.path.exists(dummy_onnx_path):
                os.remove(dummy_onnx_path)

if __name__ == "__main__":
    unittest.main()
