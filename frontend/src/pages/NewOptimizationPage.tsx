import React, { useState, useEffect, useRef } from 'react';
import {
  UploadCloud,
  FolderUp,
  FileCode,
  Database,
  Cpu,
  Sliders,
  ChevronDown,
  ChevronUp,
  Sparkles,
  Zap,
  Check,
  AlertCircle,
  Play,
  Search,
  ShieldCheck,
  Archive,
  RefreshCw,
  HardDrive,
  Layers,
  FileCheck2,
  Lock,
} from 'lucide-react';
import { SystemStatus, UploadState, ModelUploadResult, DatasetUploadResult, InputSourceType } from '../types/api';
import SourceBadge from '../components/ui/SourceBadge';
import GlassCard from '../components/ui/GlassCard';
import { apiService } from '../services/api';

interface NewOptimizationPageProps {
  systemStatus: SystemStatus | null;
  onAnalyze: (config: {
    modelUploadId?: string;
    modelId?: string;
    modelPath?: string;
    datasetUploadId?: string;
    datasetId?: string;
    datasetPath?: string;
    targetHardware: string;
    profile: string;
    calibSamples: number;
    testSamples: number;
  }) => Promise<void>;
  onDirectOptimize: (config: {
    modelUploadId?: string;
    modelId?: string;
    modelPath?: string;
    datasetUploadId?: string;
    datasetId?: string;
    datasetPath?: string;
    targetHardware: string;
    profile: string;
    calibSamples: number;
    testSamples: number;
    maxBudget: number;
  }) => Promise<void>;
  isProcessing: boolean;
}

export const NewOptimizationPage: React.FC<NewOptimizationPageProps> = ({
  systemStatus,
  onAnalyze,
  onDirectOptimize,
  isProcessing,
}) => {
  // Model input state
  const [modelState, setModelState] = useState<UploadState>('IDLE');
  const [modelUploadResult, setModelUploadResult] = useState<ModelUploadResult | null>(null);
  const [modelSource, setModelSource] = useState<InputSourceType>('USER_UPLOAD');
  const [modelProgress, setModelProgress] = useState<{ loaded: number; total: number; pct: number }>({ loaded: 0, total: 0, pct: 0 });
  const [modelError, setModelError] = useState<string | null>(null);
  const [isModelDragOver, setIsModelDragOver] = useState<boolean>(false);

  // Dataset input state
  const [datasetState, setDatasetState] = useState<UploadState>('IDLE');
  const [datasetUploadResult, setDatasetUploadResult] = useState<DatasetUploadResult | null>(null);
  const [datasetSource, setDatasetSource] = useState<InputSourceType>('USER_UPLOAD');
  const [datasetProgress, setDatasetProgress] = useState<{ loadedFiles: number; totalFiles: number; loadedBytes: number; totalBytes: number; pct: number }>({
    loadedFiles: 0,
    totalFiles: 0,
    loadedBytes: 0,
    totalBytes: 0,
    pct: 0,
  });
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [isDatasetDragOver, setIsDatasetDragOver] = useState<boolean>(false);

  // Target and Profile state
  const [targetHardware, setTargetHardware] = useState<string>('raspberrypi5');
  const [profile, setProfile] = useState<string>('balanced');

  // Advanced settings state
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [calibSamples, setCalibSamples] = useState<number>(100);
  const [testSamples, setTestSamples] = useState<number>(1000);
  const [maxBudget, setMaxBudget] = useState<number>(5);

  // DOM Refs for hidden inputs
  const modelFileInputRef = useRef<HTMLInputElement>(null);
  const datasetFolderInputRef = useRef<HTMLInputElement>(null);
  const datasetZipInputRef = useRef<HTMLInputElement>(null);

  // Available pre-verified samples from backend
  const [samples, setSamples] = useState<{ models: any[]; datasets: any[] }>({ models: [], datasets: [] });

  useEffect(() => {
    apiService.getSamples().then(setSamples).catch(console.error);
  }, []);

  // Handle Model File Selection & Upload
  const handleModelFileSelected = async (file: File) => {
    try {
      setModelError(null);
      setModelState('UPLOADING');
      setModelProgress({ loaded: 0, total: file.size, pct: 0 });
      setModelSource('USER_UPLOAD');

      const result = await apiService.uploadModelFile(file, (p) => {
        setModelProgress(p);
        if (p.pct >= 100) setModelState('VERIFYING');
      });

      setModelUploadResult(result);
      setModelState('READY');
    } catch (err: any) {
      console.error('Model upload error:', err);
      setModelError(err.message || 'Model upload failed');
      setModelState('FAILED');
    }
  };

  // Handle Dataset Folder Selection & Upload
  const handleDatasetFolderSelected = async (fileList: FileList) => {
    if (!fileList || fileList.length === 0) return;

    try {
      setDatasetError(null);
      setDatasetState('UPLOADING');
      setDatasetSource('USER_UPLOAD');

      // Use the ACTUAL File objects from: fileList (Requirement 14)
      const files = Array.from(fileList);

      // Build relativePaths normalizing \ -> / (Requirement 14)
      const relativePaths = files.map((f) =>
        (f.webkitRelativePath || f.name).replaceAll('\\', '/')
      );

      let totalBytes = 0;
      for (const f of files) totalBytes += f.size;

      // Determine root folder safely from relative paths (Requirement 14)
      let folderName = 'dataset';
      const firstRel = relativePaths[0] || '';
      if (firstRel.includes('/')) {
        folderName = firstRel.split('/')[0];
      }

      // Reset the file input AFTER capturing the files so selecting same folder triggers onChange (Requirement 14)
      if (datasetFolderInputRef.current) {
        datasetFolderInputRef.current.value = '';
      }

      // Before upload log (Requirement 11 & 14)
      console.log('[UAQE] Selected dataset folder diagnostics:', {
        fileCount: files.length,
        totalBytes,
        totalMB: (totalBytes / (1024 * 1024)).toFixed(2),
        folderName,
        firstPaths: relativePaths.slice(0, 10),
      });

      setDatasetProgress({
        loadedFiles: 0,
        totalFiles: files.length,
        loadedBytes: 0,
        totalBytes,
        pct: 0,
      });

      const result = await apiService.uploadDatasetFolder(files, relativePaths, folderName, (p) => {
        setDatasetProgress(p);
        if (p.pct >= 100) setDatasetState('RECONSTRUCTING');
      });

      // After successful upload: set dataset state to READY with confirmed result (Requirement 15)
      setDatasetUploadResult(result);
      setDatasetState('READY');
      console.log('[UAQE] Dataset upload and staging confirmed READY:', result);
    } catch (err: any) {
      console.error('[UAQE] Dataset folder upload error:', err);
      setDatasetError(err.message || 'Dataset folder upload failed');
      setDatasetState('FAILED');
    }
  };

  // Handle Dataset Zip Fallback Upload
  const handleDatasetZipSelected = async (file: File) => {
    try {
      setDatasetError(null);
      setDatasetState('UPLOADING');
      setDatasetSource('USER_UPLOAD');

      const result = await apiService.uploadDatasetZip(file, (p) => {
        setDatasetProgress({
          loadedFiles: 1,
          totalFiles: 1,
          loadedBytes: p.loaded,
          totalBytes: p.total,
          pct: p.pct,
        });
        if (p.pct >= 100) setDatasetState('RECONSTRUCTING');
      });

      setDatasetUploadResult(result);
      setDatasetState('READY');
    } catch (err: any) {
      console.error('Dataset zip upload error:', err);
      setDatasetError(err.message || 'Dataset zip upload failed');
      setDatasetState('FAILED');
    }
  };

  // Quick action: Load Pre-Verified Model Sample
  const handleUseSampleModel = (sampleId: string = 'resnet50_cifar10') => {
    const sample = samples.models.find((m) => m.id === sampleId);
    if (sample) {
      setModelUploadResult({
        upload_id: sample.id,
        filename: sample.name,
        size_bytes: (sample.size_mb || 94) * 1024 * 1024,
        sha256: sample.sha256,
        format: sample.format || 'pytorch_checkpoint',
        status: 'READY',
        source: 'PRE_VERIFIED_SAMPLE',
        staged_path: sample.path,
      });
      setModelSource('PRE_VERIFIED_SAMPLE');
      setModelState('READY');
      setModelError(null);
    } else {
      console.warn(`Sample model '${sampleId}' not found among available samples:`, samples.models);
    }
  };

  // Quick action: Load Pre-Verified Dataset Sample
  const handleUseSampleDataset = (sampleId: string = 'cifar10') => {
    const sample = samples.datasets.find((d) => d.id === sampleId);
    if (sample) {
      setDatasetUploadResult({
        upload_id: sample.id,
        folder_name: sample.name,
        file_count: sample.samples || 6,
        total_size_bytes: 162500000,
        manifest_hash: `${sample.id}_preverified_manifest_sha256`,
        detected_format: sample.format || 'cifar10_pickle',
        class_count: sample.classes || 10,
        splits: {
          train_count: sample.samples ? Math.floor(sample.samples * 0.8) : 50000,
          val_count: sample.samples ? Math.floor(sample.samples * 0.1) : 5000,
          test_count: sample.samples ? Math.floor(sample.samples * 0.1) : 5000,
          total_samples: sample.samples || 60000,
        },
        status: 'READY',
        source: 'PRE_VERIFIED_SAMPLE',
        staged_path: sample.path,
        structure_valid: true,
        validation_message: 'Pre-verified standard benchmark dataset',
      });
      setDatasetSource('PRE_VERIFIED_SAMPLE');
      setDatasetState('READY');
      setDatasetError(null);
    } else {
      console.warn(`Sample dataset '${sampleId}' not found among available samples:`, samples.datasets);
    }
  };

  // Trigger Analysis
  const handleAnalyzeClick = async () => {
    if (modelState !== 'READY' || datasetState !== 'READY') return;

    await onAnalyze({
      modelUploadId: modelSource === 'USER_UPLOAD' ? modelUploadResult?.upload_id : undefined,
      modelId: modelSource === 'PRE_VERIFIED_SAMPLE' ? modelUploadResult?.upload_id : undefined,
      modelPath: modelUploadResult?.staged_path,
      datasetUploadId: datasetSource === 'USER_UPLOAD' ? datasetUploadResult?.upload_id : undefined,
      datasetId: datasetSource === 'PRE_VERIFIED_SAMPLE' ? datasetUploadResult?.upload_id : undefined,
      datasetPath: datasetUploadResult?.staged_path,
      targetHardware,
      profile,
      calibSamples,
      testSamples,
    });
  };

  // Direct Optimize
  const handleDirectOptimizeClick = async () => {
    if (modelState !== 'READY' || datasetState !== 'READY') return;

    await onDirectOptimize({
      modelUploadId: modelSource === 'USER_UPLOAD' ? modelUploadResult?.upload_id : undefined,
      modelId: modelSource === 'PRE_VERIFIED_SAMPLE' ? modelUploadResult?.upload_id : undefined,
      modelPath: modelUploadResult?.staged_path,
      datasetUploadId: datasetSource === 'USER_UPLOAD' ? datasetUploadResult?.upload_id : undefined,
      datasetId: datasetSource === 'PRE_VERIFIED_SAMPLE' ? datasetUploadResult?.upload_id : undefined,
      datasetPath: datasetUploadResult?.staged_path,
      targetHardware,
      profile,
      calibSamples,
      testSamples,
      maxBudget,
    });
  };

  const isInputsReady = modelState === 'READY' && datasetState === 'READY';

  const hardwareOptions = [
    {
      id: 'raspberrypi5',
      name: 'Raspberry Pi 5',
      sub: 'ARM Cortex-A76 @ 2.4GHz • 8GB LPDDR4X',
      desc: 'Target device for edge vision inference and QDQ INT8 execution',
      badge: 'TARGET REVERIFY SUPPORTED',
      badgeColor: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    },
    {
      id: 'jetson_orin_nano',
      name: 'NVIDIA Jetson Orin Nano',
      sub: 'Ampere GPU (1024 CUDA) + 6-core ARM A78AE',
      desc: 'Embedded AI workstation with TensorRT and FP8/INT8 support',
      badge: 'GPU ACCELERATED',
      badgeColor: 'bg-indigo-50 text-indigo-700 border-indigo-200',
    },
    {
      id: 'stm32',
      name: 'STM32H7 (ARM Cortex-M7)',
      sub: 'ARM Cortex-M7 @ 480MHz • 1MB SRAM',
      desc: 'High-performance microcontroller target for INT8 quantized models',
      badge: 'MICROCONTROLLER',
      badgeColor: 'bg-amber-50 text-amber-700 border-amber-200',
    },
    {
      id: 'host_cpu',
      name: 'Host x86_64 CPU',
      sub: `${systemStatus?.host_environment?.cpu_count_logical || 12} Logical Cores • AVX2 / AVX-512`,
      desc: 'Direct host environment execution and latency validation',
      badge: 'HOST MEASURED',
      badgeColor: 'bg-slate-100 text-slate-700 border-slate-300',
    },
  ];

  const profileOptions = [
    {
      id: 'balanced',
      name: 'Balanced (Standard)',
      desc: 'Optimal compromise between accuracy preservation (≤1.0 pp loss), model size reduction, and edge latency.',
      icon: Sparkles,
      color: 'indigo',
    },
    {
      id: 'accuracy_first',
      name: 'Accuracy First',
      desc: 'Strict safety policy: rejects any candidate with >0.5 pp loss. Protects sensitive early convolution stages.',
      icon: ShieldCheck,
      color: 'emerald',
    },
    {
      id: 'size_first',
      name: 'Size First (Extreme Compression)',
      desc: 'Aggressive quantization (INT8/INT4 weights) and structured pruning to minimize flash & memory footprint.',
      icon: Zap,
      color: 'amber',
    },
  ];

  return (
    <div className="space-y-8 animate-fade-in max-w-7xl mx-auto pb-16">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 pb-4 border-b border-slate-200">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
              FRESH OPTIMIZATION
            </span>
            <span className="text-xs text-slate-400 font-mono">FILE & FOLDER INGESTION</span>
          </div>
          <h1 className="text-2xl md:text-3xl font-bold text-slate-900 tracking-tight">
            Configure Autonomous Optimization Job
          </h1>
          <p className="text-sm text-slate-600 mt-1">
            Provide a model file and dataset folder to launch an autonomous quantization search from zero.
          </p>
        </div>

        {/* Quick Sample Demo Loaders */}
        <div className="flex items-center flex-wrap gap-2">
          <button
            onClick={() => {
              handleUseSampleModel('mobilenetv3_sem');
              handleUseSampleDataset('semiconductor_defect');
            }}
            className="flex items-center gap-2 px-3 py-2 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 rounded-lg text-xs font-semibold border border-indigo-200 transition shadow-sm cursor-pointer"
            title="Load pre-verified MobileNetV3 ONNX + 9-Class Semiconductor Defect Dataset"
          >
            <Sparkles className="w-4 h-4 text-indigo-600" />
            Verified Semiconductor (MobileNetV3 + Defect Data)
          </button>
          <button
            onClick={() => {
              handleUseSampleModel('resnet50_cifar10');
              handleUseSampleDataset('cifar10');
            }}
            className="flex items-center gap-2 px-3 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-xs font-semibold border border-slate-300 transition shadow-sm cursor-pointer"
            title="Load pre-verified ResNet-50 + CIFAR-10 Benchmark"
          >
            <Sparkles className="w-4 h-4 text-slate-600" />
            Verified CV (ResNet-50 + CIFAR-10)
          </button>
        </div>
      </div>

      {/* Hidden File Inputs */}
      <input
        type="file"
        ref={modelFileInputRef}
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            handleModelFileSelected(e.target.files[0]);
          }
        }}
        accept=".pt,.pth,.onnx,.safetensors,.tflite,.bin"
        className="hidden"
      />

      <input
        type="file"
        ref={datasetFolderInputRef}
        onChange={(e) => {
          if (e.target.files) {
            handleDatasetFolderSelected(e.target.files);
          }
        }}
        // @ts-ignore
        webkitdirectory=""
        // @ts-ignore
        directory=""
        multiple
        className="hidden"
      />

      <input
        type="file"
        ref={datasetZipInputRef}
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            handleDatasetZipSelected(e.target.files[0]);
          }
        }}
        accept=".zip,.tar.gz,.tgz"
        className="hidden"
      />

      {/* Primary Input Ingestion Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* ========================================================================= */}
        {/* 1. MODEL INPUT FILE */}
        {/* ========================================================================= */}
        <GlassCard className="p-6 relative overflow-hidden flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="p-2.5 bg-indigo-50 text-indigo-700 rounded-xl border border-indigo-100">
                  <FileCode className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-base font-bold text-slate-900">MODEL INPUT FILE</h2>
                  <p className="text-xs text-slate-500">Single neural network checkpoint file</p>
                </div>
              </div>
              <SourceBadge source={modelSource} />
            </div>

            {/* Dropzone or Ready State */}
            {modelState === 'READY' && modelUploadResult ? (
              <div className="p-5 bg-indigo-50/40 rounded-xl border border-indigo-200/80 space-y-3">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-white rounded-lg border border-indigo-100 shadow-sm text-indigo-600">
                      <FileCheck2 className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-slate-900 truncate max-w-xs">
                        {modelUploadResult.filename}
                      </h3>
                      <p className="text-xs text-slate-500 font-mono">
                        {modelUploadResult.format.toUpperCase()} • {(modelUploadResult.size_bytes / (1024 * 1024)).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                  <span className="px-2.5 py-1 bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-full text-xs font-semibold flex items-center gap-1">
                    <Check className="w-3.5 h-3.5" /> READY
                  </span>
                </div>

                <div className="pt-2 border-t border-indigo-100 grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-slate-400 block">SHA-256 Checksum:</span>
                    <span className="font-mono text-slate-700 truncate block text-[11px]" title={modelUploadResult.sha256}>
                      {modelUploadResult.sha256.substring(0, 16)}...
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-400 block">Upload Staging ID:</span>
                    <span className="font-mono text-slate-700 text-[11px]">
                      {modelUploadResult.upload_id}
                    </span>
                  </div>
                </div>

                <div className="pt-2 flex items-center justify-between gap-2 flex-wrap">
                  <button
                    onClick={() => modelFileInputRef.current?.click()}
                    className="text-xs font-medium text-indigo-600 hover:text-indigo-800 transition"
                  >
                    Replace Model File
                  </button>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleUseSampleModel('mobilenetv3_sem')}
                      className="text-xs font-medium text-slate-500 hover:text-indigo-600 transition"
                    >
                      Sample MobileNetV3
                    </button>
                    <span className="text-slate-300">•</span>
                    <button
                      onClick={() => handleUseSampleModel('resnet50_cifar10')}
                      className="text-xs font-medium text-slate-500 hover:text-indigo-600 transition"
                    >
                      Sample ResNet-50
                    </button>
                  </div>
                </div>
              </div>
            ) : modelState === 'UPLOADING' || modelState === 'VERIFYING' ? (
              <div className="p-8 border-2 border-indigo-200 border-dashed rounded-xl text-center space-y-4 bg-indigo-50/20">
                <RefreshCw className="w-8 h-8 text-indigo-600 animate-spin mx-auto" />
                <div>
                  <h3 className="text-sm font-bold text-slate-900">
                    {modelState === 'UPLOADING' ? 'Uploading Model File...' : 'Verifying Checksum & Format...'}
                  </h3>
                  <p className="text-xs text-slate-500 font-mono mt-1">
                    {(modelProgress.loaded / (1024 * 1024)).toFixed(2)} MB / {(modelProgress.total / (1024 * 1024)).toFixed(2)} MB ({modelProgress.pct}%)
                  </p>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-indigo-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${modelProgress.pct}%` }}
                  />
                </div>
              </div>
            ) : (
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsModelDragOver(true);
                }}
                onDragLeave={() => setIsModelDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsModelDragOver(false);
                  if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                    handleModelFileSelected(e.dataTransfer.files[0]);
                  }
                }}
                className={`p-8 border-2 border-dashed rounded-xl text-center transition flex flex-col items-center justify-center gap-3 cursor-pointer ${
                  isModelDragOver
                    ? 'border-indigo-500 bg-indigo-50/60 shadow-inner'
                    : 'border-slate-200 hover:border-indigo-400 bg-slate-50/50 hover:bg-indigo-50/20'
                }`}
                onClick={() => modelFileInputRef.current?.click()}
              >
                <div className="p-3 bg-white rounded-full border border-slate-200 shadow-sm text-slate-600">
                  <UploadCloud className="w-6 h-6 text-indigo-600" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-slate-900">Drag & Drop Model File</h3>
                  <p className="text-xs text-slate-500 mt-1">
                    Supports <span className="font-semibold text-slate-700">.pt, .pth, .onnx, .safetensors, .tflite</span>
                  </p>
                </div>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    modelFileInputRef.current?.click();
                  }}
                  className="px-4 py-2 bg-white hover:bg-slate-50 text-slate-800 border border-slate-300 rounded-lg text-xs font-bold shadow-sm transition"
                >
                  Browse Model File
                </button>
              </div>
            )}

            {modelError && (
              <div className="mt-3 p-3 bg-rose-50 border border-rose-200 rounded-lg flex items-center gap-2 text-xs text-rose-700">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{modelError}</span>
              </div>
            )}
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span>Single file input</span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => handleUseSampleModel('mobilenetv3_sem')}
                className="text-indigo-600 hover:underline font-medium"
              >
                Sample MobileNetV3
              </button>
              <span className="text-slate-300">•</span>
              <button
                onClick={() => handleUseSampleModel('resnet50_cifar10')}
                className="text-indigo-600 hover:underline font-medium"
              >
                Sample ResNet-50
              </button>
            </div>
          </div>
        </GlassCard>

        {/* ========================================================================= */}
        {/* 2. DATASET FOLDER */}
        {/* ========================================================================= */}
        <GlassCard className="p-6 relative overflow-hidden flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="p-2.5 bg-emerald-50 text-emerald-700 rounded-xl border border-emerald-100">
                  <FolderUp className="w-5 h-5" />
                </div>
                <div>
                  <h2 className="text-base font-bold text-slate-900">DATASET FOLDER</h2>
                  <p className="text-xs text-slate-500">Complete dataset folder (nested structure preserved)</p>
                </div>
              </div>
              <SourceBadge source={datasetSource} />
            </div>

            {/* Dropzone or Ready State */}
            {datasetState === 'READY' && datasetUploadResult ? (
              <div className="p-5 bg-emerald-50/40 rounded-xl border border-emerald-200/80 space-y-3">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-white rounded-lg border border-emerald-100 shadow-sm text-emerald-600">
                      <Database className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-slate-900 truncate max-w-xs">
                        {datasetUploadResult.folder_name}
                      </h3>
                      <p className="text-xs text-slate-500 font-mono">
                        {datasetUploadResult.detected_format.toUpperCase()} • {datasetUploadResult.file_count} files • {(datasetUploadResult.total_size_bytes / (1024 * 1024)).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                  <span className="px-2.5 py-1 bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-full text-xs font-semibold flex items-center gap-1">
                    <Check className="w-3.5 h-3.5" /> READY
                  </span>
                </div>

                <div className="pt-2 border-t border-emerald-100 grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-slate-400 block">Classes & Splits:</span>
                    <span className="font-semibold text-slate-700 block">
                      {datasetUploadResult.class_count > 0 ? `${datasetUploadResult.class_count} classes` : 'Flat batches'}
                    </span>
                  </div>
                  <div>
                    <span className="text-slate-400 block">Manifest Hash:</span>
                    <span className="font-mono text-slate-700 truncate block text-[11px]" title={datasetUploadResult.manifest_hash}>
                      {datasetUploadResult.manifest_hash.substring(0, 16)}...
                    </span>
                  </div>
                </div>

                <div className="pt-2 flex items-center justify-between">
                  <button
                    onClick={() => datasetFolderInputRef.current?.click()}
                    className="text-xs font-medium text-emerald-600 hover:text-emerald-800 transition"
                  >
                    Replace Dataset Folder
                  </button>
                  <button
                    onClick={() => handleUseSampleDataset()}
                    className="text-xs font-medium text-slate-500 hover:text-slate-700 transition"
                  >
                    Use Verified CIFAR-10
                  </button>
                </div>
              </div>
            ) : datasetState === 'UPLOADING' || datasetState === 'RECONSTRUCTING' ? (
              <div className="p-8 border-2 border-emerald-200 border-dashed rounded-xl text-center space-y-4 bg-emerald-50/20">
                <RefreshCw className="w-8 h-8 text-emerald-600 animate-spin mx-auto" />
                <div>
                  <h3 className="text-sm font-bold text-slate-900">
                    {datasetState === 'UPLOADING' ? 'Uploading Dataset Files...' : 'Reconstructing Hierarchy & Validating...'}
                  </h3>
                  <p className="text-xs text-slate-500 font-mono mt-1">
                    {datasetProgress.loadedFiles} / {datasetProgress.totalFiles} files • {(datasetProgress.loadedBytes / (1024 * 1024)).toFixed(2)} MB / {(datasetProgress.totalBytes / (1024 * 1024)).toFixed(2)} MB ({datasetProgress.pct}%)
                  </p>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-emerald-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${datasetProgress.pct}%` }}
                  />
                </div>
              </div>
            ) : (
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDatasetDragOver(true);
                }}
                onDragLeave={() => setIsDatasetDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDatasetDragOver(false);
                  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                    handleDatasetFolderSelected(e.dataTransfer.files);
                  }
                }}
                className={`p-8 border-2 border-dashed rounded-xl text-center transition flex flex-col items-center justify-center gap-3 cursor-pointer ${
                  isDatasetDragOver
                    ? 'border-emerald-500 bg-emerald-50/60 shadow-inner'
                    : 'border-slate-200 hover:border-emerald-400 bg-slate-50/50 hover:bg-emerald-50/20'
                }`}
                onClick={() => datasetFolderInputRef.current?.click()}
              >
                <div className="p-3 bg-white rounded-full border border-slate-200 shadow-sm text-slate-600">
                  <FolderUp className="w-6 h-6 text-emerald-600" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-slate-900">Drag & Drop Dataset Folder</h3>
                  <p className="text-xs text-slate-500 mt-1">
                    Preserves nested folders, batch pickles, and subdirectories
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      datasetFolderInputRef.current?.click();
                    }}
                    className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-xs font-bold shadow-sm transition"
                  >
                    Choose Dataset Folder
                  </button>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      datasetZipInputRef.current?.click();
                    }}
                    className="px-3 py-2 bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 rounded-lg text-xs font-medium shadow-sm transition"
                    title="Upload .zip archive fallback"
                  >
                    <Archive className="w-3.5 h-3.5 inline mr-1" />
                    ZIP Fallback
                  </button>
                </div>
              </div>
            )}

            {datasetError && (
              <div className="mt-3 p-3 bg-rose-50 border border-rose-200 rounded-lg flex items-center gap-2 text-xs text-rose-700">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                <span>{datasetError}</span>
              </div>
            )}
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span>Folder structure preserved</span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => handleUseSampleDataset('semiconductor_defect')}
                className="text-emerald-600 hover:underline font-medium"
              >
                Sample Semiconductor Data
              </button>
              <span className="text-slate-300">•</span>
              <button
                onClick={() => handleUseSampleDataset('cifar10')}
                className="text-emerald-600 hover:underline font-medium"
              >
                Sample CIFAR-10
              </button>
            </div>
          </div>
        </GlassCard>
      </div>

      {/* Target Hardware Selection */}
      <GlassCard className="p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-slate-100 text-slate-700 rounded-xl border border-slate-200">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-900">TARGET HARDWARE PLATFORM</h2>
              <p className="text-xs text-slate-500">Select target runtime execution environment</p>
            </div>
          </div>
          <SourceBadge source="CONFIGURED" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {hardwareOptions.map((hw) => {
            const isSelected = targetHardware === hw.id;
            return (
              <button
                key={hw.id}
                type="button"
                onClick={() => setTargetHardware(hw.id)}
                className={`p-4 rounded-xl text-left border transition-all duration-200 flex flex-col justify-between ${
                  isSelected
                    ? 'border-indigo-600 bg-indigo-50/30 ring-2 ring-indigo-500/20 shadow-sm'
                    : 'border-slate-200 hover:border-slate-300 bg-white/70 hover:bg-slate-50'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-bold text-sm text-slate-900">{hw.name}</span>
                    <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-medium border ${hw.badgeColor}`}>
                      {hw.badge}
                    </span>
                  </div>
                  <p className="text-xs text-slate-600 font-medium mb-1">{hw.sub}</p>
                  <p className="text-[11px] text-slate-500">{hw.desc}</p>
                </div>
                <div className="mt-3 pt-2 border-t border-slate-100 flex items-center justify-between text-xs">
                  <span className="text-slate-400">Target ID: {hw.id}</span>
                  {isSelected && <Check className="w-4 h-4 text-indigo-600 font-bold" />}
                </div>
              </button>
            );
          })}
        </div>
      </GlassCard>

      {/* Optimization Profile Selection */}
      <GlassCard className="p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-slate-100 text-slate-700 rounded-xl border border-slate-200">
              <Sliders className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-900">OPTIMIZATION OBJECTIVE PROFILE</h2>
              <p className="text-xs text-slate-500">Directs search candidate prioritization and safety policy</p>
            </div>
          </div>
          <SourceBadge source="POLICY" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {profileOptions.map((p) => {
            const isSelected = profile === p.id;
            const Icon = p.icon;
            return (
              <button
                key={p.id}
                type="button"
                onClick={() => setProfile(p.id)}
                className={`p-4 rounded-xl text-left border transition-all duration-200 flex flex-col justify-between ${
                  isSelected
                    ? 'border-indigo-600 bg-indigo-50/30 ring-2 ring-indigo-500/20 shadow-sm'
                    : 'border-slate-200 hover:border-slate-300 bg-white/70 hover:bg-slate-50'
                }`}
              >
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <div className="p-1.5 rounded-lg bg-indigo-50 text-indigo-700 border border-indigo-100">
                      <Icon className="w-4 h-4" />
                    </div>
                    <span className="font-bold text-sm text-slate-900">{p.name}</span>
                  </div>
                  <p className="text-xs text-slate-600 leading-relaxed">{p.desc}</p>
                </div>
                <div className="mt-3 pt-2 border-t border-slate-100 flex items-center justify-between text-xs">
                  <span className="text-slate-400 font-mono text-[11px]">profile={p.id}</span>
                  {isSelected && <Check className="w-4 h-4 text-indigo-600 font-bold" />}
                </div>
              </button>
            );
          })}
        </div>
      </GlassCard>

      {/* Advanced Settings (Collapsible) */}
      <GlassCard className="overflow-hidden">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full p-4 flex items-center justify-between hover:bg-slate-50/50 transition"
        >
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-800">
            <Sliders className="w-4 h-4 text-slate-500" />
            <span>Advanced Search Hyperparameters</span>
            <span className="text-xs text-slate-400 font-normal">(Calibration samples, search budget, thresholds)</span>
          </div>
          {showAdvanced ? <ChevronUp className="w-4 h-4 text-slate-500" /> : <ChevronDown className="w-4 h-4 text-slate-500" />}
        </button>

        {showAdvanced && (
          <div className="p-6 border-t border-slate-100 bg-slate-50/30 grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">
                Calibration Batch Size
              </label>
              <input
                type="number"
                value={calibSamples}
                onChange={(e) => setCalibSamples(Number(e.target.value))}
                min={32}
                max={1000}
                className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 font-mono"
              />
              <span className="text-[11px] text-slate-500 mt-1 block">Number of representative samples for PTQ activation clipping</span>
            </div>

            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">
                Evaluation Test Samples
              </label>
              <input
                type="number"
                value={testSamples}
                onChange={(e) => setTestSamples(Number(e.target.value))}
                min={100}
                max={10000}
                className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 font-mono"
              />
              <span className="text-[11px] text-slate-500 mt-1 block">Samples evaluated per candidate to measure Top-1 accuracy loss</span>
            </div>

            <div>
              <label className="block text-xs font-bold text-slate-700 mb-1">
                Search Candidate Budget
              </label>
              <input
                type="number"
                value={maxBudget}
                onChange={(e) => setMaxBudget(Number(e.target.value))}
                min={1}
                max={15}
                className="w-full px-3 py-2 bg-white border border-slate-200 rounded-lg text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 font-mono"
              />
              <span className="text-[11px] text-slate-500 mt-1 block">Maximum optimization candidates evaluated in parallel search</span>
            </div>
          </div>
        )}
      </GlassCard>

      {/* Primary Action Button Bar */}
      <div className="p-6 bg-white/90 backdrop-blur border border-slate-200 rounded-2xl shadow-lg flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className={`w-3 h-3 rounded-full ${modelState === 'READY' ? 'bg-emerald-500 animate-pulse' : 'bg-slate-300'}`} />
            <span className="text-xs font-semibold text-slate-700">
              Model: {modelState === 'READY' ? 'Ready' : 'Not Loaded'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className={`w-3 h-3 rounded-full ${datasetState === 'READY' ? 'bg-emerald-500 animate-pulse' : 'bg-slate-300'}`} />
            <span className="text-xs font-semibold text-slate-700">
              Dataset: {datasetState === 'READY' ? 'Ready' : 'Not Loaded'}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3 w-full md:w-auto">
          <button
            onClick={handleAnalyzeClick}
            disabled={!isInputsReady || isProcessing}
            className={`flex-1 md:flex-initial px-6 py-3 rounded-xl font-bold text-sm shadow-md transition-all flex items-center justify-center gap-2 ${
              isInputsReady && !isProcessing
                ? 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-200 hover:shadow-lg scale-100 hover:scale-[1.02]'
                : 'bg-slate-200 text-slate-400 cursor-not-allowed'
            }`}
          >
            {isProcessing ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                Analyzing Inputs...
              </>
            ) : (
              <>
                <Search className="w-4 h-4" />
                ANALYZE MODEL & DATASET
              </>
            )}
          </button>

          <button
            onClick={handleDirectOptimizeClick}
            disabled={!isInputsReady || isProcessing}
            className={`px-6 py-3 rounded-xl font-bold text-sm shadow-md transition-all flex items-center justify-center gap-2 ${
              isInputsReady && !isProcessing
                ? 'bg-slate-900 hover:bg-black text-white shadow-slate-300'
                : 'bg-slate-100 text-slate-400 border border-slate-200 cursor-not-allowed'
            }`}
          >
            <Play className="w-4 h-4 text-emerald-400 fill-emerald-400" />
            DIRECT OPTIMIZE
          </button>
        </div>
      </div>
    </div>
  );
};

export default NewOptimizationPage;
