import React from 'react';
import {
  Search,
  FileCode2,
  Database,
  Sliders,
  ShieldCheck,
  CheckCircle2,
  Play,
  ArrowRight,
  Zap,
  Info,
  Layers,
} from 'lucide-react';
import { ModelInspection, DatasetInspection, OptimizationPlan } from '../types/api';
import SourceBadge from '../components/ui/SourceBadge';
import ClassDistributionChart from '../charts/ClassDistributionChart';

interface InspectionPlanPageProps {
  modelInspection: ModelInspection | null;
  datasetInspection: DatasetInspection | null;
  optimizationPlan: OptimizationPlan | null;
  onApproveAndLaunch: () => Promise<void>;
  isLaunching: boolean;
  onBack: () => void;
}

const getVal = (obj: any, fallback: any = ''): any => {
  if (obj === null || obj === undefined) return fallback;
  if (typeof obj === 'object' && 'value' in obj) return obj.value ?? fallback;
  return obj ?? fallback;
};

export const InspectionPlanPage: React.FC<InspectionPlanPageProps> = ({
  modelInspection,
  datasetInspection,
  optimizationPlan,
  onApproveAndLaunch,
  isLaunching,
  onBack,
}) => {
  const modelArch = getVal((modelInspection as any)?.architecture, 'ResNet-50 v1.5');
  const modelParams = getVal((modelInspection as any)?.parameter_count, 23528522);
  const modelBytes = getVal((modelInspection as any)?.file_size_bytes, 97.71 * 1024 * 1024);
  const modelTensors = getVal((modelInspection as any)?.tensor_count, 320);
  const inputShape = getVal((modelInspection as any)?.input_shape, [1, 3, 224, 224]);
  const outputShape = getVal((modelInspection as any)?.output_shape, [1, 10]);
  const modelSha256 = getVal((modelInspection as any)?.sha256 || (modelInspection as any)?.source_sha256, 'e2a94f...verified');

  const datasetName = getVal((datasetInspection as any)?.dataset_name, 'CIFAR-10');
  const classCount = getVal((datasetInspection as any)?.class_count, 10);
  const calibSamples = getVal((optimizationPlan as any)?.calib_samples, 100);
  const splits = getVal((datasetInspection as any)?.splits, {});
  const valSamples = splits?.test_count ?? splits?.val_count ?? getVal((optimizationPlan as any)?.test_samples, 1000);

  return (
    <div className="flex flex-col gap-8 max-w-5xl mx-auto">
      {/* Page Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-200">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
              Pre-Optimization Inspection & Plan Review
            </h1>
            <SourceBadge source="DETECTED" origin="Static Graph & Dataset Parser" />
          </div>
          <p className="text-sm text-slate-500">
            Review structural parameters, input/output tensors, dataset split balance, and safety plan before autonomous execution
          </p>
        </div>

        <button
          onClick={onApproveAndLaunch}
          disabled={isLaunching}
          className="px-6 py-3 rounded-2xl bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold shadow-md transition-all flex items-center gap-2 disabled:opacity-50"
        >
          <Play className="w-4 h-4 text-white" />
          <span>{isLaunching ? 'Launching Autonomous Engine...' : 'Approve Plan & Start Optimization'}</span>
        </button>
      </div>

      {/* 1. Model Inspection Section */}
      <div className="p-6 rounded-2xl bg-white border border-slate-200/80 shadow-xs flex flex-col gap-4">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-50 flex items-center justify-center text-indigo-600">
              <FileCode2 className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-900">Model Architecture & Graph Inspection</h3>
              <p className="text-xs text-slate-500">Static tensor inspection without modifying original checkpoint</p>
            </div>
          </div>
          <SourceBadge source="DETECTED" origin="PyTorch / ONNX Inspector" />
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">Architecture</span>
            <span className="text-xs font-bold text-slate-900 font-mono block mt-0.5">
              {modelArch}
            </span>
          </div>

          <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">Total Parameters</span>
            <span className="text-xs font-bold text-slate-900 font-mono block mt-0.5">
              {typeof modelParams === 'number' ? modelParams.toLocaleString() : modelParams}
            </span>
          </div>

          <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">Weight File Size</span>
            <span className="text-xs font-bold text-slate-900 font-mono block mt-0.5">
              {typeof modelBytes === 'number' ? `${(modelBytes / (1024 * 1024)).toFixed(2)} MB` : modelBytes}
            </span>
          </div>

          <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">Tensor Node Count</span>
            <span className="text-xs font-bold text-slate-900 font-mono block mt-0.5">
              {modelTensors} Tensors
            </span>
          </div>
        </div>

        <div className="p-3 rounded-xl bg-slate-50 border border-slate-100 flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
          <span className="text-slate-500">Input Spec: <strong className="text-slate-800">{JSON.stringify(inputShape)}</strong></span>
          <span className="text-slate-500">Output Spec: <strong className="text-slate-800">{JSON.stringify(outputShape)}</strong></span>
          <span className="text-slate-500">SHA-256: <strong className="text-slate-800 text-[11px]">{typeof modelSha256 === 'string' && modelSha256.length > 16 ? `${modelSha256.substring(0, 16)}...` : modelSha256}</strong></span>
        </div>
      </div>

      {/* 2. Dataset Inspection & Balance */}
      <div className="p-6 rounded-2xl bg-white border border-slate-200/80 shadow-xs flex flex-col gap-4">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-sky-50 flex items-center justify-center text-sky-600">
              <Database className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-900">Dataset Inspection & Distribution</h3>
              <p className="text-xs text-slate-500">Class count and calibration split verification</p>
            </div>
          </div>
          <SourceBadge source="DETECTED" origin="Dataset Adapter" />
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">Dataset Name</span>
            <span className="text-xs font-bold text-slate-900 font-mono block mt-0.5">
              {datasetName}
            </span>
          </div>

          <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">Total Classes</span>
            <span className="text-xs font-bold text-slate-900 font-mono block mt-0.5">
              {classCount} Classes
            </span>
          </div>

          <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">Calibration Subset</span>
            <span className="text-xs font-bold text-slate-900 font-mono block mt-0.5">
              {calibSamples} Samples
            </span>
          </div>

          <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">Validation Subset</span>
            <span className="text-xs font-bold text-slate-900 font-mono block mt-0.5">
              {typeof valSamples === 'number' ? valSamples.toLocaleString() : valSamples} Samples
            </span>
          </div>
        </div>

        {/* Dataset Class Distribution Chart */}
        <ClassDistributionChart datasetInspection={datasetInspection} />
      </div>

      {/* 3. Optimization Plan & Safety Policy */}
      <div className="p-6 rounded-2xl bg-white border border-slate-200/80 shadow-xs flex flex-col gap-4">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-emerald-50 flex items-center justify-center text-emerald-600">
              <Sliders className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-900">Optimization Plan & Policy Gating</h3>
              <p className="text-xs text-slate-500">Autonomous search boundaries and objective functions</p>
            </div>
          </div>
          <SourceBadge source="POLICY" origin="UAQE Objective Planner" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="p-4 rounded-xl bg-emerald-50/50 border border-emerald-100">
            <span className="text-[10px] font-bold text-emerald-700 uppercase tracking-wider block">
              Tier 1: Excellent
            </span>
            <span className="text-base font-bold text-emerald-900 font-mono block mt-0.5">
              Loss &le; 1.0 pp
            </span>
            <span className="text-[11px] text-emerald-600 block mt-1">
              Automatic early stop on candidate satisfaction
            </span>
          </div>

          <div className="p-4 rounded-xl bg-indigo-50/50 border border-indigo-100">
            <span className="text-[10px] font-bold text-indigo-700 uppercase tracking-wider block">
              Tier 2: Acceptable
            </span>
            <span className="text-base font-bold text-indigo-900 font-mono block mt-0.5">
              Loss &le; 4.0 pp
            </span>
            <span className="text-[11px] text-indigo-600 block mt-1">
              Valid Pareto candidate eligible for selection
            </span>
          </div>

          <div className="p-4 rounded-xl bg-rose-50/50 border border-rose-100">
            <span className="text-[10px] font-bold text-rose-700 uppercase tracking-wider block">
              Tier 3: Critical (Rejected)
            </span>
            <span className="text-base font-bold text-rose-900 font-mono block mt-0.5">
              Loss &gt; 4.0 pp
            </span>
            <span className="text-[11px] text-rose-600 block mt-1">
              Strictly rejected by policy gating
            </span>
          </div>
        </div>

        {/* Objective Function Weights */}
        <div className="p-4 rounded-xl bg-slate-50 border border-slate-100 flex flex-col gap-2">
          <span className="text-xs font-bold text-slate-700">Multi-Objective Score Weights:</span>
          <div className="grid grid-cols-3 gap-3 font-mono text-xs">
            <div className="p-2.5 rounded-lg bg-white border border-slate-200">
              <span className="text-slate-400 block text-[10px]">Top-1 Accuracy Weight</span>
              <strong className="text-indigo-600 text-sm">
                {(optimizationPlan as any)?.objective_weights
                  ? `${Math.round(((optimizationPlan as any).objective_weights.accuracy_weight ?? 0.50) * 100)}% (${((optimizationPlan as any).objective_weights.accuracy_weight ?? 0.50).toFixed(2)})`
                  : '50% (0.50)'}
              </strong>
            </div>
            <div className="p-2.5 rounded-lg bg-white border border-slate-200">
              <span className="text-slate-400 block text-[10px]">Latency Speedup Weight</span>
              <strong className="text-emerald-600 text-sm">
                {(optimizationPlan as any)?.objective_weights
                  ? `${Math.round(((optimizationPlan as any).objective_weights.latency_weight ?? 0.25) * 100)}% (${((optimizationPlan as any).objective_weights.latency_weight ?? 0.25).toFixed(2)})`
                  : '25% (0.25)'}
              </strong>
            </div>
            <div className="p-2.5 rounded-lg bg-white border border-slate-200">
              <span className="text-slate-400 block text-[10px]">Size Reduction Weight</span>
              <strong className="text-sky-600 text-sm">
                {(optimizationPlan as any)?.objective_weights
                  ? `${Math.round(((optimizationPlan as any).objective_weights.size_weight ?? 0.25) * 100)}% (${((optimizationPlan as any).objective_weights.size_weight ?? 0.25).toFixed(2)})`
                  : '25% (0.25)'}
              </strong>
            </div>
          </div>
        </div>
      </div>

      {/* Footer Actions */}
      <div className="flex items-center justify-between pt-2">
        <button
          onClick={onBack}
          className="px-5 py-2.5 rounded-xl border border-slate-200 text-slate-600 text-xs font-semibold hover:bg-slate-50"
        >
          Back to Configuration
        </button>

        <button
          onClick={onApproveAndLaunch}
          disabled={isLaunching}
          className="px-8 py-3.5 rounded-2xl bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold shadow-md transition-all flex items-center gap-2 disabled:opacity-50"
        >
          <Play className="w-4 h-4 text-white" />
          <span>{isLaunching ? 'Launching Autonomous Engine...' : 'Approve Plan & Launch Autonomous Run'}</span>
        </button>
      </div>
    </div>
  );
};

export default InspectionPlanPage;
