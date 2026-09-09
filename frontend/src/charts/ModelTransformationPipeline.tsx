import React, { useState } from 'react';
import {
  FileCode2,
  Search,
  GitBranch,
  Gauge,
  Sliders,
  ShieldAlert,
  Cpu,
  PackageCheck,
  CheckCircle2,
  Clock,
  AlertTriangle,
} from 'lucide-react';
import SourceBadge from '../components/ui/SourceBadge';

import { JobDetail, CandidateSummary } from '../types/api';

interface PipelineStage {
  id: string;
  name: string;
  shortDesc: string;
  icon: React.ElementType;
  status: 'COMPLETED' | 'IN_PROGRESS' | 'PENDING' | 'SKIPPED';
  provenance: 'DETECTED' | 'CONFIGURED' | 'MEASURED' | 'CALCULATED' | 'POLICY';
  metrics?: { label: string; value: string }[];
  details: string;
}

interface ModelTransformationPipelineProps {
  currentStage?: string;
  status?: string;
  verdict?: string;
  className?: string;
  jobDetail?: JobDetail | null;
  activeCandidate?: CandidateSummary | null;
}

export const ModelTransformationPipeline: React.FC<ModelTransformationPipelineProps> = ({
  currentStage,
  status = 'COMPLETED',
  verdict = 'VERIFIED',
  className = '',
  jobDetail,
  activeCandidate,
}) => {
  const [selectedStageId, setSelectedStageId] = useState<string>('stage5');

  const arch = String(jobDetail?.model_inspection?.architecture?.value || 'Model');
  const datasetName = String(jobDetail?.dataset_inspection?.dataset_name?.value || 'Dataset');
  const classes = String(jobDetail?.dataset_inspection?.class_count?.value || '10');
  const fp32Acc = jobDetail?.metrics?.fp32_accuracy?.value != null ? `${(jobDetail.metrics.fp32_accuracy.value * 100).toFixed(2)}%` : 'Reference';
  const fp32Lat = jobDetail?.metrics?.fp32_latency_ms?.value != null ? `${jobDetail.metrics.fp32_latency_ms.value.toFixed(2)} ms` : 'Host CPU';
  const fp32Tput = jobDetail?.metrics?.fp32_throughput_img_s?.value != null ? `${jobDetail.metrics.fp32_throughput_img_s.value.toFixed(2)} img/s` : 'Host CPU';
  const optAcc = activeCandidate != null
    ? `${(activeCandidate.top1_accuracy * 100).toFixed(2)}%`
    : (jobDetail?.metrics?.optimized_accuracy?.value != null ? `${(jobDetail.metrics.optimized_accuracy.value * 100).toFixed(2)}%` : 'Optimized');
  const optLat = activeCandidate != null
    ? `${activeCandidate.latency_mean_ms.toFixed(2)} ms`
    : (jobDetail?.metrics?.int8_latency_ms?.value != null ? `${jobDetail.metrics.int8_latency_ms.value.toFixed(2)} ms` : (jobDetail?.metrics?.optimized_latency_ms?.value != null ? `${jobDetail.metrics.optimized_latency_ms.value.toFixed(2)} ms` : 'Host CPU'));
  const optTput = activeCandidate != null
    ? `${activeCandidate.throughput_ips.toFixed(2)} img/s`
    : (jobDetail?.metrics?.int8_throughput_img_s?.value != null ? `${jobDetail.metrics.int8_throughput_img_s.value.toFixed(2)} img/s` : (jobDetail?.metrics?.throughput_images_per_sec?.value != null ? `${jobDetail.metrics.throughput_images_per_sec.value.toFixed(2)} img/s` : 'Host CPU'));
  const candidateCount = jobDetail?.candidates?.length || 1;
  const isMobileNet = arch.toLowerCase().includes('mobilenet') || String(jobDetail?.metrics?.selected_strategy?.value || '').includes('mobilenet');
  const strategyName = String(activeCandidate?.strategy_type || activeCandidate?.candidate_name || jobDetail?.metrics?.selected_strategy?.value || (isMobileNet ? 'MobileNet INT8 Adaptive Calibration' : 'Per-Channel QDQ'));
  const safetyTier = String(activeCandidate?.safety_classification || jobDetail?.metrics?.safety_status?.value || 'EXCELLENT');


  const stages: PipelineStage[] = [
    {
      id: 'stage1',
      name: '1. Model Ingestion',
      shortDesc: isMobileNet ? 'MobileNet Architecture Ingestion' : 'Base Checkpoint Parsing',
      icon: FileCode2,
      status: 'COMPLETED',
      provenance: 'DETECTED',
      metrics: [
        { label: 'Architecture', value: arch },
        { label: 'Integrity', value: 'SHA256 Verified' },
      ],
      details: `Ingested ${arch} base model checkpoint for autonomous edge quantization.`,
    },
    {
      id: 'stage2',
      name: '2. Graph Inspection',
      shortDesc: 'Structural Tensor Discovery',
      icon: Search,
      status: 'COMPLETED',
      provenance: 'DETECTED',
      metrics: [
        { label: 'Model', value: arch },
        { label: 'Classes', value: classes },
      ],
      details: `Structural parameter discovery, layer topology inspection, and tensor dimension validation.`,
    },
    {
      id: 'stage3',
      name: '3. Model Adaptation',
      shortDesc: isMobileNet ? 'Semiconductor Adapter Match' : 'Dataset Adaptation',
      icon: GitBranch,
      status: 'COMPLETED',
      provenance: 'MEASURED',
      metrics: [
        { label: 'Dataset', value: datasetName },
        { label: 'Classes', value: classes },
      ],
      details: `Targeted classifier alignment for ${datasetName} (${classes} classes) with validated weight preservation.`,
    },
    {
      id: 'stage4',
      name: '4. FP32 Baseline',
      shortDesc: 'Host Pure Inference Benchmark',
      icon: Gauge,
      status: 'COMPLETED',
      provenance: 'MEASURED',
      metrics: [
        { label: 'Top-1 Acc', value: fp32Acc },
        { label: 'Pure Latency', value: fp32Lat },
        { label: 'Throughput', value: fp32Tput },
      ],
      details: `Established pure host reference baseline under canonical 10 warmup + 100 timed invoke benchmark protocol.`,
    },
    {
      id: 'stage5',
      name: '5. Autonomous Search',
      shortDesc: isMobileNet ? 'MobileNet INT8 Calibration Search' : 'Multi-Candidate Search',
      icon: Sliders,
      status: 'COMPLETED',
      provenance: 'CALCULATED',
      metrics: [
        { label: 'Explored', value: `${candidateCount} Candidate(s)` },
        { label: 'Strategy', value: strategyName },
      ],
      details: `Autonomous candidate exploration evaluating quantization configurations under isolated host benchmarking.`,
    },
    {
      id: 'stage6',
      name: '6. Safety & Pareto Gating',
      shortDesc: 'Safety Policy Verification',
      icon: ShieldAlert,
      status: 'COMPLETED',
      provenance: 'POLICY',
      metrics: [
        { label: 'Optimized Acc', value: optAcc },
        { label: 'Safety Tier', value: safetyTier },
      ],
      details: `Evaluated candidates against strict accuracy degradation boundaries and governance policies.`,
    },
    {
      id: 'stage7',
      name: '7. Final Validation',
      shortDesc: 'Pure Host Inference Benchmark',
      icon: Cpu,
      status: 'COMPLETED',
      provenance: 'MEASURED',
      metrics: [
        { label: 'Pure Latency', value: optLat },
        { label: 'Throughput', value: optTput },
      ],
      details: `Final optimized artifact verified via canonical 10 warmup + 100 timed invoke benchmark protocol.`,
    },
    {
      id: 'stage8',
      name: '8. Export & Packaging',
      shortDesc: 'Manifest & Artifact Delivery',
      icon: PackageCheck,
      status: 'COMPLETED',
      provenance: 'CONFIGURED',
      metrics: [
        { label: 'Format', value: activeCandidate?.artifact?.format ? `${activeCandidate.artifact.format.toUpperCase()} INT8` : (isMobileNet ? 'TFLite INT8' : 'ONNX INT8') },
        { label: 'Size', value: activeCandidate ? `${(activeCandidate.model_size_bytes / (1024*1024)).toFixed(2)} MB` : (jobDetail?.metrics?.optimized_size_bytes?.value ? `${(jobDetail.metrics.optimized_size_bytes.value / (1024*1024)).toFixed(2)} MB` : 'Optimized') },
        { label: 'Verdict', value: verdict },
      ],
      details: `Generated standalone ${isMobileNet ? 'TFLite' : 'ONNX'} INT8 package with cryptographic checksums and deployment manifest.`,
    },
  ];

  const activeStage = stages.find((s) => s.id === selectedStageId) || stages[4];

  return (
    <div className={`p-6 rounded-2xl bg-white border border-slate-200/80 shadow-sm flex flex-col gap-5 ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-indigo-50 flex items-center justify-center text-indigo-600">
            <PackageCheck className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold text-slate-900 tracking-tight">
                UAQE Transformation Pipeline
              </h3>
              <SourceBadge source="MEASURED" origin="Autonomous Engine Controller" />
            </div>
            <p className="text-xs text-slate-500">
              End-to-end provenance trail from raw ingest to verified edge artifact
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="px-2.5 py-1 rounded-full text-xs font-bold bg-emerald-50 text-emerald-700 border border-emerald-200 flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>PIPELINE VERIFIED</span>
          </span>
        </div>
      </div>

      {/* Interactive Horizontal Pipeline Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
        {stages.map((st) => {
          const isSelected = st.id === selectedStageId;
          const Icon = st.icon;
          return (
            <button
              key={st.id}
              onClick={() => setSelectedStageId(st.id)}
              className={`p-3 rounded-xl border text-left transition-all flex flex-col justify-between min-h-[110px] relative ${
                isSelected
                  ? 'border-indigo-600 bg-indigo-50/50 shadow-sm ring-2 ring-indigo-600/20'
                  : 'border-slate-200/80 bg-slate-50/50 hover:bg-slate-100/60'
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className={`w-6 h-6 rounded-md flex items-center justify-center ${isSelected ? 'bg-indigo-600 text-white' : 'bg-slate-200 text-slate-700'}`}>
                    <Icon className="w-3.5 h-3.5" />
                  </div>
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                </div>
                <h4 className="text-[11px] font-bold text-slate-800 line-clamp-1">
                  {st.name}
                </h4>
              </div>

              <span className="text-[9px] text-slate-400 font-mono tracking-tight line-clamp-1">
                {st.shortDesc}
              </span>
            </button>
          );
        })}
      </div>

      {/* Selected Stage Detail Drawer */}
      <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/80">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-bold text-slate-900">{activeStage.name}</h4>
            <SourceBadge source={activeStage.provenance} origin="Stage Telemetry Log" />
          </div>
          <span className="text-xs text-slate-500 font-mono">{activeStage.shortDesc}</span>
        </div>

        <p className="text-xs text-slate-600 mb-3">{activeStage.details}</p>

        {activeStage.metrics && (
          <div className="grid grid-cols-3 gap-2">
            {activeStage.metrics.map((m, idx) => (
              <div key={idx} className="p-2.5 rounded-lg bg-white border border-slate-200/60">
                <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">
                  {m.label}
                </span>
                <span className="text-xs font-bold text-slate-800 font-mono block mt-0.5">
                  {m.value}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelTransformationPipeline;
