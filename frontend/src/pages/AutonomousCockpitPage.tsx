import React, { useState, useMemo, useEffect } from 'react';
import {
  Activity,
  Play,
  CheckCircle2,
  Clock,
  Cpu,
  HardDrive,
  ShieldCheck,
  AlertTriangle,
  ChevronRight,
  Terminal,
  Zap,
  Sliders,
  FileCheck2,
  Sparkles,
  Layers,
  ArrowRight,
} from 'lucide-react';
import {
  JobDetail,
  CandidateSummary,
  OptimizationEvent,
  TelemetryResponse,
} from '../types/api';
import SourceBadge from '../components/ui/SourceBadge';
import SafetyBadge from '../components/ui/SafetyBadge';
import ExplainPopover from '../components/ui/ExplainPopover';
import CandidateLossChart from '../charts/CandidateLossChart';
import CandidateThroughputLatencyChart from '../charts/CandidateThroughputLatencyChart';
import TaskManagerGraph, { TelemetryPoint } from '../charts/TaskManagerGraph';
import CockpitErrorBoundary from '../components/ui/CockpitErrorBoundary';
import {
  safeNumber,
  formatNumber,
  safeString,
  normalizeCandidate,
} from '../utils/candidateNormalizer';

interface AutonomousCockpitPageProps {
  jobDetail: JobDetail | null;
  events: OptimizationEvent[];
  telemetry: TelemetryResponse | null;
  liveTelemetrySamples?: TelemetryPoint[];
  isStreaming: boolean;
  onViewResults: () => void;
  onStartNewOptimization?: () => void;
}

const PIPELINE_STAGES = [
  'INITIALIZING',
  'INGESTING',
  'INSPECTING',
  'CALIBRATING',
  'PROFILING',
  'SEARCHING',
  'EVALUATING',
  'SELECTING',
  'VALIDATING',
  'PACKAGING',
  'COMPLETED',
] as const;

export type PipelineStage = (typeof PIPELINE_STAGES)[number];

export function mapToPipelineStage(raw: string | undefined | null): PipelineStage | null {
  if (!raw) return null;
  const s = raw.trim().toUpperCase();

  if (PIPELINE_STAGES.includes(s as PipelineStage)) {
    return s as PipelineStage;
  }

  if (s === 'INITIALIZATION' || s === 'INITIALIZE' || s === 'INIT' || s === 'OPTIMIZATION_STARTED') {
    return 'INITIALIZING';
  }
  if (s === 'INGEST' || s === 'INGESTION' || s === 'DATASET_INGESTING' || s === 'MODEL_INGESTING') {
    return 'INGESTING';
  }
  if (s === 'INSPECT' || s === 'INSPECTION' || s === 'AUDITING' || s === 'COMPATIBILITY_CHECK') {
    return 'INSPECTING';
  }
  if (s === 'CALIBRATE' || s === 'CALIBRATION' || s === 'PREPROCESSING' || s === 'ADAPTING') {
    return 'CALIBRATING';
  }
  if (s === 'PROFILE' || s === 'FP32_BASELINE' || s === 'BASELINE_MEASUREMENT' || s === 'BENCHMARKING') {
    return 'PROFILING';
  }
  if (s === 'SEARCH' || s === 'CANDIDATE_EXPLORATION' || s === 'GENERATING_CANDIDATES') {
    return 'SEARCHING';
  }
  if (
    s === 'EVALUATE' ||
    s === 'EVALUATION' ||
    s === 'CANDIDATE_START' ||
    s === 'CANDIDATE_DONE' ||
    s === 'EVALUATING_CANDIDATE'
  ) {
    return 'EVALUATING';
  }
  if (s === 'SELECT' || s === 'SELECTION' || s === 'BEST_CANDIDATE_SELECTED') {
    return 'SELECTING';
  }
  if (s === 'VALIDATE' || s === 'VALIDATION' || s === 'VALIDATING_CONSTRAINTS') {
    return 'VALIDATING';
  }
  if (s === 'PACKAGE' || s === 'PACKAGING_ARTIFACTS' || s === 'EXPORTING') {
    return 'PACKAGING';
  }
  if (
    s === 'COMPLETE' ||
    s === 'COMPLETED' ||
    s === 'OPTIMIZATION_COMPLETED' ||
    s === 'FINAL_RESULT' ||
    s === 'DONE'
  ) {
    return 'COMPLETED';
  }

  return null;
}

function getTargetName(target: any): string {
  if (!target) return 'Raspberry Pi 5';
  if (typeof target === 'string') return target;
  if (typeof target === 'object') {
    return target.name || target.id || target.hardware_class || 'Raspberry Pi 5';
  }
  return String(target);
}

export const AutonomousCockpitPage: React.FC<AutonomousCockpitPageProps> = ({
  jobDetail,
  events = [],
  telemetry,
  liveTelemetrySamples = [],
  isStreaming,
  onViewResults,
  onStartNewOptimization,
}) => {
  const [elapsedSec, setElapsedSec] = useState<number>(0);

  // Track elapsed timer during streaming
  useEffect(() => {
    if (!isStreaming) return;
    const timer = setInterval(() => {
      setElapsedSec((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, [isStreaming]);

  // Job isolation: filter events for current job if active
  const currentJobId = jobDetail?.job_id;
  const filteredEvents = useMemo(() => {
    if (!currentJobId) return events;
    return events.filter((e) => !e.job_id || e.job_id === currentJobId);
  }, [events, currentJobId]);

  // Aggregate telemetry points from live stream, event list, or stored execution samples
  const combinedTelemetrySamples = useMemo(() => {
    if (liveTelemetrySamples && liveTelemetrySamples.length > 0) {
      return liveTelemetrySamples;
    }

    // Check if telemetry events are in the event list
    const sseTelemetryEvents = filteredEvents.filter((e) => e.type === 'telemetry');
    if (sseTelemetryEvents.length > 0) {
      return sseTelemetryEvents.map((e) => ({
        timestamp: e.timestamp || Date.now() / 1000,
        system_cpu_percent: e.cpu_system_pct,
        process_cpu_percent: e.cpu_process_pct,
        system_ram_percent: e.ram_used_pct,
        process_ram_mb: e.ram_process_mb,
        peak_process_ram_mb: e.peak_ram_process_mb,
        system_ram_used_mb: e.ram_system_used_mb,
        system_ram_available_mb: e.ram_system_available_mb,
        cpu_count_logical: e.cpu_count_logical,
        cpu_frequency_current_mhz: e.cpu_frequency_current_mhz,
        phase: e.phase,
      }));
    }

    // Check if telemetry on disk has execution_samples
    const diskTel = telemetry as any;
    if (diskTel?.execution_samples && diskTel.execution_samples.length > 0) {
      return diskTel.execution_samples;
    }

    // Fallback: merge baseline & final samples if available
    const bSamples = telemetry?.baseline?.samples || [];
    const fSamples = telemetry?.final?.samples || [];
    return [...bSamples, ...fSamples];
  }, [liveTelemetrySamples, filteredEvents, telemetry]);

  const errorEvent = filteredEvents.find(
    (e) => (e.type || '').toUpperCase() === 'ERROR' || (e.type || '').toUpperCase() === 'FAILED'
  );
  const isFailed = jobDetail?.status === 'FAILED' || !!errorEvent;
  const errorMessage =
    errorEvent?.error ||
    errorEvent?.message ||
    (jobDetail?.status === 'FAILED' ? 'Optimization job failed during execution or safety constraint check.' : null);

  // Extract evaluated candidates from live events if jobDetail is not yet populated
  const liveCandidates: CandidateSummary[] = useMemo(() => {
    const cands: CandidateSummary[] = [];
    for (const ev of filteredEvents) {
      if (ev.type === 'candidate_done' && ev.result) {
        cands.push(normalizeCandidate(ev.result, cands.length));
      }
    }
    return cands;
  }, [filteredEvents]);

  const rawCandidates: CandidateSummary[] =
    jobDetail?.candidates && jobDetail.candidates.length > 0 ? jobDetail.candidates : liveCandidates;

  const candidates: CandidateSummary[] = useMemo(() => {
    return rawCandidates.map((c, idx) => normalizeCandidate(c, idx));
  }, [rawCandidates]);

  const status = jobDetail?.status || (isFailed ? 'FAILED' : isStreaming ? 'IN_PROGRESS' : 'IDLE');
  const isComplete = (status === 'COMPLETED' || jobDetail?.verdict === 'VERIFIED') && !isFailed;

  // Determine current active pipeline stage and genuine operation detail
  const { currentStage, currentOperationDetail, failedStage } = useMemo(() => {
    if (isComplete) {
      return {
        currentStage: 'COMPLETED' as PipelineStage | 'FAILED',
        currentOperationDetail: 'Autonomous optimization complete',
        failedStage: null as PipelineStage | null,
      };
    }

    let detectedStage: PipelineStage | null = null;
    let operationDetail: string | null = null;
    let hasFailure = isFailed;
    let failureDetail: string | null = errorMessage || null;
    let failedAt: PipelineStage | null = null;

    for (let i = filteredEvents.length - 1; i >= 0; i--) {
      const ev = filteredEvents[i];
      const evType = (ev.type || '').toUpperCase();

      if (evType === 'ERROR' || evType === 'FAILED' || evType === 'BASELINE_INVALID') {
        hasFailure = true;
        failureDetail = ev.error || ev.message || failureDetail || 'Optimization halted';
        if (ev.stage && ev.stage.toUpperCase() !== 'FAILED') {
          failedAt = mapToPipelineStage(ev.stage);
        }
        continue;
      }

      const stageFromField = mapToPipelineStage(ev.stage);
      const stageFromType = evType !== 'TELEMETRY' ? mapToPipelineStage(ev.type) : null;
      const stageFromPhase =
        ev.phase && ev.phase.toUpperCase() !== 'EXECUTING' ? mapToPipelineStage(ev.phase) : null;

      const stageFound = stageFromField || stageFromType || stageFromPhase;

      if (!detectedStage && stageFound) {
        detectedStage = stageFound;
        if (hasFailure && !failedAt) {
          failedAt = stageFound;
        }
      }

      if (!operationDetail) {
        if (ev.candidate_id && (stageFound === 'EVALUATING' || stageFound === 'SELECTING')) {
          operationDetail = `Candidate ${ev.candidate_id}`;
        } else if (ev.message && !ev.message.toLowerCase().includes('telemetry')) {
          operationDetail = ev.message;
        } else if ((ev as any).detail) {
          operationDetail = (ev as any).detail;
        }
      }

      if (detectedStage && operationDetail) {
        break;
      }
    }

    if (hasFailure) {
      return {
        currentStage: 'FAILED' as const,
        currentOperationDetail: failureDetail || 'Job execution halted',
        failedStage: failedAt || detectedStage || 'INITIALIZING',
      };
    }

    if (!detectedStage) {
      if (candidates.length > 0) detectedStage = 'EVALUATING';
      else if (isStreaming) detectedStage = 'INITIALIZING';
      else detectedStage = 'INITIALIZING';
    }

    return {
      currentStage: detectedStage,
      currentOperationDetail: operationDetail,
      failedStage: null as PipelineStage | null,
    };
  }, [filteredEvents, isComplete, isFailed, candidates, isStreaming, errorMessage]);

  if (!jobDetail && !isStreaming && filteredEvents.length === 0) {
    return (
      <CockpitErrorBoundary jobId={currentJobId} stage="INITIALIZING">
        <div className="flex flex-col items-center justify-center p-12 rounded-3xl bg-white border border-slate-200/80 shadow-xs text-center max-w-2xl mx-auto gap-4 my-12">
          <div className="w-14 h-14 rounded-2xl bg-indigo-50 flex items-center justify-center text-indigo-600 mb-1">
            <Cpu className="w-7 h-7" />
          </div>
          <h2 className="text-xl font-bold text-slate-900">No Optimization Job Active</h2>
          <p className="text-xs text-slate-500 leading-relaxed max-w-md">
            There is no autonomous optimization job currently running or selected in this workspace. Launch a fresh optimization or inspect a completed run from Job History.
          </p>
          <div className="flex items-center gap-3 mt-3">
            <button
              onClick={onStartNewOptimization}
              className="px-6 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold shadow-xs transition-all cursor-pointer"
            >
              Start New Optimization
            </button>
          </div>
        </div>
      </CockpitErrorBoundary>
    );
  }

  const bestCandidate = candidates.find((c) => c.is_satisfied && !c.is_critical) || candidates[0];

  const sseBaselineEvent = filteredEvents.find((e) => e.type === 'fp32_baseline' || e.type === 'BASELINE');
  const fp32Acc =
    jobDetail?.metrics?.fp32_accuracy?.value != null
      ? jobDetail.metrics.fp32_accuracy.value * 100
      : (sseBaselineEvent as any)?.accuracy != null
      ? (sseBaselineEvent as any).accuracy * 100
      : null;
  const fp32Lat =
    jobDetail?.metrics?.fp32_latency_ms?.value != null
      ? jobDetail.metrics.fp32_latency_ms.value
      : (sseBaselineEvent as any)?.latency_ms != null
      ? (sseBaselineEvent as any).latency_ms
      : null;
  const fp32SizeMB =
    jobDetail?.metrics?.original_size_bytes?.value != null
      ? jobDetail.metrics.original_size_bytes.value / (1024 * 1024)
      : (sseBaselineEvent as any)?.size_bytes != null
      ? (sseBaselineEvent as any).size_bytes / (1024 * 1024)
      : jobDetail?.model_inspection?.file_size_bytes?.value
      ? jobDetail.model_inspection.file_size_bytes.value / (1024 * 1024)
      : null;

  const optAcc = bestCandidate && safeNumber(bestCandidate.top1_accuracy) != null
    ? safeNumber(bestCandidate.top1_accuracy)! * 100
    : null;
  const lossPP = bestCandidate ? safeNumber(bestCandidate.accuracy_loss_pp) : null;
  const optLat = bestCandidate ? safeNumber(bestCandidate.latency_mean_ms) : null;
  const optSizeMB = bestCandidate && safeNumber(bestCandidate.model_size_bytes) != null
    ? safeNumber(bestCandidate.model_size_bytes)! / (1024 * 1024)
    : null;
  const latencyChangePct =
    fp32Lat != null && optLat != null && fp32Lat > 0
      ? ((fp32Lat - optLat) / fp32Lat) * 100
      : (safeNumber(jobDetail?.metrics?.latency_change_percent) ?? null);
  const deltaPP =
    optAcc !== null && fp32Acc !== null
      ? optAcc - fp32Acc
      : (safeNumber(jobDetail?.metrics?.accuracy_delta_pp) ?? null);

  const baselineInvalid =
    jobDetail?.metrics?.baseline_status?.value === 'INVALID_BASELINE' ||
    filteredEvents.some((e) => e.type === 'baseline_invalid');

  return (
    <CockpitErrorBoundary jobId={currentJobId} stage={currentStage}>
      <div className="flex flex-col gap-6 max-w-7xl mx-auto font-sans">
      {/* Invalid Baseline Warning Alert */}
      {baselineInvalid && (
        <div className="p-4 rounded-2xl bg-rose-50 border-2 border-rose-300 text-rose-900 flex items-center gap-3">
          <AlertTriangle className="w-6 h-6 text-rose-600 shrink-0" />
          <div>
            <h3 className="font-bold text-sm">QUANTIZATION GOVERNANCE NOT VERIFIED (Baseline Invalid)</h3>
            <p className="text-xs text-rose-700 mt-0.5">
              FP32 reference baseline did not satisfy the minimum validity policy ({jobDetail?.metrics?.baseline_threshold_percent?.value ?? 25.0}%). Candidate search halted to prevent false optimization claims.
            </p>
          </div>
        </div>
      )}

      {/* Top Cockpit Status Bar */}
      <div className="p-6 rounded-3xl bg-slate-900 text-white shadow-xl flex flex-col md:flex-row md:items-center md:justify-between gap-6 relative overflow-hidden">
        <div className="relative z-10">
          <div className="flex flex-wrap items-center gap-2 mb-2">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                isStreaming ? 'bg-amber-400 animate-ping' : isComplete ? 'bg-emerald-400' : 'bg-slate-400'
              }`}
            />
            <span className="text-xs font-mono font-bold tracking-wider text-emerald-400 uppercase">
              {isStreaming ? 'AUTONOMOUS PIPELINE IN EXECUTION' : isComplete ? 'AUTONOMOUS OPTIMIZATION COMPLETE' : 'ENGINE READY'}
            </span>
            <span className="text-xs text-slate-400 font-mono">Job ID: {jobDetail?.job_id || 'STREAMING_JOB'}</span>
            <span className="text-xs text-slate-400 font-mono">
              Target: {getTargetName(jobDetail?.optimization_plan?.target_hardware?.value)}
            </span>
            {isStreaming && (
              <span className="px-2 py-0.5 rounded-md bg-white/10 text-xs font-mono font-bold text-amber-300">
                Elapsed: {elapsedSec}s
              </span>
            )}
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight">
            Autonomous Optimization Cockpit
          </h1>
          <p className="text-xs sm:text-sm text-slate-300 mt-1">
            Single-input multi-candidate exploration with real-time safety gating and host CPU/RAM telemetry
          </p>
        </div>

        <div className="relative z-10 flex items-center gap-3">
          {isComplete && (
            <button
              onClick={onViewResults}
              className="px-6 py-3.5 rounded-2xl bg-emerald-500 hover:bg-emerald-400 text-slate-950 font-extrabold text-xs shadow-lg transition-all flex items-center gap-2 cursor-pointer"
            >
              <FileCheck2 className="w-4 h-4" />
              <span>View Verified Results & Deployment</span>
            </button>
          )}
        </div>
      </div>

      {/* Failure Banner */}
      {isFailed && (
        <div className="p-5 rounded-2xl bg-rose-50 border border-rose-200 flex items-start justify-between gap-4 text-rose-900 shadow-xs animate-fade-in">
          <div className="flex items-start gap-3">
            <div className="p-2 bg-rose-100 rounded-xl text-rose-600 mt-0.5">
              <AlertTriangle className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-rose-950">Optimization Job Halted or Failed</h3>
              <p className="text-xs text-rose-700 mt-1 font-mono leading-relaxed max-w-3xl">
                {errorMessage || 'A critical error or baseline invalidation occurred.'}
              </p>
            </div>
          </div>
          {onStartNewOptimization && (
            <button
              onClick={onStartNewOptimization}
              className="px-4 py-2 bg-rose-600 hover:bg-rose-700 text-white text-xs font-bold rounded-xl shadow-xs transition cursor-pointer flex-shrink-0"
            >
              Start New Run
            </button>
          )}
        </div>
      )}

      {/* 11-Stage Engine Pipeline Visualizer */}
      <div className="p-5 rounded-2xl bg-white border border-slate-200/80 shadow-xs flex flex-col gap-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-100">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-indigo-600" />
            <span className="text-xs font-bold text-slate-900 uppercase tracking-wider">
              Autonomous Optimization Pipeline Execution
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-mono text-slate-500">
              Active Stage:{' '}
              <strong
                className={`${
                  isFailed ? 'text-rose-600' : isComplete ? 'text-emerald-600' : 'text-indigo-600'
                } font-bold`}
              >
                {currentStage}
              </strong>
              {currentOperationDetail && !isComplete && (
                <span
                  className={`ml-2 px-2 py-0.5 rounded text-[10px] font-mono font-medium ${
                    isFailed ? 'bg-rose-50 text-rose-700' : 'bg-indigo-50 text-indigo-700'
                  }`}
                >
                  {currentOperationDetail}
                </span>
              )}
            </span>
          </div>
        </div>

        <div className="overflow-x-auto pb-2">
          <div className="flex items-center min-w-[800px] gap-1.5">
            {PIPELINE_STAGES.map((stage, idx) => {
              const stageIdx =
                currentStage === 'FAILED'
                  ? failedStage
                    ? PIPELINE_STAGES.indexOf(failedStage)
                    : 0
                  : PIPELINE_STAGES.indexOf(currentStage as PipelineStage);

              const isStageFailed = isFailed && (idx === stageIdx || (stageIdx === -1 && idx === 0));
              const isPast =
                isComplete ||
                (!isFailed && stageIdx !== -1 && idx < stageIdx) ||
                (isFailed && stageIdx !== -1 && idx < stageIdx);
              const isCurrent = !isComplete && !isFailed && stageIdx === idx;

              const statusText = isStageFailed
                ? 'FAILED'
                : isPast
                ? 'COMPLETED'
                : isCurrent
                ? 'ACTIVE'
                : 'PENDING';

              return (
                <React.Fragment key={stage}>
                  <div
                    className={`flex-1 p-2 rounded-xl border flex flex-col gap-1 transition-all ${
                      isStageFailed
                        ? 'border-rose-400 bg-rose-50/60 shadow-xs ring-1 ring-rose-300'
                        : isCurrent
                        ? 'border-indigo-400 bg-indigo-50/50 shadow-xs ring-1 ring-indigo-300'
                        : isPast
                        ? 'border-emerald-200 bg-emerald-50/30'
                        : 'border-slate-200 bg-slate-50/40 text-slate-400'
                    }`}
                  >
                    <div className="flex items-center justify-between text-[10px]">
                      <span className="font-mono font-bold">{(idx + 1).toString().padStart(2, '0')}</span>
                      {isStageFailed ? (
                        <AlertTriangle className="w-3 h-3 text-rose-600" />
                      ) : isPast ? (
                        <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                      ) : isCurrent ? (
                        <span className="w-2 h-2 rounded-full bg-indigo-600 animate-ping" />
                      ) : (
                        <span className="w-1.5 h-1.5 rounded-full bg-slate-300" />
                      )}
                    </div>
                    <span
                      className={`text-[10px] font-bold tracking-tight uppercase truncate ${
                        isStageFailed
                          ? 'text-rose-950'
                          : isCurrent
                          ? 'text-indigo-900'
                          : isPast
                          ? 'text-emerald-950'
                          : 'text-slate-400'
                      }`}
                    >
                      {stage}
                    </span>
                    <div
                      className={`text-[9px] font-mono font-bold flex items-center gap-1 ${
                        isStageFailed
                          ? 'text-rose-600'
                          : isCurrent
                          ? 'text-indigo-600'
                          : isPast
                          ? 'text-emerald-600'
                          : 'text-slate-400'
                      }`}
                    >
                      <span>{isStageFailed ? '✕' : isPast ? '✓' : isCurrent ? '●' : '○'}</span>
                      <span>{statusText}</span>
                    </div>
                    {isCurrent && currentOperationDetail && (
                      <span
                        className="text-[8px] font-mono text-indigo-700 truncate block mt-0.5"
                        title={currentOperationDetail}
                      >
                        {currentOperationDetail}
                      </span>
                    )}
                  </div>
                  {idx < PIPELINE_STAGES.length - 1 && (
                    <div
                      className={`w-3 h-0.5 flex-shrink-0 ${
                        isPast
                          ? 'bg-emerald-400'
                          : isCurrent
                          ? 'bg-indigo-300'
                          : isStageFailed
                          ? 'bg-rose-300'
                          : 'bg-slate-200'
                      }`}
                    />
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </div>
      </div>

      {/* Real-Time Task Manager Resource Monitoring Graphs (CPU & RAM) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <TaskManagerGraph
          metricType="cpu"
          samples={combinedTelemetrySamples}
          isLive={isStreaming}
          isReplay={!isStreaming && !isComplete && combinedTelemetrySamples.length > 0}
        />
        <TaskManagerGraph
          metricType="ram"
          samples={combinedTelemetrySamples}
          isLive={isStreaming}
          isReplay={!isStreaming && !isComplete && combinedTelemetrySamples.length > 0}
        />
      </div>

      {/* FP32 Baseline vs Current Best Candidate */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* FP32 Baseline */}
        <div className="p-6 rounded-2xl bg-white border border-slate-200/80 shadow-xs flex flex-col justify-between">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-slate-400" />
              <h3 className="text-sm font-bold text-slate-800">Reference FP32 Baseline</h3>
            </div>
            <SourceBadge
              source={fp32Acc !== null ? 'MEASURED' : isStreaming ? 'MEASURING' : 'PENDING'}
              origin="Host CPU Benchmark"
            />
          </div>

          <div className="grid grid-cols-2 gap-4 my-4">
            <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
              <span className="text-[10px] font-semibold text-slate-400 uppercase">Top-1 Accuracy</span>
              <span className="text-xl font-bold text-slate-900 font-mono block mt-1">
                {fp32Acc !== null ? `${fp32Acc.toFixed(2)}%` : isStreaming ? 'MEASURING' : 'PENDING'}
              </span>
              <span className="text-[10px] text-slate-400">
                {jobDetail?.optimization_plan?.test_samples?.value || 1000} Test Samples
              </span>
            </div>

            <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
              <span className="text-[10px] font-semibold text-slate-400 uppercase">Host Latency</span>
              <span className="text-xl font-bold text-slate-900 font-mono block mt-1">
                {fp32Lat !== null ? `${fp32Lat.toFixed(2)} ms` : isStreaming ? 'MEASURING' : 'PENDING'}
              </span>
              <span className="text-[10px] text-slate-400">
                {fp32Lat ? `${(1000.0 / fp32Lat).toFixed(2)} img/s` : isStreaming ? 'Benchmarking host' : 'Host benchmark pending'}
              </span>
            </div>

            <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
              <span className="text-[10px] font-semibold text-slate-400 uppercase">Model Weight Size</span>
              <span className="text-xl font-bold text-slate-900 font-mono block mt-1">
                {fp32SizeMB !== null ? `${fp32SizeMB.toFixed(2)} MB` : isStreaming ? 'MEASURING' : 'PENDING'}
              </span>
              <span className="text-[10px] text-slate-400">Uncompressed FP32</span>
            </div>

            <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
              <span className="text-[10px] font-semibold text-slate-400 uppercase">Baseline Status</span>
              <span className="text-sm font-bold text-slate-800 font-mono block mt-1">
                {baselineInvalid ? 'INVALID_BASELINE' : fp32Acc !== null ? 'VALID' : isStreaming ? 'MEASURING' : 'PENDING'}
              </span>
              <span className="text-[10px] text-slate-400">Threshold: 25.0%</span>
            </div>
          </div>

          <span className="text-[11px] text-slate-400 font-mono">
            Model: {jobDetail?.model_inspection?.architecture?.value || 'Ingesting & Inspecting Architecture...'}
          </span>
        </div>

        {/* Selected / Current Best Candidate */}
        <div className="p-6 rounded-2xl bg-indigo-50/40 border border-indigo-200/80 shadow-xs flex flex-col justify-between">
          <div className="flex items-center justify-between pb-3 border-b border-indigo-100">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-600" />
              <h3 className="text-sm font-bold text-indigo-950">
                Selected Pareto Candidate ({bestCandidate?.candidate_name || (isStreaming ? 'Searching...' : 'Pending')})
              </h3>
            </div>
            {lossPP !== null ? (
              <SafetyBadge
                classification={bestCandidate?.safety_classification || 'EXCELLENT'}
                lossPP={lossPP}
              />
            ) : (
              <span className="text-[10px] font-mono font-bold text-slate-400">
                {isStreaming ? 'MEASURING' : 'PENDING'}
              </span>
            )}
          </div>

          <div className="grid grid-cols-2 gap-4 my-4">
            <div className="p-3 rounded-xl bg-white border border-indigo-100">
              <span className="text-[10px] font-semibold text-indigo-500 uppercase">Top-1 Accuracy</span>
              <span className="text-xl font-bold text-emerald-600 font-mono block mt-1">
                {optAcc !== null ? `${optAcc.toFixed(2)}%` : isStreaming ? 'MEASURING' : 'PENDING'}
              </span>
              <span className="text-[10px] text-emerald-600 font-semibold font-mono">
                {deltaPP !== null
                  ? deltaPP >= 0
                    ? `+${deltaPP.toFixed(2)} pp IMPROVEMENT`
                    : `${deltaPP.toFixed(2)} pp (${Math.abs(deltaPP).toFixed(2)} pp loss)`
                  : isStreaming ? 'Evaluating safety' : 'Pending'}
              </span>
            </div>

            <div className="p-3 rounded-xl bg-white border border-indigo-100">
              <span className="text-[10px] font-semibold text-indigo-500 uppercase">Host Latency</span>
              <span className="text-xl font-bold text-indigo-900 font-mono block mt-1">
                {optLat !== null ? `${optLat.toFixed(2)} ms` : isStreaming ? 'MEASURING' : 'PENDING'}
              </span>
              <span className="text-[10px] text-emerald-600 font-semibold font-mono">
                {latencyChangePct !== null
                  ? latencyChangePct >= 0
                    ? `Speedup: +${latencyChangePct.toFixed(1)}%`
                    : `${latencyChangePct.toFixed(1)}% slower`
                  : isStreaming ? 'Calculating speedup' : 'Pending'}
              </span>
            </div>

            <div className="p-3 rounded-xl bg-white border border-indigo-100">
              <span className="text-[10px] font-semibold text-indigo-500 uppercase">Model Size</span>
              <span className="text-xl font-bold text-indigo-900 font-mono block mt-1">
                {optSizeMB !== null ? `${optSizeMB.toFixed(2)} MB` : isStreaming ? 'MEASURING' : 'PENDING'}
              </span>
              <span className="text-[10px] text-indigo-600 font-semibold font-mono">
                {bestCandidate?.size_reduction_percent != null
                  ? `Reduction: -${formatNumber(bestCandidate.size_reduction_percent, 1)}%`
                  : 'Evaluating compression'}
              </span>
            </div>

            <div className="p-3 rounded-xl bg-white border border-indigo-100">
              <span className="text-[10px] font-semibold text-indigo-500 uppercase">Composite Score</span>
              <span className="text-xl font-bold text-indigo-950 font-mono block mt-1">
                {bestCandidate && safeNumber(bestCandidate.composite_score) != null
                  ? formatNumber(bestCandidate.composite_score, 3)
                  : '---'}
              </span>
              <span className="text-[10px] text-emerald-700 font-semibold">
                {bestCandidate?.is_satisfied ? 'Objective Satisfied' : 'Evaluating trade-offs'}
              </span>
            </div>
          </div>

          <span className="text-[11px] text-indigo-700 font-mono">
            Strategy: {bestCandidate?.strategy_type || (jobDetail?.metrics?.selected_strategy?.value || 'Adaptive INT8 Quantization')}
          </span>
        </div>
      </div>

      {/* Candidate Generation Timeline & Rejection Explanations */}
      <div className="p-6 rounded-2xl bg-white border border-slate-200/80 shadow-xs flex flex-col gap-4">
        <div className="flex items-center justify-between pb-3 border-b border-slate-100">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                Autonomous Candidate Evaluation Timeline
              </h3>
              <SourceBadge source="MEASURED" origin="Search Controller Log" />
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              Real candidate execution results and UAQE safety policy rejection decisions
            </p>
          </div>
          <span className="text-xs font-mono text-slate-400">{candidates.length} candidates evaluated</span>
        </div>

        <div className="grid grid-cols-1 gap-3">
          {candidates.length === 0 ? (
            <div className="p-8 text-center text-slate-400 font-mono text-xs italic">
              {isStreaming ? 'Generating and calibrating initial quantization candidates...' : 'No candidate records.'}
            </div>
          ) : (
            candidates.map((cand, candIdx) => {
              const isWinner = cand.is_satisfied && !cand.is_critical;
              const candId = safeString(cand.candidate_id, `cand_${candIdx + 1}`);
              const lossVal = safeNumber(cand.accuracy_loss_pp);
              const accVal = safeNumber(cand.top1_accuracy);
              const sizeMB = safeNumber(cand.model_size_bytes) != null ? cand.model_size_bytes / (1024 * 1024) : null;
              const sizeRedPct = safeNumber(cand.size_reduction_percent);
              const latVal = safeNumber(cand.latency_mean_ms);
              const tputVal = safeNumber(cand.throughput_ips);
              const scoreVal = safeNumber(cand.composite_score);

              return (
                <div
                  key={candId}
                  className={`p-4 rounded-xl border transition-all flex flex-col md:flex-row md:items-center md:justify-between gap-4 ${
                    isWinner ? 'border-emerald-300 bg-emerald-50/30' : 'border-slate-200 bg-slate-50/50'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <div
                      className={`w-9 h-9 rounded-lg flex items-center justify-center font-bold text-xs font-mono ${
                        isWinner ? 'bg-emerald-600 text-white shadow-xs' : 'bg-rose-100 text-rose-800'
                      }`}
                    >
                      #{candId.replace('CAND-', '').replace('cand_', '')}
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <h4 className="text-xs font-bold text-slate-900">{safeString(cand.candidate_name, `Candidate ${candIdx + 1}`)}</h4>
                        <SafetyBadge
                          classification={cand.safety_classification}
                          lossPP={lossVal}
                        />
                        {isWinner && (
                          <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-100 text-emerald-800 border border-emerald-200">
                            WINNER
                          </span>
                        )}
                      </div>
                      <p className="text-[11px] text-slate-500 mt-0.5 font-mono">
                        Strategy: {safeString(cand.strategy_type, String(jobDetail?.metrics?.selected_strategy?.value || 'Adaptive INT8'))}
                      </p>
                    </div>
                  </div>

                  <div className="flex flex-wrap items-center gap-3">
                    {/* Accuracy */}
                    <div className="text-right font-mono text-xs">
                      <span className="text-slate-400 block text-[10px]">Top-1 Acc</span>
                      <strong className="text-slate-800">
                        {accVal != null ? `${formatNumber(accVal * 100, 2)}%` : 'NOT AVAILABLE'}
                      </strong>
                      <span
                        className={`block text-[10px] font-bold ${
                          lossVal != null && lossVal > 4.0
                            ? 'text-rose-600'
                            : lossVal != null && lossVal > 1.0
                            ? 'text-amber-600'
                            : 'text-emerald-600'
                        }`}
                      >
                        {lossVal != null
                          ? lossVal <= 0
                            ? `+${Math.abs(lossVal).toFixed(2)} pp`
                            : `-${lossVal.toFixed(2)} pp`
                          : 'PENDING'}
                      </span>
                    </div>

                    {/* Size */}
                    <div className="text-right font-mono text-xs">
                      <span className="text-slate-400 block text-[10px]">Size</span>
                      <strong className="text-slate-800">
                        {sizeMB != null ? `${formatNumber(sizeMB, 1)} MB` : 'NOT AVAILABLE'}
                      </strong>
                      <span className="block text-[10px] text-indigo-600 font-bold">
                        {sizeRedPct != null ? `-${formatNumber(sizeRedPct, 1)}%` : 'NOT AVAILABLE'}
                      </span>
                    </div>

                    {/* Latency */}
                    <div className="text-right font-mono text-xs">
                      <span className="text-slate-400 block text-[10px]">Host Latency</span>
                      <strong className="text-slate-800">
                        {latVal != null ? `${formatNumber(latVal, 1)} ms` : 'NOT AVAILABLE'}
                      </strong>
                      <span className="block text-[10px] text-emerald-600 font-bold">
                        {tputVal != null ? `${formatNumber(tputVal, 1)} img/s` : 'NOT AVAILABLE'}
                      </span>
                    </div>

                    {/* Score */}
                    <div className="text-right font-mono text-xs pl-2 border-l border-slate-200">
                      <span className="text-slate-400 block text-[10px]">Score</span>
                      <strong className="text-indigo-600 text-sm">
                        {scoreVal != null ? formatNumber(scoreVal, 3) : '---'}
                      </strong>
                    </div>

                    {/* Explain Popover */}
                    <ExplainPopover candidate={cand} />
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Visual Graphs: Candidate Loss Trajectory & Speedup */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CandidateLossChart candidates={candidates} />
        <CandidateThroughputLatencyChart candidates={candidates} />
      </div>

      {/* Real-time SSE Execution Console */}
      <div className="p-6 rounded-2xl bg-slate-950 text-slate-200 shadow-xl border border-slate-800 flex flex-col gap-3 font-mono">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800">
          <div className="flex items-center gap-2 text-xs">
            <Terminal className="w-4 h-4 text-emerald-400" />
            <span className="font-bold text-slate-300">UAQE Engine Event Stream (SSE)</span>
          </div>
          <span className="text-[10px] text-slate-500">Live Socket Active</span>
        </div>

        <div className="h-44 overflow-y-auto flex flex-col gap-1.5 text-xs">
          {events.length === 0 ? (
            <div className="text-slate-600 italic">Listening for autonomous engine dispatch events...</div>
          ) : (
            events.map((ev, idx) => (
              <div key={idx} className="flex items-start gap-2 leading-relaxed">
                <span className="text-slate-500 flex-shrink-0">
                  [{new Date(ev.timestamp ? ev.timestamp * 1000 : Date.now()).toLocaleTimeString()}]
                </span>
                <span className="text-indigo-400 flex-shrink-0 font-bold">[{ev.type}]</span>
                <span className="text-slate-300">{ev.message || JSON.stringify(ev.result || {})}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  </CockpitErrorBoundary>
);
};

export default AutonomousCockpitPage;
