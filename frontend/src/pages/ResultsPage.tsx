import React, { useState, useMemo, useEffect } from 'react';
import {
  FileCheck,
  CheckCircle2,
  AlertTriangle,
  RotateCcw,
  Download,
  Clock,
  Cpu,
  HardDrive,
  Zap,
  Gauge,
  Sparkles,
  ArrowRight,
  ChevronDown,
  ChevronUp,
  Sliders,
  Layers,
  ShieldCheck,
  Package,
} from 'lucide-react';
import {
  JobDetail,
  CandidateSummary,
  TelemetryResponse,
} from '../types/api';
import { normalizeCandidate } from '../utils/candidateNormalizer';
import { getActiveCandidateMetrics } from '../utils/candidateMetrics';
import SourceBadge from '../components/ui/SourceBadge';
import SafetyBadge from '../components/ui/SafetyBadge';
import ParetoFrontierChart from '../charts/ParetoFrontierChart';
import CandidateLossChart from '../charts/CandidateLossChart';
import AccuracySizeScatterChart from '../charts/AccuracySizeScatterChart';
import AccuracyLatencyScatterChart from '../charts/AccuracyLatencyScatterChart';
import CandidateThroughputLatencyChart from '../charts/CandidateThroughputLatencyChart';
import QuantizationCoverageChart from '../charts/QuantizationCoverageChart';
import LayerPrecisionVisualizer from '../charts/LayerPrecisionVisualizer';
import ModelTransformationPipeline from '../charts/ModelTransformationPipeline';
import ResourceComparisonCard from '../charts/ResourceComparisonCard';
import TaskManagerGraph from '../charts/TaskManagerGraph';

interface ResultsPageProps {
  jobDetail: JobDetail | null;
  telemetry: TelemetryResponse | null;
  onDownloadArtifact: (target: string, candidateId?: string) => void;
  onStartNewOptimization: () => void;
}

function getTargetName(target: any): string {
  if (!target) return 'Raspberry Pi 5';
  if (typeof target === 'string') return target;
  if (typeof target === 'object') {
    return target.name || target.id || target.hardware_class || 'Raspberry Pi 5';
  }
  return String(target);
}

export const ResultsPage: React.FC<ResultsPageProps> = ({
  jobDetail,
  telemetry,
  onDownloadArtifact,
  onStartNewOptimization,
}) => {
  const [activeChartTab, setActiveChartTab] = useState<
    'overview' | 'pareto' | 'telemetry' | 'precision' | 'deployment'
  >('overview');
  const [showTechnicalJson, setShowTechnicalJson] = useState<boolean>(false);

  // Normalized candidates list
  const candidates: CandidateSummary[] = useMemo(() => {
    return (jobDetail?.candidates || []).map(normalizeCandidate);
  }, [jobDetail?.candidates]);

  // Determine default winner candidate
  const winningCandidate = useMemo(() => {
    if (candidates.length === 0) return null;
    return candidates.find((c) => c.is_satisfied && !c.is_critical) || candidates[0] || null;
  }, [candidates]);

  // Selected candidate ID with URL query param and sessionStorage persistence
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(() => {
    try {
      const sp = new URLSearchParams(window.location.search);
      const fromUrl = sp.get('candidate_id');
      if (fromUrl) return fromUrl;
      if (jobDetail?.job_id) {
        const fromStorage = sessionStorage.getItem(`uaqe_selected_cand_${jobDetail.job_id}`);
        if (fromStorage) return fromStorage;
      }
    } catch {
      // ignore
    }
    return null;
  });

  // Sync state if jobDetail changes and storage has a preference
  useEffect(() => {
    if (!jobDetail?.job_id) return;
    try {
      const sp = new URLSearchParams(window.location.search);
      const fromUrl = sp.get('candidate_id');
      if (fromUrl) {
        setSelectedCandidateId(fromUrl);
        return;
      }
      const fromStorage = sessionStorage.getItem(`uaqe_selected_cand_${jobDetail.job_id}`);
      if (fromStorage) {
        setSelectedCandidateId(fromStorage);
        return;
      }
    } catch {
      // ignore
    }
  }, [jobDetail?.job_id]);

  // Active candidate: match selectedCandidateId or fall back to winning candidate
  const activeCandidate = useMemo(() => {
    if (selectedCandidateId) {
      const found = candidates.find((c) => c.candidate_id === selectedCandidateId);
      if (found) return found;
    }
    return winningCandidate;
  }, [candidates, selectedCandidateId, winningCandidate]);

  // Handle explicit candidate selection
  const handleSelectCandidate = (candId: string) => {
    setSelectedCandidateId(candId);
    try {
      if (jobDetail?.job_id) {
        sessionStorage.setItem(`uaqe_selected_cand_${jobDetail.job_id}`, candId);
      }
      const url = new URL(window.location.href);
      url.searchParams.set('candidate_id', candId);
      window.history.replaceState(null, '', url.toString());
    } catch (e) {
      console.warn('Could not persist candidate selection:', e);
    }
  };

  // Handle reset to winner candidate
  const handleResetToWinner = () => {
    const winnerId = winningCandidate?.candidate_id || null;
    setSelectedCandidateId(winnerId);
    try {
      if (jobDetail?.job_id) {
        sessionStorage.removeItem(`uaqe_selected_cand_${jobDetail.job_id}`);
      }
      const url = new URL(window.location.href);
      url.searchParams.delete('candidate_id');
      window.history.replaceState(null, '', url.toString());
    } catch (e) {
      console.warn('Could not reset candidate selection:', e);
    }
  };

  // Single canonical metric object for all candidate-dependent components
  const metrics = useMemo(() => {
    return getActiveCandidateMetrics(jobDetail, activeCandidate, telemetry);
  }, [jobDetail, activeCandidate, telemetry]);

  // Strict candidate-specific telemetry phase (never stale winner fallback)
  const activeCandidateTelemetry = useMemo(() => {
    if (activeCandidate?.candidate_id && telemetry?.phases) {
      return telemetry.phases[activeCandidate.candidate_id] || null;
    }
    return null;
  }, [activeCandidate?.candidate_id, telemetry?.phases]);

  if (!jobDetail) {
    return (
      <div className="flex flex-col items-center justify-center p-12 rounded-3xl bg-white border border-slate-200/80 shadow-xs text-center max-w-2xl mx-auto gap-4 my-12 font-sans">
        <div className="w-14 h-14 rounded-2xl bg-indigo-50 flex items-center justify-center text-indigo-600 mb-1">
          <FileCheck className="w-7 h-7" />
        </div>
        <h2 className="text-xl font-bold text-slate-900">No Optimization Results Loaded</h2>
        <p className="text-xs text-slate-500 leading-relaxed max-w-md">
          There are no optimization metrics currently active in this workspace. Launch a new optimization job or select a past job from Job History to view verification manifests.
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
    );
  }

  // DEV-mode consistency assertion
  if (import.meta.env.DEV && activeCandidate) {
    const activeCandId = activeCandidate.candidate_id;
    if (!activeCandId) {
      console.error('[UAQE] Candidate identity missing: activeCandidate has no candidate_id', {
        job_id: jobDetail.job_id,
        activeCandidate,
      });
    }
  }

  const isInvalid = jobDetail.metrics?.baseline_status?.value === 'INVALID_BASELINE' || jobDetail.verdict === 'FAILED';

  // Fallback telemetry samples for replay if candidate execution samples are empty
  const diskTel = telemetry as any;
  const replayTelemetrySamples = (metrics.executionSamples && metrics.executionSamples.length > 0)
    ? metrics.executionSamples
    : diskTel?.execution_samples || [
        ...(telemetry?.baseline?.samples || []),
        ...(telemetry?.final?.samples || []),
      ];

  return (
    <div className="flex flex-col gap-6 max-w-7xl mx-auto font-sans">
      {/* Invalid Baseline Warning Alert */}
      {isInvalid && (
        <div className="p-4 rounded-2xl bg-rose-50 border-2 border-rose-300 text-rose-900 flex items-center gap-3">
          <AlertTriangle className="w-6 h-6 text-rose-600 shrink-0" />
          <div>
            <h3 className="font-bold text-sm">QUANTIZATION GOVERNANCE NOT VERIFIED (Baseline Invalid)</h3>
            <p className="text-xs text-rose-700 mt-0.5">
              The FP32 reference baseline did not satisfy the minimum validity threshold ({jobDetail.metrics?.baseline_threshold_percent?.value ?? 25.0}%). Candidate search halted to prevent false optimization claims.
            </p>
          </div>
        </div>
      )}

      {/* Hero Header */}
      <div className="p-8 rounded-3xl bg-slate-900 text-white shadow-xl flex flex-col md:flex-row md:items-center md:justify-between gap-6 relative overflow-hidden">
        <div className="relative z-10">
          <div className="flex items-center gap-2 mb-2">
            <span
              className={`px-3 py-1 rounded-full text-xs font-bold font-mono ${
                isInvalid
                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                  : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
              } flex items-center gap-1.5`}
            >
              {isInvalid ? <AlertTriangle className="w-4 h-4 text-rose-400" /> : <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
              <span>{jobDetail.verdict || 'VERIFIED ARTIFACT'}</span>
            </span>
            <SourceBadge
              source={isInvalid ? 'POLICY' : 'MEASURED'}
              origin={isInvalid ? 'Baseline Guard' : 'Host CPU Benchmark'}
            />
          </div>

          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight">
            Autonomous Optimization Results & Verification Manifest
          </h1>
          <p className="text-xs sm:text-sm text-slate-300 mt-1">
            Job {jobDetail.job_id} • {String(jobDetail.model_inspection?.architecture?.value || 'ResNet-50')} + {String(jobDetail.dataset_inspection?.dataset_name?.value || 'CIFAR-10')} • Target: {getTargetName(jobDetail.optimization_plan?.target_hardware?.value)}
          </p>
        </div>

        {/* Replay Candidate Scrubber */}
        {candidates.length > 1 && (
          <div className="relative z-10 flex flex-col gap-2 p-3.5 rounded-2xl bg-white/10 backdrop-blur-md border border-white/15">
            <div className="flex items-center justify-between text-xs text-slate-300 font-mono">
              <span className="flex items-center gap-1">
                <RotateCcw className="w-3.5 h-3.5 text-indigo-400" />
                <span>Candidate Scrubber:</span>
              </span>
              <button
                onClick={handleResetToWinner}
                className="text-[10px] text-indigo-300 hover:text-white underline cursor-pointer"
              >
                Reset to Winner
              </button>
            </div>
            <div className="flex items-center gap-1.5 flex-wrap">
              {candidates.map((c, idx) => {
                const isSelected = c.candidate_id === metrics.candidateId;
                return (
                  <button
                    key={c.candidate_id || idx}
                    onClick={() => handleSelectCandidate(c.candidate_id)}
                    className={`px-3 py-1.5 rounded-xl text-xs font-mono font-bold transition-all cursor-pointer ${
                      isSelected
                        ? 'bg-indigo-600 text-white shadow-sm ring-2 ring-indigo-400/50'
                        : 'bg-white/10 text-slate-300 hover:bg-white/20'
                    }`}
                  >
                    Cand #{idx + 1}
                    {c.candidate_id ? ` (${c.candidate_id})` : ''}
                    {c.is_satisfied && !c.is_critical ? ' ★' : ''}
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Hero 3-Card KPI Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Accuracy Card */}
        <div className="p-6 rounded-3xl bg-white border border-slate-200/80 shadow-xs flex flex-col justify-between">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Top-1 Accuracy</span>
            <SafetyBadge
              classification={metrics.safetyClassification as any}
              lossPP={metrics.accuracyLossPp}
            />
          </div>

          <div className="my-4 flex items-baseline justify-between">
            <div>
              <span className="text-xs text-slate-400 font-mono block">FP32 Baseline</span>
              <span className="text-lg font-bold text-slate-500 font-mono">
                {metrics.fp32Accuracy !== null ? `${metrics.fp32Accuracy.toFixed(2)}%` : 'PENDING'}
              </span>
            </div>
            <ArrowRight className="w-5 h-5 text-slate-300" />
            <div className="text-right">
              <span className={`text-xs font-mono block font-semibold ${isInvalid ? 'text-rose-500' : 'text-emerald-600'}`}>
                {isInvalid ? 'Search Halted' : 'Optimized INT8'}
              </span>
              <span className="text-2xl font-extrabold text-slate-900 font-mono">
                {isInvalid ? 'NOT RUN' : (metrics.optAccuracy !== null ? `${metrics.optAccuracy.toFixed(2)}%` : 'PENDING')}
              </span>
            </div>
          </div>

          <div className={`p-2.5 rounded-xl border flex items-center justify-between text-xs ${
            isInvalid ? 'bg-rose-50 border-rose-100' : 'bg-emerald-50 border-emerald-100'
          }`}>
            <span className={isInvalid ? 'text-rose-800 font-semibold' : 'text-emerald-800 font-semibold'}>
              {isInvalid ? 'Governance Status:' : 'Accuracy Delta:'}
            </span>
            <strong className={`font-mono ${isInvalid ? 'text-rose-900' : 'text-emerald-900'}`}>
              {isInvalid
                ? 'Candidate search halted (invalid baseline)'
                : metrics.accuracyDeltaPp !== null
                ? metrics.accuracyDeltaPp >= 0
                  ? `+${metrics.accuracyDeltaPp.toFixed(2)} pp delta (improvement)`
                  : `${metrics.accuracyDeltaPp.toFixed(2)} pp delta (${metrics.accuracyLossPp?.toFixed(2)} pp loss)`
                : 'PENDING'}
            </strong>
          </div>
        </div>

        {/* Model Size Card */}
        <div className="p-6 rounded-3xl bg-white border border-slate-200/80 shadow-xs flex flex-col justify-between">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Model Size & Footprint</span>
            <SourceBadge source={isInvalid ? 'POLICY' : 'MEASURED'} origin={isInvalid ? 'Baseline Guard' : 'Disk Size'} />
          </div>

          <div className="my-4 flex items-baseline justify-between">
            <div>
              <span className="text-xs text-slate-400 font-mono block">Original FP32</span>
              <span className="text-lg font-bold text-slate-500 font-mono">
                {metrics.fp32SizeMB !== null ? `${metrics.fp32SizeMB.toFixed(2)} MB` : 'PENDING'}
              </span>
            </div>
            <ArrowRight className="w-5 h-5 text-slate-300" />
            <div className="text-right">
              <span className={`text-xs font-mono block font-semibold ${isInvalid ? 'text-slate-400' : 'text-indigo-600'}`}>
                {isInvalid ? 'INT8 Package' : 'INT8 Package'}
              </span>
              <span className="text-2xl font-extrabold text-slate-900 font-mono">
                {isInvalid ? 'NONE' : (metrics.optSizeMB !== null ? `${metrics.optSizeMB.toFixed(2)} MB` : 'PENDING')}
              </span>
            </div>
          </div>

          <div className={`p-2.5 rounded-xl border flex items-center justify-between text-xs ${
            isInvalid ? 'bg-slate-50 border-slate-200' : 'bg-indigo-50 border-indigo-100'
          }`}>
            <span className={isInvalid ? 'text-slate-600 font-semibold' : 'text-indigo-800 font-semibold'}>File Compression:</span>
            <strong className={`font-mono ${isInvalid ? 'text-slate-700' : 'text-indigo-900'}`}>
              {isInvalid ? 'No compression (Halted)' : (metrics.sizeReductionPercent !== null ? `-${metrics.sizeReductionPercent.toFixed(1)}% reduction` : 'PENDING')}
            </strong>
          </div>
        </div>

        {/* Latency & Speedup Card */}
        <div className="p-6 rounded-3xl bg-white border border-slate-200/80 shadow-xs flex flex-col justify-between">
          <div className="flex items-center justify-between pb-3 border-b border-slate-100">
            <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">Pure Host Inference Latency</span>
            <SourceBadge source={isInvalid ? 'POLICY' : 'MEASURED'} origin={isInvalid ? 'Baseline Guard' : 'Pure Invoke Benchmark'} />
          </div>

          <div className="my-4 flex items-baseline justify-between">
            <div>
              <span className="text-xs text-slate-400 font-mono block">Pure FP32 Latency</span>
              <span className="text-lg font-bold text-slate-500 font-mono">
                {metrics.fp32LatencyMs !== null ? `${metrics.fp32LatencyMs.toFixed(2)} ms` : 'PENDING'}
              </span>
              {metrics.fp32ThroughputIps != null && (
                <span className="text-[10px] text-slate-400 font-mono block">
                  {metrics.fp32ThroughputIps.toFixed(1)} img/s
                </span>
              )}
            </div>
            <ArrowRight className="w-5 h-5 text-slate-300" />
            <div className="text-right">
              <span className={`text-xs font-mono block font-semibold ${isInvalid ? 'text-slate-400' : 'text-emerald-600'}`}>
                {isInvalid ? 'INT8 Latency' : 'Pure INT8 Latency'}
              </span>
              <span className="text-2xl font-extrabold text-slate-900 font-mono">
                {isInvalid ? 'NONE' : (metrics.optLatencyMs !== null ? `${metrics.optLatencyMs.toFixed(2)} ms` : 'PENDING')}
              </span>
              {!isInvalid && metrics.optThroughputIps != null && (
                <span className="text-[10px] text-emerald-600 font-mono block font-semibold">
                  Pure Throughput: {metrics.optThroughputIps.toFixed(1)} img/s
                </span>
              )}
            </div>
          </div>

          <div className={`p-2.5 rounded-xl border flex items-center justify-between text-xs ${
            isInvalid ? 'bg-slate-50 border-slate-200' : 'bg-emerald-50 border-emerald-100'
          }`}>
            <span className={isInvalid ? 'text-slate-600 font-semibold' : 'text-emerald-800 font-semibold'}>Execution Speedup:</span>
            <strong className={`font-mono ${isInvalid ? 'text-slate-700' : 'text-emerald-900'}`}>
              {isInvalid
                ? 'Candidate search halted'
                : metrics.speedupPercent !== null
                ? metrics.speedupPercent >= 0
                  ? `+${metrics.speedupPercent.toFixed(1)}% speedup`
                  : `${metrics.speedupPercent.toFixed(1)}% slower`
                : 'PENDING'}
            </strong>
          </div>

          {!isInvalid && (
            <div className="mt-2 pt-2 border-t border-slate-100 flex flex-col gap-1 text-[10px] text-slate-500 font-mono">
              <div className="flex items-center justify-between">
                <span>Protocol: <strong>10 warmups • 100 invokes • {metrics.threads || 2} threads</strong></span>
                <span className="px-1.5 py-0.5 rounded bg-slate-100 font-semibold text-slate-700 uppercase">
                  {metrics.runtime ? 'CANONICAL' : 'MEASURED'}
                </span>
              </div>
              {metrics.runtime && (
                <span className="text-slate-400 truncate">
                  Runtime: {String(metrics.runtime)}
                  {metrics.delegate && (
                    <span className="ml-1 text-slate-500 font-semibold">
                      ({String(metrics.delegate)})
                    </span>
                  )}
                  {metrics.delegatedOps != null && (
                    <span className="ml-1.5 text-slate-400">
                      • {metrics.delegatedOps} delegated
                      {metrics.fallbackOps != null ? ` / ${metrics.fallbackOps} fallback` : ''}
                    </span>
                  )}
                </span>
              )}
              <div className="text-[9px] text-slate-400/90 italic mt-0.5 leading-tight">
                * FP32 and INT8 use different inference runtimes; latency reflects the complete runtime stack.
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Tabs Navigation */}
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between overflow-x-auto pb-1">
          <div className="flex bg-slate-100 p-1 rounded-2xl text-xs font-semibold">
            <button
              onClick={() => setActiveChartTab('overview')}
              className={`px-4 py-2 rounded-xl transition-all ${
                activeChartTab === 'overview'
                  ? 'bg-white text-slate-900 shadow-xs font-bold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              1. Resource Efficiency
            </button>
            <button
              onClick={() => setActiveChartTab('pareto')}
              className={`px-4 py-2 rounded-xl transition-all ${
                activeChartTab === 'pareto'
                  ? 'bg-white text-slate-900 shadow-xs font-bold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              2. Pareto & Exploration
            </button>
            <button
              onClick={() => setActiveChartTab('telemetry')}
              className={`px-4 py-2 rounded-xl transition-all ${
                activeChartTab === 'telemetry'
                  ? 'bg-white text-slate-900 shadow-xs font-bold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              3. Telemetry Replay
            </button>
            <button
              onClick={() => setActiveChartTab('precision')}
              className={`px-4 py-2 rounded-xl transition-all ${
                activeChartTab === 'precision'
                  ? 'bg-white text-slate-900 shadow-xs font-bold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              4. Layer Precision
            </button>
            <button
              onClick={() => setActiveChartTab('deployment')}
              className={`px-4 py-2 rounded-xl transition-all ${
                activeChartTab === 'deployment'
                  ? 'bg-white text-slate-900 shadow-xs font-bold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              5. Deployment & Package
            </button>
          </div>
        </div>

        {/* Tab 1: Overview & Resource Efficiency */}
        {activeChartTab === 'overview' && (
          <div className="flex flex-col gap-6">
            <ResourceComparisonCard
              baseline={telemetry?.baseline}
              optimized={activeCandidateTelemetry}
              fp32LatencyMs={metrics.fp32LatencyMs}
              optimizedLatencyMs={metrics.optLatencyMs}
              throughputIps={metrics.optThroughputIps}
              cpuAvgPct={metrics.cpuAvgPct}
              cpuPeakPct={metrics.cpuPeakPct}
              ramAvgMb={metrics.ramAvgMb}
              ramPeakMb={metrics.ramPeakMb}
              candidateName={metrics.candidateName}
            />
            <ModelTransformationPipeline
              jobDetail={jobDetail}
              status="COMPLETED"
              verdict={jobDetail.verdict}
              activeCandidate={activeCandidate}
            />
          </div>
        )}

        {/* Tab 2: Pareto & Candidate Exploration */}
        {activeChartTab === 'pareto' && (
          <div className="flex flex-col gap-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <ParetoFrontierChart
                candidates={candidates}
                selectedCandidateId={metrics.candidateId}
                baselineAccuracy={metrics.fp32Accuracy ?? 0}
                baselineSizeMB={metrics.fp32SizeMB ?? 0}
                baselineLatency={metrics.fp32LatencyMs ?? 0}
              />
              <CandidateLossChart candidates={candidates} />
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <AccuracySizeScatterChart
                candidates={candidates}
                baselineAccuracy={metrics.fp32Accuracy ?? 0}
                baselineSizeMB={metrics.fp32SizeMB ?? 0}
              />
              <AccuracyLatencyScatterChart
                candidates={candidates}
                baselineAccuracy={metrics.fp32Accuracy ?? 0}
                baselineLatency={metrics.fp32LatencyMs ?? 0}
              />
            </div>
          </div>
        )}

        {/* Tab 3: System Telemetry Replay */}
        {activeChartTab === 'telemetry' && (
          <div className="flex flex-col gap-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <TaskManagerGraph
                metricType="cpu"
                samples={replayTelemetrySamples}
                isReplay={true}
              />
              <TaskManagerGraph
                metricType="ram"
                samples={replayTelemetrySamples}
                isReplay={true}
              />
            </div>
          </div>
        )}

        {/* Tab 4: Layer Precision & Coverage */}
        {activeChartTab === 'precision' && (
          <div className="flex flex-col gap-6">
            <LayerPrecisionVisualizer
              architectureName={jobDetail.model_inspection?.architecture?.value || 'ResNet-50'}
              jobDetail={jobDetail}
              baselineStatus={jobDetail.metrics?.baseline_status?.value}
            />
            <QuantizationCoverageChart />
          </div>
        )}

        {/* Tab 5: Deployment & Packaging */}
        {activeChartTab === 'deployment' && (
          <div className="flex flex-col gap-6">
            {/* Target Hardware Validation Separation */}
            <div className="p-6 rounded-3xl bg-white border border-slate-200/80 shadow-xs flex flex-col gap-4">
              <div className="flex items-center justify-between pb-3 border-b border-slate-100">
                <div className="flex items-center gap-2">
                  <Cpu className="w-5 h-5 text-indigo-600" />
                  <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                    Target Hardware Deployment & Separation
                  </h3>
                </div>
                <SourceBadge source="POLICY" origin="Hardware Governance" />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 rounded-2xl bg-emerald-50/40 border border-emerald-200">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold text-emerald-950">Host Workstation CPU</span>
                    <SourceBadge source="MEASURED" origin="psutil & Host ORT" />
                  </div>
                  <p className="text-xs text-slate-600 leading-relaxed mb-3">
                    Directly measured on development host ({telemetry?.environment_info?.os || 'Windows AMD64'}). Measured latency of{' '}
                    <strong className="text-emerald-800 font-mono">
                      {metrics.optLatencyMs ? `${metrics.optLatencyMs.toFixed(2)} ms/image` : 'PENDING'}
                    </strong>
                    .
                  </p>
                  <div className="flex items-center gap-2 text-xs font-mono text-emerald-800 font-bold">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                    <span>HOST VALIDATION MEASURED</span>
                  </div>
                </div>

                <div className="p-4 rounded-2xl bg-amber-50/40 border border-amber-200">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold text-amber-950">
                      {getTargetName(jobDetail.optimization_plan?.target_hardware?.value)} Physical Target
                    </span>
                    <SourceBadge source="PENDING" origin="Remote Board Profiler" />
                  </div>
                  <p className="text-xs text-slate-600 leading-relaxed mb-3">
                    Model compiled with optimized edge operators ({metrics.strategyName || 'MobileNet INT8'}). Physical on-device execution requires connected hardware bench.
                  </p>
                  <div className="flex items-center gap-2 text-xs font-mono text-amber-800 font-bold">
                    <Clock className="w-4 h-4 text-amber-600" />
                    <span>PHYSICAL VALIDATION PENDING</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Artifact Packaging Hub */}
            <div className="p-6 rounded-3xl bg-white border border-slate-200/80 shadow-xs flex flex-col gap-4">
              <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                <div className="flex items-center gap-2">
                  <Package className="w-5 h-5 text-indigo-600" />
                  <div>
                    <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                      Artifact Packaging Hub
                    </h3>
                    <p className="text-xs text-slate-500">
                      Universal self-contained `.uaqe` package, standalone ONNX/TFLite models, and manifests
                    </p>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                {/* model.uaqe */}
                <button
                  onClick={() => onDownloadArtifact('model.uaqe')}
                  className="p-4 rounded-2xl border border-slate-200 bg-slate-50/50 hover:bg-indigo-50/40 hover:border-indigo-300 transition-all text-left flex flex-col justify-between group cursor-pointer"
                >
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="px-2 py-0.5 rounded-md text-[9px] font-bold font-mono bg-indigo-600 text-white">
                        UAQE PACKAGE
                      </span>
                      <Download className="w-4 h-4 text-slate-400 group-hover:text-indigo-600" />
                    </div>
                    <h4 className="text-xs font-bold text-slate-900">model.uaqe</h4>
                    <p className="text-[11px] text-slate-500 mt-1">Universal Embedded AI Archive</p>
                  </div>
                  <span className="text-[10px] text-indigo-700 font-mono mt-3 font-semibold">Self-Contained Bundle</span>
                </button>

                {/* Candidate Artifact (Dynamic: ONNX or TFLite based on active candidate) */}
                <button
                  onClick={() => {
                    if (metrics.artifact.downloadUrl) {
                      onDownloadArtifact(metrics.artifact.downloadUrl, metrics.candidateId || undefined);
                    } else {
                      onDownloadArtifact(metrics.artifact.filename, metrics.candidateId || undefined);
                    }
                  }}
                  className="p-4 rounded-2xl border border-slate-200 bg-slate-50/50 hover:bg-indigo-50/40 hover:border-indigo-300 transition-all text-left flex flex-col justify-between group cursor-pointer"
                >
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="px-2 py-0.5 rounded-md text-[9px] font-bold font-mono bg-emerald-100 text-emerald-800">
                        {metrics.artifact.formatLabel || (metrics.artifact.format === 'tflite' ? 'TFLITE INT8' : 'ONNX INT8')}
                      </span>
                      <Download className="w-4 h-4 text-slate-400 group-hover:text-indigo-600" />
                    </div>
                    <h4 className="text-xs font-bold text-slate-900 truncate" title={metrics.artifact.filename}>
                      {metrics.artifact.filename}
                    </h4>
                    <p className="text-[11px] text-slate-500 mt-1">
                      {metrics.candidateId ? `Candidate ${metrics.candidateId} Artifact` : 'Quantized Inference Graph'}
                    </p>
                  </div>
                  <div className="mt-3 flex flex-col gap-0.5">
                    <span className="text-[10px] text-slate-400 font-mono">
                      {metrics.artifact.sizeMB != null
                        ? `${metrics.artifact.sizeMB.toFixed(2)} MB`
                        : metrics.optSizeMB != null
                        ? `${metrics.optSizeMB.toFixed(2)} MB`
                        : 'Inference Ready'}
                    </span>
                    {metrics.artifact.sha256 && (
                      <span className="text-[9px] text-slate-400 font-mono truncate" title={metrics.artifact.sha256}>
                        SHA: {metrics.artifact.sha256.substring(0, 8)}...
                      </span>
                    )}
                  </div>
                </button>

                {/* report.md */}
                <button
                  onClick={() => onDownloadArtifact('report.md')}
                  className="p-4 rounded-2xl border border-slate-200 bg-slate-50/50 hover:bg-indigo-50/40 hover:border-indigo-300 transition-all text-left flex flex-col justify-between group cursor-pointer"
                >
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="px-2 py-0.5 rounded-md text-[9px] font-bold font-mono bg-sky-100 text-sky-800">
                        MARKDOWN
                      </span>
                      <Download className="w-4 h-4 text-slate-400 group-hover:text-indigo-600" />
                    </div>
                    <h4 className="text-xs font-bold text-slate-900">report.md</h4>
                    <p className="text-[11px] text-slate-500 mt-1">Executive Verification Report</p>
                  </div>
                  <span className="text-[10px] text-slate-400 font-mono mt-3">Comprehensive Audit Trail</span>
                </button>

                {/* telemetry.json */}
                <button
                  onClick={() => onDownloadArtifact('telemetry.json')}
                  className="p-4 rounded-2xl border border-slate-200 bg-slate-50/50 hover:bg-indigo-50/40 hover:border-indigo-300 transition-all text-left flex flex-col justify-between group cursor-pointer"
                >
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="px-2 py-0.5 rounded-md text-[9px] font-bold font-mono bg-amber-100 text-amber-800">
                        TELEMETRY
                      </span>
                      <Download className="w-4 h-4 text-slate-400 group-hover:text-indigo-600" />
                    </div>
                    <h4 className="text-xs font-bold text-slate-900">telemetry.json</h4>
                    <p className="text-[11px] text-slate-500 mt-1">Host CPU & RAM Time Series</p>
                  </div>
                  <span className="text-[10px] text-slate-400 font-mono mt-3">psutil Sampling Logs</span>
                </button>
              </div>
            </div>

            {/* Complete Cryptographic Provenance Details */}
            <div className="p-6 rounded-3xl bg-white border border-slate-200/80 shadow-xs flex flex-col gap-3 font-mono text-xs">
              <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider font-sans">
                Cryptographic Artifact Provenance
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-slate-600">
                <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                  <span className="text-slate-400 block text-[10px]">ACTIVE CANDIDATE ARTIFACT SHA-256</span>
                  <span className="font-bold text-slate-800 break-all">
                    {metrics.artifact.sha256 || 'PENDING_HASH'}
                  </span>
                </div>
                <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                  <span className="text-slate-400 block text-[10px]">ORIGINAL MODEL SHA-256</span>
                  <span className="font-bold text-slate-800 break-all">
                    {jobDetail.model_inspection?.sha256?.value || 'VERIFIED_SAMPLE'}
                  </span>
                </div>
                <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                  <span className="text-slate-400 block text-[10px]">JOB IDENTIFIER & ACTIVE CANDIDATE</span>
                  <span className="font-bold text-slate-800">
                    {jobDetail.job_id} {metrics.candidateId ? `• [${metrics.candidateId}]` : ''}
                  </span>
                </div>
                <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                  <span className="text-slate-400 block text-[10px]">WEIGHT SOURCE & ADAPTATION</span>
                  <span className="font-bold text-slate-800">
                    {jobDetail.metrics?.weight_source?.value || 'ORIGINAL_MODEL'} / {jobDetail.metrics?.adaptation_status?.value || 'NOT_REQUIRED'}
                  </span>
                </div>
                <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                  <span className="text-slate-400 block text-[10px]">BASELINE STATUS & POLICY</span>
                  <span className="font-bold text-slate-800">
                    {jobDetail.metrics?.baseline_status?.value || 'VALID'} (Policy v{jobDetail.metrics?.baseline_validity_policy_version?.value || '1.0'})
                  </span>
                </div>
                <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                  <span className="text-slate-400 block text-[10px]">OPTIMIZED RUNTIME & DELEGATE</span>
                  <span className="font-bold text-slate-800">
                    {metrics.runtime || 'Default'} {metrics.delegate ? `(${metrics.delegate})` : ''}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Raw Technical JSON Debug Expander */}
      <div className="rounded-2xl bg-white border border-slate-200/80 shadow-xs overflow-hidden">
        <button
          onClick={() => setShowTechnicalJson(!showTechnicalJson)}
          className="w-full p-4 flex items-center justify-between bg-slate-50 hover:bg-slate-100 transition-all text-xs font-semibold text-slate-700 cursor-pointer"
        >
          <span>Raw Technical Metrics & Provenance JSON</span>
          {showTechnicalJson ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        {showTechnicalJson && (
          <div className="p-4 bg-slate-950 text-slate-200 font-mono text-xs overflow-x-auto max-h-96">
            <pre>{JSON.stringify(jobDetail, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultsPage;
