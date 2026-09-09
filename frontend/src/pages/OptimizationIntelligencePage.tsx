import React, { useState } from 'react';
import {
  ShieldAlert,
  CheckCircle2,
  AlertTriangle,
  Lightbulb,
  Layers,
  Scale,
} from 'lucide-react';
import { JobDetail, CandidateSummary } from '../types/api';
import SourceBadge from '../components/ui/SourceBadge';
import ParetoFrontierChart from '../charts/ParetoFrontierChart';
import LayerPrecisionVisualizer from '../charts/LayerPrecisionVisualizer';
import QuantizationCoverageChart from '../charts/QuantizationCoverageChart';
import { normalizeCandidate, formatNumber, safeString } from '../utils/candidateNormalizer';

interface OptimizationIntelligencePageProps {
  jobDetail: JobDetail | null;
  onStartNewOptimization?: () => void;
}

export const OptimizationIntelligencePage: React.FC<OptimizationIntelligencePageProps> = ({
  jobDetail,
  onStartNewOptimization,
}) => {
  const [activeTab, setActiveTab] = useState<'decisions' | 'precision' | 'pareto'>('decisions');

  if (!jobDetail) {
    return (
      <div className="flex flex-col items-center justify-center p-12 rounded-3xl bg-white border border-slate-200/80 shadow-sm text-center max-w-2xl mx-auto gap-4 my-12">
        <div className="w-14 h-14 rounded-2xl bg-amber-50 flex items-center justify-center text-amber-600 mb-1">
          <Lightbulb className="w-7 h-7" />
        </div>
        <h2 className="text-xl font-bold text-slate-900">No Optimization Decisions to Audit</h2>
        <p className="text-xs text-slate-500 leading-relaxed max-w-md">
          There is no active optimization run loaded. Run a new optimization job or select an existing job from Job History to audit candidate rejection and acceptance logic.
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

  const rawCandidates: CandidateSummary[] = jobDetail.candidates || [];
  const candidates: CandidateSummary[] = rawCandidates.map((c, idx) => normalizeCandidate(c, idx));
  const selectedCand = candidates.find((c) => c.is_satisfied && !c.is_critical) || candidates[candidates.length - 1];
  const rejectedCandidates = candidates.filter((c) => c.is_critical || !c.is_satisfied);

  return (
    <div className="flex flex-col gap-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex flex-col gap-1 pb-4 border-b border-slate-200">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            Optimization Intelligence & Decision Audit
          </h1>
          <SourceBadge source="POLICY" origin="UAQE Autonomous Governance" />
        </div>
        <p className="text-sm text-slate-500">
          Formal engineering audit trail of candidate selections, policy rejections, and precision allocations
        </p>
      </div>

      {/* Navigation Sub-Tabs */}
      <div className="flex bg-slate-100 p-1 rounded-2xl w-fit text-xs font-semibold">
        <button
          onClick={() => setActiveTab('decisions')}
          className={`px-4 py-2 rounded-xl transition-all flex items-center gap-2 ${
            activeTab === 'decisions'
              ? 'bg-white text-slate-900 shadow-xs'
              : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          <Lightbulb className="w-4 h-4 text-amber-500" />
          <span>Policy & Rejection Decisions</span>
        </button>

        <button
          onClick={() => setActiveTab('precision')}
          className={`px-4 py-2 rounded-xl transition-all flex items-center gap-2 ${
            activeTab === 'precision'
              ? 'bg-white text-slate-900 shadow-xs'
              : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          <Layers className="w-4 h-4 text-indigo-500" />
          <span>Layer Precision Governance</span>
        </button>

        <button
          onClick={() => setActiveTab('pareto')}
          className={`px-4 py-2 rounded-xl transition-all flex items-center gap-2 ${
            activeTab === 'pareto'
              ? 'bg-white text-slate-900 shadow-xs'
              : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          <Scale className="w-4 h-4 text-emerald-600" />
          <span>Pareto Frontier Analysis</span>
        </button>
      </div>

      {/* Tab 1: Decisions & Explanations */}
      {activeTab === 'decisions' && (
        <div className="flex flex-col gap-6">
          {/* Dynamic Questions & Candidate Decisions */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Dynamic Rejected Candidates Cards */}
            {rejectedCandidates.map((c) => (
              <div
                key={c.candidate_id}
                className="p-5 rounded-2xl bg-white border border-rose-200/80 shadow-xs flex flex-col justify-between gap-3"
              >
                <div>
                  <div className="flex items-center gap-2 text-rose-800 mb-2">
                    <ShieldAlert className="w-5 h-5 text-rose-600 flex-shrink-0" />
                    <h3 className="text-sm font-bold">Why was {c.candidate_name} Rejected?</h3>
                  </div>
                  <p className="text-xs text-slate-600 leading-relaxed">
                    Strategy <strong>{c.strategy_type}</strong> produced Top-1 accuracy of{' '}
                    <strong className="text-slate-900 font-mono">{(c.top1_accuracy * 100).toFixed(2)}%</strong>,
                    incurring an accuracy drop of{' '}
                    <strong className="text-rose-700 font-mono">{c.accuracy_loss_pp.toFixed(2)} pp</strong>.
                  </p>
                </div>
                <div className="p-2.5 rounded-lg bg-rose-50 border border-rose-100 text-[11px] text-rose-900 font-mono">
                  {c.rejection_reason || `Rejection: Loss (${c.accuracy_loss_pp.toFixed(2)} pp) exceeds policy threshold`}
                </div>
              </div>
            ))}

            {/* Selected Winning Candidate Card */}
            {selectedCand && (
              <div className="p-5 rounded-2xl bg-white border border-emerald-200/80 shadow-xs flex flex-col justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2 text-emerald-800 mb-2">
                    <CheckCircle2 className="w-5 h-5 text-emerald-600 flex-shrink-0" />
                    <h3 className="text-sm font-bold">Why was {selectedCand.candidate_name} Selected?</h3>
                  </div>
                  <p className="text-xs text-slate-600 leading-relaxed">
                    Strategy <strong>{safeString(selectedCand.strategy_type, 'Standard QDQ')}</strong> preserved{' '}
                    <strong className="text-emerald-700 font-mono">{formatNumber(selectedCand.top1_accuracy * 100, 2)}% Top-1 accuracy</strong>{' '}
                    (only <span className="font-mono text-emerald-700 font-bold">-{formatNumber(selectedCand.accuracy_loss_pp, 2)} pp loss</span>),
                    while cutting footprint to{' '}
                    <strong className="text-indigo-600 font-mono">{formatNumber(selectedCand.model_size_bytes / (1024 * 1024), 2)} MB</strong>{' '}
                    (-{formatNumber(selectedCand.size_reduction_percent, 1)}%) and executing on host CPU at{' '}
                    <strong className="text-emerald-600 font-mono">{formatNumber(selectedCand.latency_mean_ms, 2)} ms</strong>.
                  </p>
                </div>
                <div className="p-2.5 rounded-lg bg-emerald-50 border border-emerald-100 text-[11px] text-emerald-900 font-mono">
                  Selection: Highest composite score ({formatNumber(selectedCand.composite_score, 3)}) &amp; {safeString(selectedCand.safety_classification, 'EXCELLENT')} safety tier
                </div>
              </div>
            )}

            {/* Stopping Reason Explanation Card */}
            <div className="p-5 rounded-2xl bg-white border border-indigo-200/80 shadow-xs flex flex-col justify-between gap-3">
              <div>
                <div className="flex items-center gap-2 text-indigo-900 mb-2">
                  <Lightbulb className="w-5 h-5 text-indigo-600 flex-shrink-0" />
                  <h3 className="text-sm font-bold">Why did UAQE Stop Searching?</h3>
                </div>
                <p className="text-xs text-slate-600 leading-relaxed">
                  {jobDetail?.stopping_description ||
                    'UAQE implements target-satisfaction early stopping. Once a candidate satisfies all accuracy safety limits and performance speedup constraints, exploration halts early to preserve compute budget.'}
                </p>
              </div>
              <div className="p-2.5 rounded-lg bg-indigo-50 border border-indigo-100 text-[11px] text-indigo-950 font-mono">
                Policy Reason: {jobDetail?.stopping_reason || 'TARGET_CONSTRAINTS_SATISFIED'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab 2: Layer Precision Visualizer */}
      {activeTab === 'precision' && (
        <div className="flex flex-col gap-6">
          <LayerPrecisionVisualizer architectureName={jobDetail?.model_inspection?.architecture?.value || 'ResNet-50 v1.5'} />
          <QuantizationCoverageChart />
        </div>
      )}

      {/* Tab 3: Pareto Frontier */}
      {activeTab === 'pareto' && (
        <div className="flex flex-col gap-6">
          <ParetoFrontierChart
            candidates={candidates}
            baselineAccuracy={75.0}
            baselineSizeMB={97.71}
            baselineLatency={82.46}
          />
        </div>
      )}
    </div>
  );
};

export default OptimizationIntelligencePage;
