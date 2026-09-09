import React from 'react';
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from 'recharts';
import { CandidateSummary } from '../types/api';
import SourceBadge from '../components/ui/SourceBadge';
import { Scale, AlertCircle } from 'lucide-react';

interface ParetoFrontierChartProps {
  candidates: CandidateSummary[];
  selectedCandidateId?: string | null;
  baselineAccuracy?: number;
  baselineSizeMB?: number;
  baselineLatency?: number;
  className?: string;
}

export const ParetoFrontierChart: React.FC<ParetoFrontierChartProps> = ({
  candidates,
  selectedCandidateId,
  baselineAccuracy = 75.0,
  baselineSizeMB = 97.71,
  baselineLatency = 82.46,
  className = '',
}) => {
  if (!candidates || candidates.length === 0) {
    return (
      <div className={`p-6 rounded-2xl bg-white border border-slate-200/80 shadow-sm flex flex-col items-center justify-center min-h-[300px] text-center ${className}`}>
        <AlertCircle className="w-8 h-8 text-amber-500 mb-2" />
        <p className="text-xs font-mono font-bold tracking-wider text-slate-500 uppercase">
          NOT AVAILABLE FOR THIS MODEL/RUNTIME
        </p>
        <p className="text-xs text-slate-400 mt-1 max-w-sm">
          No Pareto frontier data available for this execution.
        </p>
      </div>
    );
  }

  // Separate valid vs critical
  const validPoints = candidates
    .filter((c) => !c.is_critical)
    .map((c) => ({
      id: c.candidate_id,
      name: c.candidate_name,
      sizeMb: parseFloat((c.model_size_bytes / (1024 * 1024)).toFixed(2)),
      accuracy: parseFloat((c.top1_accuracy * 100).toFixed(2)),
      latencyMs: parseFloat(c.latency_mean_ms.toFixed(2)),
      score: c.composite_score,
      isSelected: selectedCandidateId ? c.candidate_id === selectedCandidateId : (c.is_satisfied && !c.is_critical),
      status: 'VALID',
    }));

  const criticalPoints = candidates
    .filter((c) => c.is_critical)
    .map((c) => ({
      id: c.candidate_id,
      name: c.candidate_name,
      sizeMb: parseFloat((c.model_size_bytes / (1024 * 1024)).toFixed(2)),
      accuracy: parseFloat((c.top1_accuracy * 100).toFixed(2)),
      latencyMs: parseFloat(c.latency_mean_ms.toFixed(2)),
      score: c.composite_score,
      isSelected: selectedCandidateId ? c.candidate_id === selectedCandidateId : false,
      status: 'CRITICAL',
    }));

  const allPoints = [...validPoints, ...criticalPoints];

  return (
    <div className={`p-6 rounded-2xl bg-white border border-slate-200/80 shadow-sm flex flex-col gap-4 ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-emerald-50 flex items-center justify-center text-emerald-600">
            <Scale className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold text-slate-900 tracking-tight">
                Pareto Frontier & Multi-Objective Tradeoff
              </h3>
              <SourceBadge source="CALCULATED" origin="UAQE Search Manager" />
            </div>
            <p className="text-xs text-slate-500">
              Optimal trade-off space: Top-1 Accuracy vs Model Footprint (MB)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
            <span className="text-slate-600 font-medium">Safe Candidate</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
            <span className="text-slate-600 font-medium">Policy Gated</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-indigo-600" />
            <span className="text-slate-600 font-medium">★ Winner</span>
          </div>
        </div>
      </div>

      <div className="h-[280px] w-full pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis
              type="number"
              dataKey="sizeMb"
              name="Model Size"
              unit=" MB"
              stroke="#94a3b8"
              fontSize={11}
              tickLine={false}
              label={{ value: 'Footprint (MB)', position: 'insideBottom', offset: -10, fontSize: 10, fill: '#94a3b8' }}
            />
            <YAxis
              type="number"
              dataKey="accuracy"
              name="Top-1 Accuracy"
              unit="%"
              domain={['auto', 'auto']}
              stroke="#94a3b8"
              fontSize={11}
              tickLine={false}
              label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft', offset: 15, fontSize: 10, fill: '#94a3b8' }}
            />
            <Tooltip
              cursor={{ strokeDasharray: '3 3' }}
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const d = payload[0].payload;
                  return (
                    <div className="p-3 bg-white rounded-xl shadow-lg border border-slate-200 text-xs font-sans">
                      <p className="font-bold text-slate-900 mb-1">{d.name}</p>
                      <div className="space-y-1 font-mono text-[11px]">
                        <p className="text-slate-600">
                          Status:{' '}
                          <span
                            className={`font-semibold ${
                              d.status === 'CRITICAL' ? 'text-rose-600' : 'text-emerald-600'
                            }`}
                          >
                            {d.status}
                          </span>
                        </p>
                        <p className="text-slate-600">
                          Accuracy: <span className="font-semibold text-slate-900">{d.accuracy}%</span>
                        </p>
                        <p className="text-slate-600">
                          Size: <span className="font-semibold text-slate-900">{d.sizeMb} MB</span>
                        </p>
                        <p className="text-slate-600">
                          Latency: <span className="font-semibold text-slate-900">{d.latencyMs} ms</span>
                        </p>
                        <p className="text-slate-600">
                          Objective Score: <span className="font-semibold text-indigo-600">{d.score.toFixed(4)}</span>
                        </p>
                        {d.isSelected && <p className="text-emerald-600 font-bold mt-1">★ FINAL SELECTED WINNER</p>}
                      </div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Scatter name="Pareto Points" data={allPoints}>
              {allPoints.map((entry, index) => {
                let fill = '#10B981';
                if (entry.status === 'CRITICAL') fill = '#F43F5E';
                if (entry.isSelected) fill = '#4F46E5';
                return (
                  <Cell
                    key={`pareto-cell-${index}`}
                    fill={fill}
                    r={entry.isSelected ? 9 : 6}
                    stroke="#FFFFFF"
                    strokeWidth={2}
                  />
                );
              })}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ParetoFrontierChart;
