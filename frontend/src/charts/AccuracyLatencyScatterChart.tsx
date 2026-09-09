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
import { Clock, AlertCircle } from 'lucide-react';

interface AccuracyLatencyScatterChartProps {
  candidates: CandidateSummary[];
  baselineAccuracy?: number;
  baselineLatency?: number;
  selectedCandidateId?: string | null;
  className?: string;
}

export const AccuracyLatencyScatterChart: React.FC<AccuracyLatencyScatterChartProps> = ({
  candidates,
  baselineAccuracy = 75.0,
  baselineLatency = 82.46,
  selectedCandidateId,
  className = '',
}) => {
  const points = [
    {
      id: 'fp32_baseline',
      name: 'FP32 Baseline',
      latencyMs: parseFloat(baselineLatency.toFixed(2)),
      accuracy: parseFloat(baselineAccuracy.toFixed(2)),
      isFp32: true,
      isSelected: false,
      isCritical: false,
    },
    ...candidates.map((c) => ({
      id: c.candidate_id,
      name: c.candidate_name,
      latencyMs: parseFloat(c.latency_mean_ms.toFixed(2)),
      accuracy: parseFloat((c.top1_accuracy * 100).toFixed(2)),
      isFp32: false,
      isSelected: c.is_satisfied && !c.is_critical,
      isCritical: c.is_critical,
    })),
  ];

  if (candidates.length === 0) {
    return (
      <div className={`p-6 rounded-2xl bg-white border border-slate-200/80 shadow-sm flex flex-col items-center justify-center min-h-[300px] text-center ${className}`}>
        <AlertCircle className="w-8 h-8 text-amber-500 mb-2" />
        <p className="text-xs font-mono font-bold tracking-wider text-slate-500 uppercase">
          NOT AVAILABLE FOR THIS MODEL/RUNTIME
        </p>
      </div>
    );
  }

  return (
    <div className={`p-6 rounded-2xl bg-white border border-slate-200/80 shadow-sm flex flex-col gap-4 ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-emerald-50 flex items-center justify-center text-emerald-600">
            <Clock className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold text-slate-900 tracking-tight">
                Top-1 Accuracy vs Host CPU Latency (ms)
              </h3>
              <SourceBadge source="MEASURED" origin="Host Validator" />
            </div>
            <p className="text-xs text-slate-500">
              Scatter plot of inference latency against accuracy retention
            </p>
          </div>
        </div>
      </div>

      <div className="h-[280px] w-full pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 30, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis
              type="number"
              dataKey="latencyMs"
              name="Inference Latency"
              unit=" ms"
              stroke="#94a3b8"
              fontSize={11}
              tickLine={false}
              label={{ value: 'Host CPU Latency (ms/img)', position: 'insideBottom', offset: -10, fontSize: 10, fill: '#94a3b8' }}
            />
            <YAxis
              type="number"
              dataKey="accuracy"
              name="Accuracy"
              unit="%"
              domain={['auto', 'auto']}
              stroke="#94a3b8"
              fontSize={11}
              tickLine={false}
              label={{ value: 'Top-1 Accuracy (%)', angle: -90, position: 'insideLeft', offset: 15, fontSize: 10, fill: '#94a3b8' }}
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
                          Accuracy: <span className="font-semibold text-slate-900">{d.accuracy}%</span>
                        </p>
                        <p className="text-slate-600">
                          Latency: <span className="font-semibold text-slate-900">{d.latencyMs} ms</span>
                        </p>
                        {d.isSelected && <p className="text-emerald-600 font-bold mt-1">★ FINAL SELECTED</p>}
                      </div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Scatter name="Candidates" data={points}>
              {points.map((entry, index) => {
                let fill = '#4F46E5';
                if (entry.isFp32) fill = '#64748B';
                if (entry.isCritical) fill = '#E11D48';
                if (entry.isSelected) fill = '#059669';
                return <Cell key={`cell-${index}`} fill={fill} r={entry.isSelected ? 8 : 6} stroke="#FFFFFF" strokeWidth={2} />;
              })}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default AccuracyLatencyScatterChart;
