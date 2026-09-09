import React from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts';
import { CandidateSummary } from '../types/api';
import SourceBadge from '../components/ui/SourceBadge';
import { ShieldAlert, AlertCircle } from 'lucide-react';
import { safeNumber, safeString } from '../utils/candidateNormalizer';

interface CandidateLossChartProps {
  candidates: CandidateSummary[];
  maxAllowedPp?: number;
  className?: string;
}

export const CandidateLossChart: React.FC<CandidateLossChartProps> = ({
  candidates,
  maxAllowedPp = 4.0,
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
          No candidate loss evaluations available yet.
        </p>
      </div>
    );
  }

  const data = candidates.map((c, idx) => {
    const lossNum = safeNumber(c.accuracy_loss_pp, 0) ?? 0;
    const accNum = (safeNumber(c.top1_accuracy, 0) ?? 0) * 100;
    return {
      name: `Cand ${idx + 1}`,
      fullName: safeString(c.candidate_name, `Candidate ${idx + 1}`),
      loss: Number(lossNum.toFixed(2)),
      accuracy: accNum.toFixed(2),
      classification: safeString(c.safety_classification, 'UNKNOWN'),
      status: c.is_satisfied ? 'VALID' : 'CRITICAL',
      fill: c.is_satisfied && !c.is_critical ? '#10b981' : '#f43f5e',
    };
  });

  const validLosses = candidates.map((c) => safeNumber(c.accuracy_loss_pp, 0) ?? 0);
  const safePolicyGate = safeNumber(maxAllowedPp, 4.0) ?? 4.0;
  const maxLoss = Math.max(...(validLosses.length > 0 ? validLosses : [0]), safePolicyGate, 6.0);

  return (
    <div className={`p-6 rounded-2xl bg-white border border-slate-200/80 shadow-sm flex flex-col gap-4 ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-rose-50 flex items-center justify-center text-rose-600">
            <ShieldAlert className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-slate-900 tracking-tight">
              Candidate Accuracy Loss & Safety Gating
            </h3>
            <SourceBadge source="MEASURED" origin="Accuracy Validator" />
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <span className="text-slate-400">Policy Gate:</span>
          <span className="font-mono font-bold text-rose-600">&le; {maxAllowedPp.toFixed(2)} pp</span>
        </div>
      </div>

      <p className="text-xs text-slate-500">
        Loss tolerance boundary: &le; {maxAllowedPp.toFixed(2)} pp max allowed drop
      </p>

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
            <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} tickLine={false} />
            <YAxis domain={[0, Math.ceil(maxLoss) + 1]} stroke="#94a3b8" fontSize={11} tickLine={false} unit=" pp" />
            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                borderRadius: '12px',
                boxShadow: '0 4px 20px -2px rgba(0,0,0,0.1)',
                border: '1px solid #e2e8f0',
                fontSize: '12px',
              }}
              formatter={(val: any, name: any, item: any) => [
                `${val} pp (${item.payload.classification})`,
                item.payload.fullName,
              ]}
            />

            {/* Safety Threshold Reference Lines */}
            <ReferenceLine
              y={1.0}
              stroke="#10b981"
              strokeDasharray="4 4"
              label={{ value: 'EXCELLENT (≤ 1.0 pp)', fill: '#059669', fontSize: 10, position: 'insideBottomRight' }}
            />
            <ReferenceLine
              y={maxAllowedPp}
              stroke="#f43f5e"
              strokeWidth={1.5}
              label={{ value: `SAFETY GATE (≤ ${maxAllowedPp.toFixed(1)} pp)`, fill: '#e11d48', fontSize: 10, position: 'insideTopRight' }}
            />

            <Bar dataKey="loss" radius={[6, 6, 0, 0]} maxBarSize={48} fill="#6366f1" />
            <Line type="monotone" dataKey="loss" stroke="#0f172a" strokeWidth={2} dot={{ r: 4, fill: '#0f172a' }} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default CandidateLossChart;
