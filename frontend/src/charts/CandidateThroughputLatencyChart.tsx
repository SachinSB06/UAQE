import React from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceLine,
} from 'recharts';
import { CandidateSummary } from '../types/api';
import SourceBadge from '../components/ui/SourceBadge';
import { FastForward, AlertCircle } from 'lucide-react';
import { safeNumber, safeString, formatNumber } from '../utils/candidateNormalizer';

interface CandidateThroughputLatencyChartProps {
  candidates: CandidateSummary[];
  baselineLatency?: number;
  baselineThroughput?: number;
  className?: string;
}

export const CandidateThroughputLatencyChart: React.FC<CandidateThroughputLatencyChartProps> = ({
  candidates,
  baselineLatency = 82.46,
  baselineThroughput = 12.13,
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
          No optimization candidates have been evaluated yet.
        </p>
      </div>
    );
  }

  const safeBaseLat = safeNumber(baselineLatency, 82.46) ?? 82.46;
  const safeBaseTput = safeNumber(baselineThroughput, 12.13) ?? 12.13;

  const chartData = [
    {
      name: 'FP32 Baseline',
      latency: safeBaseLat,
      throughput: safeBaseTput,
      accuracy: 75.0,
      loss: 0.0,
      isWinner: false,
      fill: '#94a3b8',
    },
    ...candidates.map((c, idx) => {
      const lat = safeNumber(c.latency_mean_ms, 0) ?? 0;
      const tput = safeNumber(c.throughput_ips, 0) ?? 0;
      const acc = (safeNumber(c.top1_accuracy, 0) ?? 0) * 100;
      const loss = safeNumber(c.accuracy_loss_pp, 0) ?? 0;
      const isWinner = Boolean(c.is_satisfied && !c.is_critical);

      return {
        name: safeString(c.candidate_name, `Candidate ${idx + 1}`),
        latency: Number(lat.toFixed(2)),
        throughput: Number(tput.toFixed(2)),
        accuracy: Number(acc.toFixed(2)),
        loss: Number(loss.toFixed(2)),
        isWinner,
        fill: isWinner ? '#10b981' : '#f43f5e',
      };
    }),
  ];

  return (
    <div className={`p-6 rounded-2xl bg-white border border-slate-200/80 shadow-sm flex flex-col gap-4 ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-indigo-50 flex items-center justify-center text-indigo-600">
            <FastForward className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold text-slate-900 tracking-tight">
                Candidate Throughput & Latency Trajectory
              </h3>
              <SourceBadge source="MEASURED" origin="Host CPU Benchmark" />
            </div>
            <p className="text-xs text-slate-500">
              Direct comparison of candidate inference speedup vs FP32 baseline
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <span className="text-slate-400">Baseline Latency:</span>
          <span className="font-mono font-bold text-slate-700">{formatNumber(safeBaseLat, 2)} ms</span>
        </div>
      </div>

      {/* Main Dual Axis Chart */}
      <div className="h-[280px] w-full pt-2">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="name" stroke="#64748b" fontSize={11} tickLine={false} interval={0} />
            <YAxis
              yAxisId="left"
              stroke="#6366f1"
              fontSize={11}
              tickLine={false}
              unit=" ms"
              label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft', offset: 10, fontSize: 10, fill: '#6366f1' }}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              stroke="#10b981"
              fontSize={11}
              tickLine={false}
              unit=" ips"
              label={{ value: 'Throughput (img/s)', angle: 90, position: 'insideRight', offset: 10, fontSize: 10, fill: '#10b981' }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                borderRadius: '12px',
                boxShadow: '0 4px 20px -2px rgba(0,0,0,0.1)',
                border: '1px solid #e2e8f0',
                fontSize: '12px',
              }}
              formatter={(val: any, name: any, item: any) => {
                if (name === 'Latency (ms)') return [`${val} ms (Loss: ${item.payload.loss} pp)`, name];
                return [`${val} img/sec`, name];
              }}
            />
            <Legend verticalAlign="top" height={36} iconType="circle" />
            <ReferenceLine yAxisId="left" y={safeBaseLat} stroke="#cbd5e1" strokeDasharray="3 3" label={{ value: 'FP32 Baseline Latency', position: 'insideTopRight', fill: '#94a3b8', fontSize: 10 }} />
            <Bar yAxisId="left" dataKey="latency" name="Latency (ms)" barSize={36} radius={[6, 6, 0, 0]} fill="#6366f1" />
            <Line yAxisId="right" type="monotone" dataKey="throughput" name="Throughput (img/s)" stroke="#10b981" strokeWidth={3} dot={{ r: 5, fill: '#10b981' }} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default CandidateThroughputLatencyChart;
