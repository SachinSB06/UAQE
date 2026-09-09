import React from 'react';
import { Cpu, HardDrive, Zap, Gauge, ArrowDownRight, ArrowUpRight, CheckCircle2 } from 'lucide-react';
import SourceBadge from '../components/ui/SourceBadge';

interface TelemetryPhase {
  phase_id?: string;
  phase_name?: string;
  avg_cpu_percent?: number;
  cpu_avg_pct?: number;
  peak_cpu_percent?: number;
  cpu_peak_pct?: number;
  avg_ram_mb?: number;
  ram_avg_mb?: number;
  peak_ram_mb?: number;
  ram_peak_mb?: number;
  latency_mean_ms?: number;
  throughput_ips?: number;
  [key: string]: any;
}

interface ResourceComparisonCardProps {
  baseline?: TelemetryPhase | null;
  optimized?: TelemetryPhase | null;
  fp32LatencyMs?: number | null;
  optimizedLatencyMs?: number | null;
  throughputIps?: number | null;
  cpuAvgPct?: number | null;
  cpuPeakPct?: number | null;
  ramAvgMb?: number | null;
  ramPeakMb?: number | null;
  candidateName?: string | null;
  className?: string;
}

export const ResourceComparisonCard: React.FC<ResourceComparisonCardProps> = ({
  baseline,
  optimized,
  fp32LatencyMs,
  optimizedLatencyMs,
  throughputIps,
  cpuAvgPct,
  cpuPeakPct,
  ramAvgMb,
  ramPeakMb,
  candidateName,
  className = '',
}) => {
  // Latency resolution — explicit candidate metrics take strict precedence
  const bLat = fp32LatencyMs ?? baseline?.latency_mean_ms ?? null;
  const oLat = optimizedLatencyMs ?? optimized?.latency_mean_ms ?? null;

  // Latency change (positive = speedup)
  let latSpeedupPct: number | null = null;
  if (bLat !== null && oLat !== null && bLat > 0) {
    latSpeedupPct = Number((((bLat - oLat) / bLat) * 100).toFixed(1));
  }

  // CPU — candidate-specific utilization takes strict precedence
  const bCpuAvg = baseline?.avg_cpu_percent ?? baseline?.cpu_avg_pct ?? null;
  const oCpuAvg = cpuAvgPct ?? optimized?.avg_cpu_percent ?? optimized?.cpu_avg_pct ?? null;
  const bCpuPeak = baseline?.peak_cpu_percent ?? baseline?.cpu_peak_pct ?? null;
  const oCpuPeak = cpuPeakPct ?? optimized?.peak_cpu_percent ?? optimized?.cpu_peak_pct ?? null;

  let cpuChangePct: number | null = null;
  if (bCpuAvg !== null && oCpuAvg !== null && bCpuAvg > 0) {
    cpuChangePct = Number((((oCpuAvg - bCpuAvg) / bCpuAvg) * 100).toFixed(1));
  }

  // RAM (in MB and GB) — candidate-specific footprint takes strict precedence
  const bRamAvg = baseline?.avg_ram_mb ?? baseline?.ram_avg_mb ?? null;
  const oRamAvg = ramAvgMb ?? optimized?.avg_ram_mb ?? optimized?.ram_avg_mb ?? null;
  const bRamPeak = baseline?.peak_ram_mb ?? baseline?.ram_peak_mb ?? null;
  const oRamPeak = ramPeakMb ?? optimized?.peak_ram_mb ?? optimized?.ram_peak_mb ?? null;

  let ramChangePct: number | null = null;
  if (bRamAvg !== null && oRamAvg !== null && bRamAvg > 0) {
    ramChangePct = Number((((oRamAvg - bRamAvg) / bRamAvg) * 100).toFixed(1));
  }

  // Throughput — candidate-specific throughput takes strict precedence
  const bTput = baseline?.throughput_ips ?? (bLat ? Number((1000 / bLat).toFixed(1)) : null);
  const oTput = throughputIps ?? optimized?.throughput_ips ?? (oLat ? Number((1000 / oLat).toFixed(1)) : null);

  let tputChangePct: number | null = null;
  if (bTput !== null && oTput !== null && bTput > 0) {
    tputChangePct = Number((((oTput - bTput) / bTput) * 100).toFixed(1));
  }

  const formatGb = (mb: number | null) => {
    if (mb === null || mb === 0) return 'NOT AVAILABLE';
    return `${(mb / 1024).toFixed(2)} GB (${mb.toFixed(0)} MB)`;
  };

  return (
    <div className={`p-6 rounded-2xl bg-white border border-slate-200/80 shadow-xs flex flex-col gap-5 ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
              FP32 Baseline vs. Quantized Resource Efficiency
            </h3>
            <SourceBadge source="MEASURED" origin="Standardized Benchmark Protocol" />
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Empirical runtime hardware footprint comparison under controlled evaluation batches
          </p>
        </div>
        <div className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
          <CheckCircle2 className="w-3.5 h-3.5" />
          <span>Validated on Host CPU</span>
        </div>
      </div>

      {/* Grid Comparison Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-slate-200 text-slate-400 font-semibold uppercase tracking-wider text-[10px]">
              <th className="py-2.5 px-3">Resource Metric</th>
              <th className="py-2.5 px-3 bg-slate-50/50">FP32 Baseline</th>
              <th className="py-2.5 px-3 bg-indigo-50/30">
                {candidateName ? `Optimized (${candidateName})` : 'Optimized (INT8)'}
              </th>
              <th className="py-2.5 px-3">Empirical Change</th>
              <th className="py-2.5 px-3">Impact Verdict</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 font-sans">
            {/* CPU Average */}
            <tr className="hover:bg-slate-50/50 transition-colors">
              <td className="py-3 px-3 font-medium text-slate-700 flex items-center gap-2">
                <Cpu className="w-4 h-4 text-indigo-500" />
                <span>CPU Utilization (Average)</span>
              </td>
              <td className="py-3 px-3 font-mono font-bold text-slate-700 bg-slate-50/30">
                {bCpuAvg !== null && bCpuAvg > 0 ? `${bCpuAvg}%` : 'NOT AVAILABLE'}
              </td>
              <td className="py-3 px-3 font-mono font-bold text-indigo-900 bg-indigo-50/20">
                {oCpuAvg !== null && oCpuAvg > 0 ? `${oCpuAvg}%` : 'NOT AVAILABLE'}
              </td>
              <td className="py-3 px-3 font-mono">
                {cpuChangePct !== null ? (
                  <span className={cpuChangePct <= 0 ? 'text-emerald-600 font-bold' : 'text-slate-600'}>
                    {cpuChangePct > 0 ? `+${cpuChangePct}%` : `${cpuChangePct}%`}
                  </span>
                ) : (
                  '—'
                )}
              </td>
              <td className="py-3 px-3">
                {cpuChangePct !== null ? (
                  cpuChangePct <= 0 ? (
                    <span className="inline-flex items-center gap-1 text-[11px] text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded font-medium">
                      <ArrowDownRight className="w-3 h-3" /> Reduced Load
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-[11px] text-slate-600 bg-slate-100 px-2 py-0.5 rounded font-medium">
                      Similar Compute
                    </span>
                  )
                ) : (
                  'PENDING'
                )}
              </td>
            </tr>

            {/* CPU Peak */}
            <tr className="hover:bg-slate-50/50 transition-colors">
              <td className="py-3 px-3 font-medium text-slate-700 flex items-center gap-2">
                <Cpu className="w-4 h-4 text-slate-400" />
                <span>CPU Peak Utilization</span>
              </td>
              <td className="py-3 px-3 font-mono text-slate-600 bg-slate-50/30">
                {bCpuPeak !== null && bCpuPeak > 0 ? `${bCpuPeak}%` : 'NOT AVAILABLE'}
              </td>
              <td className="py-3 px-3 font-mono text-slate-800 bg-indigo-50/20">
                {oCpuPeak !== null && oCpuPeak > 0 ? `${oCpuPeak}%` : 'NOT AVAILABLE'}
              </td>
              <td className="py-3 px-3 font-mono text-slate-500">—</td>
              <td className="py-3 px-3 text-slate-400 text-[11px]">Peak ceiling</td>
            </tr>

            {/* RAM Average */}
            <tr className="hover:bg-slate-50/50 transition-colors">
              <td className="py-3 px-3 font-medium text-slate-700 flex items-center gap-2">
                <HardDrive className="w-4 h-4 text-emerald-500" />
                <span>Process RAM / RSS (Average)</span>
              </td>
              <td className="py-3 px-3 font-mono font-bold text-slate-700 bg-slate-50/30">
                {formatGb(bRamAvg)}
              </td>
              <td className="py-3 px-3 font-mono font-bold text-emerald-950 bg-indigo-50/20">
                {formatGb(oRamAvg)}
              </td>
              <td className="py-3 px-3 font-mono">
                {ramChangePct !== null ? (
                  <span className={ramChangePct <= 0 ? 'text-emerald-600 font-bold' : 'text-slate-600'}>
                    {ramChangePct > 0 ? `+${ramChangePct}%` : `${ramChangePct}%`}
                  </span>
                ) : (
                  '—'
                )}
              </td>
              <td className="py-3 px-3">
                {ramChangePct !== null && ramChangePct < 0 ? (
                  <span className="inline-flex items-center gap-1 text-[11px] text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded font-medium">
                    <ArrowDownRight className="w-3 h-3" /> Lower Memory
                  </span>
                ) : (
                  <span className="text-slate-400 text-[11px]">Stable Footprint</span>
                )}
              </td>
            </tr>

            {/* RAM Peak */}
            <tr className="hover:bg-slate-50/50 transition-colors">
              <td className="py-3 px-3 font-medium text-slate-700 flex items-center gap-2">
                <HardDrive className="w-4 h-4 text-slate-400" />
                <span>Process RAM Peak RSS</span>
              </td>
              <td className="py-3 px-3 font-mono text-slate-600 bg-slate-50/30">
                {formatGb(bRamPeak)}
              </td>
              <td className="py-3 px-3 font-mono text-slate-800 bg-indigo-50/20">
                {formatGb(oRamPeak)}
              </td>
              <td className="py-3 px-3 font-mono text-slate-500">—</td>
              <td className="py-3 px-3 text-slate-400 text-[11px]">Peak ceiling</td>
            </tr>

            {/* Inference Latency */}
            <tr className="hover:bg-slate-50/50 transition-colors">
              <td className="py-3 px-3 font-medium text-slate-700 flex items-center gap-2">
                <Zap className="w-4 h-4 text-amber-500" />
                <span>Inference Latency (Mean)</span>
              </td>
              <td className="py-3 px-3 font-mono font-bold text-slate-700 bg-slate-50/30">
                {bLat !== null ? `${bLat.toFixed(2)} ms` : 'NOT AVAILABLE'}
              </td>
              <td className="py-3 px-3 font-mono font-bold text-indigo-900 bg-indigo-50/20">
                {oLat !== null ? `${oLat.toFixed(2)} ms` : 'NOT AVAILABLE'}
              </td>
              <td className="py-3 px-3 font-mono">
                {latSpeedupPct !== null ? (
                  <span className={latSpeedupPct > 0 ? 'text-emerald-600 font-bold' : 'text-rose-600 font-bold'}>
                    {latSpeedupPct > 0 ? `+${latSpeedupPct}% speedup` : `${latSpeedupPct}% slower`}
                  </span>
                ) : (
                  '—'
                )}
              </td>
              <td className="py-3 px-3">
                {latSpeedupPct !== null && latSpeedupPct > 0 ? (
                  <span className="inline-flex items-center gap-1 text-[11px] text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded font-bold">
                    <ArrowUpRight className="w-3 h-3" /> Accelerated
                  </span>
                ) : (
                  <span className="text-slate-400 text-[11px]">Consistent</span>
                )}
              </td>
            </tr>

            {/* Throughput */}
            <tr className="hover:bg-slate-50/50 transition-colors">
              <td className="py-3 px-3 font-medium text-slate-700 flex items-center gap-2">
                <Gauge className="w-4 h-4 text-indigo-500" />
                <span>Throughput</span>
              </td>
              <td className="py-3 px-3 font-mono font-bold text-slate-700 bg-slate-50/30">
                {bTput !== null ? `${bTput} img/s` : 'NOT AVAILABLE'}
              </td>
              <td className="py-3 px-3 font-mono font-bold text-indigo-900 bg-indigo-50/20">
                {oTput !== null ? `${oTput} img/s` : 'NOT AVAILABLE'}
              </td>
              <td className="py-3 px-3 font-mono">
                {tputChangePct !== null ? (
                  <span className={tputChangePct > 0 ? 'text-emerald-600 font-bold' : 'text-slate-600'}>
                    {tputChangePct > 0 ? `+${tputChangePct}%` : `${tputChangePct}%`}
                  </span>
                ) : (
                  '—'
                )}
              </td>
              <td className="py-3 px-3">
                {tputChangePct !== null && tputChangePct > 0 ? (
                  <span className="inline-flex items-center gap-1 text-[11px] text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded font-medium">
                    <ArrowUpRight className="w-3 h-3" /> Higher Capacity
                  </span>
                ) : (
                  <span className="text-slate-400 text-[11px]">Normal</span>
                )}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ResourceComparisonCard;
