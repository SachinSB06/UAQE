import React, { useState } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';
import { TelemetryResponse, TelemetryPhaseData, TelemetrySample } from '../types/api';
import SourceBadge from '../components/ui/SourceBadge';
import { Activity, Cpu, HardDrive, AlertCircle } from 'lucide-react';

interface RuntimeTelemetryChartProps {
  telemetry: TelemetryResponse | null;
  className?: string;
}

export const RuntimeTelemetryChart: React.FC<RuntimeTelemetryChartProps> = ({
  telemetry,
  className = '',
}) => {
  const [activeMetric, setActiveMetric] = useState<'cpu' | 'ram' | 'both'>('both');
  const [selectedPhase, setSelectedPhase] = useState<'baseline' | 'final' | 'all'>('all');

  if (!telemetry || (!telemetry.baseline && !telemetry.final && (!telemetry.phases || Object.keys(telemetry.phases).length === 0))) {
    return (
      <div className={`p-6 rounded-2xl bg-white border border-slate-200/80 shadow-sm flex flex-col items-center justify-center min-h-[300px] text-center ${className}`}>
        <AlertCircle className="w-8 h-8 text-amber-500 mb-2" />
        <p className="text-xs font-mono font-bold tracking-wider text-slate-500 uppercase">
          NOT AVAILABLE FOR THIS MODEL/RUNTIME
        </p>
        <p className="text-xs text-slate-400 mt-1 max-w-sm">
          Telemetry stream was not recorded for this execution or host sampling is pending.
        </p>
      </div>
    );
  }

  // Aggregate or choose samples
  const baselineSamples = telemetry.baseline?.samples || [];
  const finalSamples = telemetry.final?.samples || [];

  // Create unified timeline data for plotting
  const maxLen = Math.max(baselineSamples.length, finalSamples.length);
  const chartData = [];

  for (let i = 0; i < maxLen; i++) {
    const b = baselineSamples[i];
    const f = finalSamples[i];
    chartData.push({
      sampleIndex: i + 1,
      timeSec: (i * 0.05).toFixed(2),
      baselineCpu: b ? b.process_cpu_percent : null,
      finalCpu: f ? f.process_cpu_percent : null,
      baselineRam: b ? b.process_ram_mb : null,
      finalRam: f ? f.process_ram_mb : null,
      systemCpu: f ? f.system_cpu_percent : b ? b.system_cpu_percent : null,
    });
  }

  return (
    <div className={`p-6 rounded-2xl bg-white border border-slate-200/80 shadow-sm flex flex-col gap-4 ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-indigo-50 flex items-center justify-center text-indigo-600">
            <Activity className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold text-slate-900 tracking-tight">
                Runtime Host CPU & RAM Telemetry
              </h3>
              <SourceBadge source="MEASURED" origin="psutil Host Sampler" />
            </div>
            <p className="text-xs text-slate-500">
              Live background sampling of process and system memory/compute footprints
            </p>
          </div>
        </div>

        {/* Metric Toggles */}
        <div className="flex items-center gap-2">
          <div className="flex bg-slate-100 p-0.5 rounded-lg text-xs font-medium">
            <button
              onClick={() => setActiveMetric('both')}
              className={`px-2.5 py-1 rounded-md transition-all ${
                activeMetric === 'both'
                  ? 'bg-white text-slate-900 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Dual View
            </button>
            <button
              onClick={() => setActiveMetric('cpu')}
              className={`px-2.5 py-1 rounded-md transition-all ${
                activeMetric === 'cpu'
                  ? 'bg-white text-slate-900 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Cpu className="w-3 h-3 inline mr-1" /> CPU
            </button>
            <button
              onClick={() => setActiveMetric('ram')}
              className={`px-2.5 py-1 rounded-md transition-all ${
                activeMetric === 'ram'
                  ? 'bg-white text-slate-900 shadow-xs font-semibold'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <HardDrive className="w-3 h-3 inline mr-1" /> RAM
            </button>
          </div>
        </div>
      </div>

      {/* Summary KPI Badges */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
          <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">
            Baseline Avg CPU
          </span>
          <span className="text-base font-bold text-slate-800 font-mono">
            {telemetry.baseline?.avg_cpu_percent?.toFixed(1) ?? 'N/A'}%
          </span>
          <span className="text-[10px] text-slate-400 block mt-0.5">
            Peak: {telemetry.baseline?.peak_cpu_percent?.toFixed(1) ?? 'N/A'}%
          </span>
        </div>

        <div className="p-3 rounded-xl bg-indigo-50/50 border border-indigo-100/60">
          <span className="text-[10px] font-semibold text-indigo-500 uppercase tracking-wider block">
            Optimized Avg CPU
          </span>
          <span className="text-base font-bold text-indigo-900 font-mono">
            {telemetry.final?.avg_cpu_percent?.toFixed(1) ?? 'N/A'}%
          </span>
          <span className="text-[10px] text-indigo-400 block mt-0.5">
            Peak: {telemetry.final?.peak_cpu_percent?.toFixed(1) ?? 'N/A'}%
          </span>
        </div>

        <div className="p-3 rounded-xl bg-slate-50 border border-slate-100">
          <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">
            Baseline Peak RAM
          </span>
          <span className="text-base font-bold text-slate-800 font-mono">
            {telemetry.baseline?.peak_ram_mb?.toFixed(1) ?? 'N/A'} MB
          </span>
          <span className="text-[10px] text-slate-400 block mt-0.5">
            Avg: {telemetry.baseline?.avg_ram_mb?.toFixed(1) ?? 'N/A'} MB
          </span>
        </div>

        <div className="p-3 rounded-xl bg-emerald-50/50 border border-emerald-100/60">
          <span className="text-[10px] font-semibold text-emerald-600 uppercase tracking-wider block">
            Optimized Peak RAM
          </span>
          <span className="text-base font-bold text-emerald-900 font-mono">
            {telemetry.final?.peak_ram_mb?.toFixed(1) ?? 'N/A'} MB
          </span>
          <span className="text-[10px] text-emerald-600 block mt-0.5">
            Avg: {telemetry.final?.avg_ram_mb?.toFixed(1) ?? 'N/A'} MB
          </span>
        </div>
      </div>

      {/* Main Chart Area */}
      <div className="h-[280px] w-full pt-2">
        <ResponsiveContainer width="100%" height="100%">
          {activeMetric === 'ram' ? (
            <AreaChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
              <defs>
                <linearGradient id="baselineRamGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#94a3b8" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#94a3b8" stopOpacity={0.0} />
                </linearGradient>
                <linearGradient id="finalRamGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="sampleIndex" stroke="#94a3b8" fontSize={11} tickLine={false} label={{ value: 'Sample # (50ms intervals)', position: 'insideBottomRight', offset: -5, fontSize: 10, fill: '#94a3b8' }} />
              <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} unit=" MB" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#ffffff',
                  borderRadius: '12px',
                  boxShadow: '0 4px 20px -2px rgba(0,0,0,0.1)',
                  border: '1px solid #e2e8f0',
                  fontSize: '12px',
                }}
              />
              <Legend verticalAlign="top" height={36} iconType="circle" />
              <Area type="monotone" dataKey="baselineRam" name="Baseline RAM (MB)" stroke="#64748b" strokeWidth={2} fillOpacity={1} fill="url(#baselineRamGrad)" />
              <Area type="monotone" dataKey="finalRam" name="Optimized INT8 RAM (MB)" stroke="#10b981" strokeWidth={2} fillOpacity={1} fill="url(#finalRamGrad)" />
            </AreaChart>
          ) : (
            <LineChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="sampleIndex" stroke="#94a3b8" fontSize={11} tickLine={false} label={{ value: 'Sample # (50ms intervals)', position: 'insideBottomRight', offset: -5, fontSize: 10, fill: '#94a3b8' }} />
              <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} unit="%" domain={[0, 'auto']} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#ffffff',
                  borderRadius: '12px',
                  boxShadow: '0 4px 20px -2px rgba(0,0,0,0.1)',
                  border: '1px solid #e2e8f0',
                  fontSize: '12px',
                }}
              />
              <Legend verticalAlign="top" height={36} iconType="circle" />
                  <Line type="monotone" dataKey="baselineCpu" name="Baseline CPU %" stroke="#64748b" strokeWidth={2} dot={false} strokeDasharray="3 3" />
                  <Line type="monotone" dataKey="finalCpu" name="Optimized INT8 CPU %" stroke="#6366f1" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="systemCpu" name="System Overall CPU %" stroke="#cbd5e1" strokeWidth={1} dot={false} />
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>

      {/* Provenance Footer */}
      <div className="pt-2 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
        <span>Host Environment: {telemetry.environment_info?.os} ({telemetry.environment_info?.cpu_count_logical} Logical Cores, {telemetry.environment_info?.total_ram_mb?.toFixed(0)} MB RAM)</span>
        <span>Target Status: <span className="font-semibold text-slate-600">{telemetry.target_hardware_status}</span></span>
      </div>
    </div>
  );
};

export default RuntimeTelemetryChart;
