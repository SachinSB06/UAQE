import React, { useState, useMemo } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import { Cpu, HardDrive, Pause, Play, Activity, Clock } from 'lucide-react';

export interface TelemetryPoint {
  timestamp: number;
  system_cpu_percent?: number;
  process_cpu_percent?: number;
  cpu_system_pct?: number;
  cpu_process_pct?: number;
  system_ram_percent?: number;
  ram_used_pct?: number;
  process_ram_mb?: number;
  ram_process_mb?: number;
  peak_process_ram_mb?: number;
  system_ram_used_mb?: number;
  system_ram_available_mb?: number;
  cpu_frequency_current_mhz?: number;
  cpu_count_logical?: number;
  phase?: string;
  [key: string]: any;
}

interface TaskManagerGraphProps {
  metricType: 'cpu' | 'ram';
  samples: TelemetryPoint[];
  isLive?: boolean;
  isReplay?: boolean;
  className?: string;
  height?: number;
}

export const TaskManagerGraph: React.FC<TaskManagerGraphProps> = ({
  metricType,
  samples = [],
  isLive = false,
  isReplay = false,
  className = '',
  height = 240,
}) => {
  const [windowRange, setWindowRange] = useState<'30s' | '60s' | 'full'>('60s');
  const [isPaused, setIsPaused] = useState(false);

  // Normalize samples into uniform data points
  const normalizedData = useMemo(() => {
    if (!samples || samples.length === 0) return [];

    const startTime = samples[0]?.timestamp || Date.now() / 1000;

    return samples.map((s, idx) => {
      const ts = s.timestamp || startTime + idx * 0.5;
      const relSec = Math.max(0, ts - startTime);

      const sysCpu = Number(s.system_cpu_percent ?? s.cpu_system_pct ?? 0);
      const procCpu = Number(s.process_cpu_percent ?? s.cpu_process_pct ?? 0);
      const sysRamPct = Number(s.system_ram_percent ?? s.ram_used_pct ?? 0);
      const procRamMb = Number(s.process_ram_mb ?? s.ram_process_mb ?? 0);
      const peakRamMb = Number(s.peak_process_ram_mb ?? procRamMb);

      return {
        idx,
        timestamp: ts,
        relSec: Number(relSec.toFixed(1)),
        timeLabel: `${relSec.toFixed(0)}s`,
        systemCpu: Math.min(100, Math.max(0, sysCpu)),
        processCpu: Math.min(100, Math.max(0, procCpu)),
        systemRamPct: Math.min(100, Math.max(0, sysRamPct)),
        processRamMb: Math.max(0, procRamMb),
        processRamGb: Number((procRamMb / 1024).toFixed(2)),
        peakRamMb: Math.max(0, peakRamMb),
        peakRamGb: Number((peakRamMb / 1024).toFixed(2)),
        sysRamUsedMb: s.system_ram_used_mb ? Number((s.system_ram_used_mb / 1024).toFixed(2)) : undefined,
        sysRamAvailMb: s.system_ram_available_mb ? Number((s.system_ram_available_mb / 1024).toFixed(2)) : undefined,
        cpuFreqMhz: s.cpu_frequency_current_mhz,
        cpuCores: s.cpu_count_logical,
        phase: s.phase || 'EXECUTING',
      };
    });
  }, [samples]);

  // Filter based on selected rolling window
  const windowedData = useMemo(() => {
    if (normalizedData.length === 0) return [];
    if (windowRange === 'full') return normalizedData;

    const maxSec = normalizedData[normalizedData.length - 1]?.relSec || 0;
    const windowSec = windowRange === '30s' ? 30 : 60;
    const minSec = Math.max(0, maxSec - windowSec);

    return normalizedData.filter((d) => d.relSec >= minSec);
  }, [normalizedData, windowRange]);

  // Compute live KPIs
  const kpis = useMemo(() => {
    if (normalizedData.length === 0) {
      return {
        currentSys: null,
        currentProc: null,
        avgSys: null,
        peakSys: null,
        avgProc: null,
        peakProc: null,
        cores: null,
        freq: null,
      };
    }

    const last = normalizedData[normalizedData.length - 1];

    if (metricType === 'cpu') {
      const sysVals = normalizedData.map((d) => d.systemCpu);
      const procVals = normalizedData.map((d) => d.processCpu);
      const avgSys = sysVals.reduce((a, b) => a + b, 0) / sysVals.length;
      const peakSys = Math.max(...sysVals);
      const avgProc = procVals.reduce((a, b) => a + b, 0) / procVals.length;
      const peakProc = Math.max(...procVals);

      return {
        currentSys: last.systemCpu,
        currentProc: last.processCpu,
        avgSys: Number(avgSys.toFixed(1)),
        peakSys: Number(peakSys.toFixed(1)),
        avgProc: Number(avgProc.toFixed(1)),
        peakProc: Number(peakProc.toFixed(1)),
        cores: last.cpuCores,
        freq: last.cpuFreqMhz,
      };
    } else {
      const ramPctVals = normalizedData.map((d) => d.systemRamPct);
      const procMbVals = normalizedData.map((d) => d.processRamMb);
      const avgSys = ramPctVals.reduce((a, b) => a + b, 0) / ramPctVals.length;
      const peakSys = Math.max(...ramPctVals);
      const avgProc = procMbVals.reduce((a, b) => a + b, 0) / procMbVals.length;
      const peakProc = Math.max(...procMbVals);

      return {
        currentSys: last.systemRamPct,
        currentProc: last.processRamMb,
        currentProcGb: last.processRamGb,
        avgSys: Number(avgSys.toFixed(1)),
        peakSys: Number(peakSys.toFixed(1)),
        avgProc: Number(avgProc.toFixed(0)),
        avgProcGb: Number((avgProc / 1024).toFixed(2)),
        peakProc: Number(peakProc.toFixed(0)),
        peakProcGb: Number((peakProc / 1024).toFixed(2)),
        sysUsedGb: last.sysRamUsedMb,
        sysAvailGb: last.sysRamAvailMb,
      };
    }
  }, [normalizedData, metricType]);

  const hasData = normalizedData.length > 0;

  return (
    <div
      className={`p-5 rounded-2xl bg-white border border-slate-200/80 shadow-xs flex flex-col gap-3 font-sans ${className}`}
    >
      {/* Header Controls Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2.5">
          <div
            className={`w-8 h-8 rounded-lg flex items-center justify-center ${
              metricType === 'cpu'
                ? 'bg-indigo-50 text-indigo-600'
                : 'bg-emerald-50 text-emerald-600'
            }`}
          >
            {metricType === 'cpu' ? <Cpu className="w-4 h-4" /> : <HardDrive className="w-4 h-4" />}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                {metricType === 'cpu' ? 'System & Process CPU Utilization' : 'System & Process RAM Utilization'}
              </h3>
              {isLive && !isPaused && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-200 animate-pulse">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span>
                  LIVE OBSERVABILITY
                </span>
              )}
              {isLive && isPaused && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-50 text-amber-700 border border-amber-200">
                  PAUSED
                </span>
              )}
              {isReplay && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-slate-100 text-slate-700 border border-slate-200">
                  <Clock className="w-2.5 h-2.5" />
                  REPLAY
                </span>
              )}
              {!isLive && !isReplay && hasData && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-slate-50 text-slate-600 border border-slate-200">
                  FROZEN (COMPLETED)
                </span>
              )}
            </div>
            <p className="text-[11px] text-slate-500 mt-0.5">
              {metricType === 'cpu'
                ? 'Continuous sampling via psutil host daemon'
                : 'Process RSS and physical memory consumption'}
            </p>
          </div>
        </div>

        {/* Rolling Window & Pause Controls */}
        <div className="flex items-center gap-2">
          {isLive && (
            <button
              onClick={() => setIsPaused(!isPaused)}
              className="px-2 py-1 rounded-md text-xs font-medium border border-slate-200 text-slate-700 hover:bg-slate-50 flex items-center gap-1 transition-all cursor-pointer"
              title={isPaused ? 'Resume live autoscroll' : 'Pause live view'}
            >
              {isPaused ? <Play className="w-3 h-3 text-emerald-600" /> : <Pause className="w-3 h-3 text-slate-500" />}
              {isPaused ? 'Resume' : 'Pause'}
            </button>
          )}

          <div className="flex bg-slate-100 p-0.5 rounded-lg text-[11px] font-medium text-slate-600">
            <button
              onClick={() => setWindowRange('30s')}
              className={`px-2 py-1 rounded-md transition-all ${
                windowRange === '30s' ? 'bg-white text-slate-900 shadow-xs font-bold' : 'hover:text-slate-900'
              }`}
            >
              30s
            </button>
            <button
              onClick={() => setWindowRange('60s')}
              className={`px-2 py-1 rounded-md transition-all ${
                windowRange === '60s' ? 'bg-white text-slate-900 shadow-xs font-bold' : 'hover:text-slate-900'
              }`}
            >
              60s
            </button>
            <button
              onClick={() => setWindowRange('full')}
              className={`px-2 py-1 rounded-md transition-all ${
                windowRange === 'full' ? 'bg-white text-slate-900 shadow-xs font-bold' : 'hover:text-slate-900'
              }`}
            >
              Full
            </button>
          </div>
        </div>
      </div>

      {/* KPI Stats Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-2">
        {metricType === 'cpu' ? (
          <>
            <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
              <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">
                System CPU
              </span>
              <span className="text-sm font-bold text-slate-900 font-mono">
                {kpis.currentSys !== null ? `${kpis.currentSys}%` : 'PENDING'}
              </span>
            </div>
            <div className="p-2.5 rounded-xl bg-indigo-50/50 border border-indigo-100/60">
              <span className="text-[10px] font-semibold text-indigo-500 uppercase tracking-wider block">
                Process CPU
              </span>
              <span className="text-sm font-bold text-indigo-900 font-mono">
                {kpis.currentProc !== null ? `${kpis.currentProc}%` : 'PENDING'}
              </span>
            </div>
            <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
              <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">
                System Avg
              </span>
              <span className="text-sm font-bold text-slate-800 font-mono">
                {kpis.avgSys !== null ? `${kpis.avgSys}%` : '—'}
              </span>
            </div>
            <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
              <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">
                System Peak
              </span>
              <span className="text-sm font-bold text-slate-800 font-mono">
                {kpis.peakSys !== null ? `${kpis.peakSys}%` : '—'}
              </span>
            </div>
            <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
              <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">
                Logical Cores
              </span>
              <span className="text-sm font-bold text-slate-800 font-mono">
                {kpis.cores ?? 'MEASURED'}
              </span>
            </div>
            <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
              <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">
                Frequency
              </span>
              <span className="text-sm font-bold text-slate-800 font-mono">
                {kpis.freq ? `${kpis.freq} MHz` : 'MEASURED'}
              </span>
            </div>
          </>
        ) : (
          <>
            <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
              <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">
                System RAM %
              </span>
              <span className="text-sm font-bold text-slate-900 font-mono">
                {kpis.currentSys !== null ? `${kpis.currentSys}%` : 'PENDING'}
              </span>
            </div>
            <div className="p-2.5 rounded-xl bg-emerald-50/50 border border-emerald-100/60">
              <span className="text-[10px] font-semibold text-emerald-600 uppercase tracking-wider block">
                Process RSS
              </span>
              <span className="text-sm font-bold text-emerald-950 font-mono">
                {kpis.currentProc !== null ? `${kpis.currentProc} MB` : 'PENDING'}
              </span>
            </div>
            <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
              <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">
                Peak Process
              </span>
              <span className="text-sm font-bold text-slate-800 font-mono">
                {kpis.peakProc !== null ? `${kpis.peakProc} MB` : '—'}
              </span>
            </div>
            <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
              <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">
                Avg Process
              </span>
              <span className="text-sm font-bold text-slate-800 font-mono">
                {kpis.avgProc !== null ? `${kpis.avgProc} MB` : '—'}
              </span>
            </div>
            <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
              <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">
                System Used
              </span>
              <span className="text-sm font-bold text-slate-800 font-mono">
                {kpis.sysUsedGb ? `${kpis.sysUsedGb} GB` : 'MEASURED'}
              </span>
            </div>
            <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
              <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block">
                System Avail
              </span>
              <span className="text-sm font-bold text-slate-800 font-mono">
                {kpis.sysAvailGb ? `${kpis.sysAvailGb} GB` : 'MEASURED'}
              </span>
            </div>
          </>
        )}
      </div>

      {/* Main Task Manager Chart */}
      <div style={{ width: '100%', height }} className="relative bg-slate-50/50 rounded-xl p-2 border border-slate-100">
        {!hasData ? (
          <div className="w-full h-full flex flex-col items-center justify-center text-center p-6">
            <Activity className="w-6 h-6 text-slate-300 animate-spin mb-2" />
            <p className="text-xs font-mono font-bold text-slate-500 uppercase tracking-wider">
              {isLive ? 'INITIALIZING HOST SAMPLER...' : 'NOT AVAILABLE'}
            </p>
            <p className="text-[11px] text-slate-400 mt-0.5">
              {isLive ? 'Streaming CPU & RAM time series via SSE...' : 'No telemetry samples stored for this run.'}
            </p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={windowedData} margin={{ top: 10, right: 15, left: -10, bottom: 5 }}>
              <defs>
                <linearGradient id="sysGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#4F46E5" stopOpacity={0.25} />
                  <stop offset="95%" stopColor="#4F46E5" stopOpacity={0.0} />
                </linearGradient>
                <linearGradient id="procGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10B981" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#10B981" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              {/* Task Manager Precision Grid */}
              <CartesianGrid strokeDasharray="2 2" stroke="#E2E8F0" vertical={true} horizontal={true} />
              <XAxis
                dataKey="timeLabel"
                stroke="#94A3B8"
                fontSize={10}
                tickLine={false}
                axisLine={{ stroke: '#CBD5E1' }}
              />
              <YAxis
                domain={metricType === 'cpu' ? [0, 100] : [0, 'auto']}
                stroke="#94A3B8"
                fontSize={10}
                tickLine={false}
                axisLine={{ stroke: '#CBD5E1' }}
                unit={metricType === 'cpu' ? '%' : 'MB'}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#FFFFFF',
                  borderColor: '#E2E8F0',
                  borderRadius: '0.75rem',
                  fontSize: '11px',
                  boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                }}
                formatter={(val: any, name: string) => {
                  if (metricType === 'cpu') {
                    return [`${val}%`, name === 'systemCpu' ? 'System CPU' : 'Process CPU'];
                  } else {
                    return [
                      name === 'processRamMb' ? `${val} MB` : `${val}%`,
                      name === 'processRamMb' ? 'Process RSS' : 'System RAM %',
                    ];
                  }
                }}
                labelFormatter={(label) => `Time: ${label}`}
              />
              {metricType === 'cpu' ? (
                <>
                  <Area
                    type="monotone"
                    dataKey="systemCpu"
                    stroke="#4F46E5"
                    strokeWidth={1.8}
                    fillOpacity={1}
                    fill="url(#sysGradient)"
                    name="systemCpu"
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="processCpu"
                    stroke="#10B981"
                    strokeWidth={1.5}
                    strokeDasharray="3 3"
                    dot={false}
                    name="processCpu"
                    isAnimationActive={false}
                  />
                </>
              ) : (
                <>
                  <Area
                    type="monotone"
                    dataKey="processRamMb"
                    stroke="#10B981"
                    strokeWidth={2}
                    fillOpacity={1}
                    fill="url(#procGradient)"
                    name="processRamMb"
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="systemRamPct"
                    stroke="#6366F1"
                    strokeWidth={1.2}
                    dot={false}
                    name="systemRamPct"
                    isAnimationActive={false}
                  />
                </>
              )}
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Legend & Legend Indicator */}
      <div className="flex items-center justify-between text-[11px] text-slate-500 pt-1">
        <div className="flex items-center gap-4">
          {metricType === 'cpu' ? (
            <>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-indigo-600"></span>
                <span>System CPU (%)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-0.5 bg-emerald-500 border-t border-dashed border-emerald-500"></span>
                <span>Process CPU (%)</span>
              </div>
            </>
          ) : (
            <>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                <span>Process RSS (MB)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-0.5 bg-indigo-500"></span>
                <span>System RAM (%)</span>
              </div>
            </>
          )}
        </div>
        <div className="text-slate-400 font-mono text-[10px]">
          {hasData ? `${normalizedData.length} samples recorded` : 'No samples'}
        </div>
      </div>
    </div>
  );
};

export default TaskManagerGraph;
