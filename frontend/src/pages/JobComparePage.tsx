import React, { useState, useEffect } from 'react';
import {
  GitCompare,
  ArrowRight,
  Sparkles,
  Cpu,
  HardDrive,
  Zap,
  Gauge,
  CheckCircle2,
  AlertTriangle,
  Layers,
  ArrowDownRight,
  ArrowUpRight,
} from 'lucide-react';
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  CartesianGrid,
  BarChart,
  Bar,
  Legend,
} from 'recharts';
import { JobSummary } from '../types/api';
import { apiService } from '../services/api';
import SafetyBadge from '../components/ui/SafetyBadge';
import SourceBadge from '../components/ui/SourceBadge';

interface JobComparePageProps {
  jobs: JobSummary[];
  initialJobA?: string;
  initialJobB?: string;
  onSelectJob?: (jobId: string) => void;
}

export const JobComparePage: React.FC<JobComparePageProps> = ({
  jobs = [],
  initialJobA,
  initialJobB,
  onSelectJob,
}) => {
  const [jobAId, setJobAId] = useState<string>(initialJobA || (jobs[0]?.job_id ?? ''));
  const [jobBId, setJobBId] = useState<string>(initialJobB || (jobs[1]?.job_id ?? jobs[0]?.job_id ?? ''));
  const [comparisonData, setComparisonData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (jobs.length > 0 && !jobAId) setJobAId(jobs[0].job_id);
    if (jobs.length > 1 && !jobBId) setJobBId(jobs[1].job_id);
  }, [jobs]);

  useEffect(() => {
    if (!jobAId || !jobBId) return;

    let isMounted = true;
    setIsLoading(true);
    setError(null);

    apiService
      .compareJobs(jobAId, jobBId)
      .then((res) => {
        if (isMounted) {
          setComparisonData(res);
          setIsLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || 'Failed to compare jobs');
          setIsLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [jobAId, jobBId]);

  const a = comparisonData?.job_a;
  const b = comparisonData?.job_b;

  // Chart data
  const scatterData = [
    {
      name: a ? `Job A (${a.architecture})` : 'Job A',
      accuracy: a?.optimized_accuracy ? a.optimized_accuracy * 100 : 0,
      sizeMb: a?.optimized_size_bytes ? a.optimized_size_bytes / (1024 * 1024) : 0,
      latency: a?.optimized_latency_ms || 0,
      job: 'Job A',
    },
    {
      name: b ? `Job B (${b.architecture})` : 'Job B',
      accuracy: b?.optimized_accuracy ? b.optimized_accuracy * 100 : 0,
      sizeMb: b?.optimized_size_bytes ? b.optimized_size_bytes / (1024 * 1024) : 0,
      latency: b?.optimized_latency_ms || 0,
      job: 'Job B',
    },
  ];

  const resourceBarData = [
    {
      metric: 'CPU Avg (%)',
      'Job A': a?.optimized_cpu_avg_pct || a?.baseline_cpu_avg_pct || 0,
      'Job B': b?.optimized_cpu_avg_pct || b?.baseline_cpu_avg_pct || 0,
    },
    {
      metric: 'CPU Peak (%)',
      'Job A': a?.optimized_cpu_peak_pct || a?.baseline_cpu_peak_pct || 0,
      'Job B': b?.optimized_cpu_peak_pct || b?.baseline_cpu_peak_pct || 0,
    },
    {
      metric: 'RAM (MB / 100)',
      'Job A': a?.optimized_ram_avg_mb ? Number((a.optimized_ram_avg_mb / 100).toFixed(1)) : 0,
      'Job B': b?.optimized_ram_avg_mb ? Number((b.optimized_ram_avg_mb / 100).toFixed(1)) : 0,
    },
  ];

  return (
    <div className="flex flex-col gap-6 max-w-7xl mx-auto font-sans">
      {/* Top Header */}
      <div className="p-6 rounded-3xl bg-slate-900 text-white shadow-xl flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <GitCompare className="w-5 h-5 text-indigo-400" />
            <span className="text-xs font-mono font-bold tracking-wider text-indigo-400 uppercase">
              CROSS-RUN REPRODUCIBILITY & BENCHMARK INTELLIGENCE
            </span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight">Optimization Job Comparison</h1>
          <p className="text-xs text-slate-300 mt-1">
            Side-by-side empirical trade-off analysis between architectures, hardware targets, or quantization strategies
          </p>
        </div>
      </div>

      {/* Selectors Bar */}
      <div className="p-5 rounded-2xl bg-white border border-slate-200/80 shadow-xs flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex-1 flex flex-col gap-1.5">
          <label className="text-xs font-bold text-slate-500 uppercase tracking-wider">Job A (Reference)</label>
          <select
            value={jobAId}
            onChange={(e) => setJobAId(e.target.value)}
            className="w-full px-3 py-2 text-xs font-mono rounded-xl bg-slate-50 border border-slate-200 text-slate-900 focus:outline-hidden focus:ring-2 focus:ring-indigo-500"
          >
            {jobs.map((j) => (
              <option key={j.job_id} value={j.job_id}>
                {j.job_id} — {j.model_name} ({j.target_hardware})
              </option>
            ))}
          </select>
        </div>

        <div className="hidden sm:flex items-center justify-center pt-5">
          <div className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center text-slate-400">
            <ArrowRight className="w-4 h-4" />
          </div>
        </div>

        <div className="flex-1 flex flex-col gap-1.5">
          <label className="text-xs font-bold text-slate-500 uppercase tracking-wider">Job B (Comparison)</label>
          <select
            value={jobBId}
            onChange={(e) => setJobBId(e.target.value)}
            className="w-full px-3 py-2 text-xs font-mono rounded-xl bg-slate-50 border border-slate-200 text-slate-900 focus:outline-hidden focus:ring-2 focus:ring-indigo-500"
          >
            {jobs.map((j) => (
              <option key={j.job_id} value={j.job_id}>
                {j.job_id} — {j.model_name} ({j.target_hardware})
              </option>
            ))}
          </select>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-rose-600" />
          <span>{error}</span>
        </div>
      )}

      {isLoading ? (
        <div className="p-12 text-center text-slate-400 text-xs font-mono">Loading side-by-side job telemetry...</div>
      ) : a && b ? (
        <>
          {/* Comparison KPI Table */}
          <div className="p-6 rounded-2xl bg-white border border-slate-200/80 shadow-xs overflow-x-auto">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-4">
              <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
                Quantitative Comparison Matrix
              </h3>
              <SourceBadge source="MEASURED" origin="Authoritative Artifact Metrics" />
            </div>

            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-400 font-semibold uppercase tracking-wider text-[10px]">
                  <th className="py-2.5 px-3">Metric Dimension</th>
                  <th className="py-2.5 px-3 bg-indigo-50/40 text-indigo-900 font-bold">
                    Job A ({a.job_id.slice(-8)})
                  </th>
                  <th className="py-2.5 px-3 bg-emerald-50/40 text-emerald-900 font-bold">
                    Job B ({b.job_id.slice(-8)})
                  </th>
                  <th className="py-2.5 px-3">Direct Difference / Comparison</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {/* Model Architecture */}
                <tr>
                  <td className="py-2.5 px-3 font-medium text-slate-600">Model Architecture</td>
                  <td className="py-2.5 px-3 font-mono font-bold text-slate-900 bg-indigo-50/10">
                    {a.architecture}
                  </td>
                  <td className="py-2.5 px-3 font-mono font-bold text-slate-900 bg-emerald-50/10">
                    {b.architecture}
                  </td>
                  <td className="py-2.5 px-3 text-slate-500 font-mono text-[11px]">
                    {a.architecture === b.architecture ? 'Identical Architecture' : 'Different Architecture'}
                  </td>
                </tr>

                {/* Dataset */}
                <tr>
                  <td className="py-2.5 px-3 font-medium text-slate-600">Dataset Domain</td>
                  <td className="py-2.5 px-3 font-mono text-slate-800 bg-indigo-50/10">{a.dataset}</td>
                  <td className="py-2.5 px-3 font-mono text-slate-800 bg-emerald-50/10">{b.dataset}</td>
                  <td className="py-2.5 px-3 text-slate-500 text-[11px]">
                    {a.dataset === b.dataset ? 'Same Dataset' : 'Independent Dataset'}
                  </td>
                </tr>

                {/* Target Hardware */}
                <tr>
                  <td className="py-2.5 px-3 font-medium text-slate-600">Hardware Target</td>
                  <td className="py-2.5 px-3 font-mono text-slate-800 bg-indigo-50/10">{a.target}</td>
                  <td className="py-2.5 px-3 font-mono text-slate-800 bg-emerald-50/10">{b.target}</td>
                  <td className="py-2.5 px-3 text-slate-500 text-[11px]">
                    {a.target === b.target ? 'Same Target' : `${a.target} vs ${b.target}`}
                  </td>
                </tr>

                {/* Baseline Top-1 Accuracy */}
                <tr>
                  <td className="py-2.5 px-3 font-medium text-slate-600">FP32 Baseline Accuracy</td>
                  <td className="py-2.5 px-3 font-mono text-slate-700 bg-indigo-50/10">
                    {a.baseline_accuracy != null ? `${(a.baseline_accuracy * 100).toFixed(2)}%` : 'PENDING'}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-slate-700 bg-emerald-50/10">
                    {b.baseline_accuracy != null ? `${(b.baseline_accuracy * 100).toFixed(2)}%` : 'PENDING'}
                  </td>
                  <td className="py-2.5 px-3 text-slate-500 font-mono text-[11px]">—</td>
                </tr>

                {/* Optimized Top-1 Accuracy */}
                <tr>
                  <td className="py-2.5 px-3 font-medium text-slate-700 font-semibold">
                    Optimized INT8 Accuracy
                  </td>
                  <td className="py-2.5 px-3 font-mono font-bold text-slate-900 bg-indigo-50/10">
                    {a.optimized_accuracy != null ? `${(a.optimized_accuracy * 100).toFixed(2)}%` : 'PENDING'}
                  </td>
                  <td className="py-2.5 px-3 font-mono font-bold text-slate-900 bg-emerald-50/10">
                    {b.optimized_accuracy != null ? `${(b.optimized_accuracy * 100).toFixed(2)}%` : 'PENDING'}
                  </td>
                  <td className="py-2.5 px-3 font-mono font-bold">
                    {a.optimized_accuracy != null && b.optimized_accuracy != null ? (
                      (b.optimized_accuracy - a.optimized_accuracy) >= 0 ? (
                        <span className="text-emerald-600">
                          +{((b.optimized_accuracy - a.optimized_accuracy) * 100).toFixed(2)} pp (Job B higher)
                        </span>
                      ) : (
                        <span className="text-indigo-600">
                          {((b.optimized_accuracy - a.optimized_accuracy) * 100).toFixed(2)} pp (Job A higher)
                        </span>
                      )
                    ) : (
                      '—'
                    )}
                  </td>
                </tr>

                {/* Accuracy Delta / Loss */}
                <tr>
                  <td className="py-2.5 px-3 font-medium text-slate-600">Accuracy Loss (pp)</td>
                  <td className="py-2.5 px-3 font-mono bg-indigo-50/10">
                    {a.accuracy_loss_pp != null ? `${a.accuracy_loss_pp.toFixed(2)} pp` : '—'}
                  </td>
                  <td className="py-2.5 px-3 font-mono bg-emerald-50/10">
                    {b.accuracy_loss_pp != null ? `${b.accuracy_loss_pp.toFixed(2)} pp` : '—'}
                  </td>
                  <td className="py-2.5 px-3 text-slate-500 font-mono text-[11px]">
                    Lower loss is better
                  </td>
                </tr>

                {/* Model Size */}
                <tr>
                  <td className="py-2.5 px-3 font-medium text-slate-600">Optimized Footprint</td>
                  <td className="py-2.5 px-3 font-mono bg-indigo-50/10">
                    {a.optimized_size_bytes
                      ? `${(a.optimized_size_bytes / (1024 * 1024)).toFixed(2)} MB (-${a.storage_reduction_percent?.toFixed(1)}%)`
                      : '—'}
                  </td>
                  <td className="py-2.5 px-3 font-mono bg-emerald-50/10">
                    {b.optimized_size_bytes
                      ? `${(b.optimized_size_bytes / (1024 * 1024)).toFixed(2)} MB (-${b.storage_reduction_percent?.toFixed(1)}%)`
                      : '—'}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-[11px] text-slate-600">
                    {a.optimized_size_bytes && b.optimized_size_bytes ? (
                      a.optimized_size_bytes < b.optimized_size_bytes
                        ? `Job A is ${(b.optimized_size_bytes / a.optimized_size_bytes).toFixed(1)}x smaller`
                        : `Job B is ${(a.optimized_size_bytes / b.optimized_size_bytes).toFixed(1)}x smaller`
                    ) : (
                      '—'
                    )}
                  </td>
                </tr>

                {/* Host Latency */}
                <tr>
                  <td className="py-2.5 px-3 font-medium text-slate-600">Host Latency</td>
                  <td className="py-2.5 px-3 font-mono bg-indigo-50/10">
                    {a.optimized_latency_ms ? `${a.optimized_latency_ms.toFixed(2)} ms` : '—'}
                  </td>
                  <td className="py-2.5 px-3 font-mono bg-emerald-50/10">
                    {b.optimized_latency_ms ? `${b.optimized_latency_ms.toFixed(2)} ms` : '—'}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-[11px]">
                    {a.optimized_latency_ms && b.optimized_latency_ms ? (
                      a.optimized_latency_ms < b.optimized_latency_ms ? (
                        <span className="text-indigo-600">Job A is faster</span>
                      ) : (
                        <span className="text-emerald-600">Job B is faster</span>
                      )
                    ) : (
                      '—'
                    )}
                  </td>
                </tr>

                {/* Speedup */}
                <tr>
                  <td className="py-2.5 px-3 font-medium text-slate-600">Speedup Percentage</td>
                  <td className="py-2.5 px-3 font-mono bg-indigo-50/10">
                    {a.latency_change_percent != null ? `+${a.latency_change_percent.toFixed(1)}%` : '—'}
                  </td>
                  <td className="py-2.5 px-3 font-mono bg-emerald-50/10">
                    {b.latency_change_percent != null ? `+${b.latency_change_percent.toFixed(1)}%` : '—'}
                  </td>
                  <td className="py-2.5 px-3 text-slate-500 text-[11px] font-mono">Relative to own FP32</td>
                </tr>

                {/* Throughput */}
                <tr>
                  <td className="py-2.5 px-3 font-medium text-slate-600">Throughput</td>
                  <td className="py-2.5 px-3 font-mono bg-indigo-50/10">
                    {a.throughput_ips ? `${a.throughput_ips} img/s` : '—'}
                  </td>
                  <td className="py-2.5 px-3 font-mono bg-emerald-50/10">
                    {b.throughput_ips ? `${b.throughput_ips} img/s` : '—'}
                  </td>
                  <td className="py-2.5 px-3 text-slate-500 font-mono text-[11px]">—</td>
                </tr>

                {/* CPU Utilization */}
                <tr>
                  <td className="py-2.5 px-3 font-medium text-slate-600">CPU Usage (Avg / Peak)</td>
                  <td className="py-2.5 px-3 font-mono bg-indigo-50/10">
                    {a.optimized_cpu_avg_pct ? `${a.optimized_cpu_avg_pct}% / ${a.optimized_cpu_peak_pct}%` : 'PENDING'}
                  </td>
                  <td className="py-2.5 px-3 font-mono bg-emerald-50/10">
                    {b.optimized_cpu_avg_pct ? `${b.optimized_cpu_avg_pct}% / ${b.optimized_cpu_peak_pct}%` : 'PENDING'}
                  </td>
                  <td className="py-2.5 px-3 text-slate-500 text-[11px]">Real host psutil metrics</td>
                </tr>

                {/* RAM RSS */}
                <tr>
                  <td className="py-2.5 px-3 font-medium text-slate-600">Process RAM (Avg / Peak)</td>
                  <td className="py-2.5 px-3 font-mono bg-indigo-50/10">
                    {a.optimized_ram_avg_mb ? `${a.optimized_ram_avg_mb} MB / ${a.optimized_ram_peak_mb} MB` : 'PENDING'}
                  </td>
                  <td className="py-2.5 px-3 font-mono bg-emerald-50/10">
                    {b.optimized_ram_avg_mb ? `${b.optimized_ram_avg_mb} MB / ${b.optimized_ram_peak_mb} MB` : 'PENDING'}
                  </td>
                  <td className="py-2.5 px-3 text-slate-500 text-[11px]">Process RSS</td>
                </tr>

                {/* Safety Status */}
                <tr>
                  <td className="py-2.5 px-3 font-medium text-slate-600">Safety Policy Tier</td>
                  <td className="py-2.5 px-3 bg-indigo-50/10">
                    <SafetyBadge classification={a.safety_status} lossPP={a.accuracy_loss_pp} />
                  </td>
                  <td className="py-2.5 px-3 bg-emerald-50/10">
                    <SafetyBadge classification={b.safety_status} lossPP={b.accuracy_loss_pp} />
                  </td>
                  <td className="py-2.5 px-3 text-slate-500 text-[11px]">Governance Tier</td>
                </tr>

                {/* Final Verdict */}
                <tr>
                  <td className="py-2.5 px-3 font-medium text-slate-900 font-bold">Optimization Verdict</td>
                  <td className="py-2.5 px-3 font-mono font-extrabold bg-indigo-50/10">
                    <span className={a.verdict === 'VERIFIED' ? 'text-emerald-700' : 'text-rose-700'}>
                      {a.verdict}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 font-mono font-extrabold bg-emerald-50/10">
                    <span className={b.verdict === 'VERIFIED' ? 'text-emerald-700' : 'text-rose-700'}>
                      {b.verdict}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 text-slate-500 text-[11px]">Full Verification</td>
                </tr>
              </tbody>
            </table>
          </div>

          {/* Visual Comparison Charts */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Accuracy vs Size Tradeoff */}
            <div className="p-6 rounded-2xl bg-white border border-slate-200/80 shadow-xs flex flex-col gap-3">
              <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                Accuracy vs. Model Size Trade-Off
              </h4>
              <div className="h-60 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 10, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                    <XAxis
                      type="number"
                      dataKey="sizeMb"
                      name="Model Size"
                      unit="MB"
                      stroke="#94A3B8"
                      fontSize={11}
                    />
                    <YAxis
                      type="number"
                      dataKey="accuracy"
                      name="Top-1 Accuracy"
                      unit="%"
                      domain={[0, 100]}
                      stroke="#94A3B8"
                      fontSize={11}
                    />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter name="Jobs" data={scatterData} fill="#4F46E5" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Hardware Compute & Memory Comparison */}
            <div className="p-6 rounded-2xl bg-white border border-slate-200/80 shadow-xs flex flex-col gap-3">
              <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                Hardware Compute & Memory Footprint
              </h4>
              <div className="h-60 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={resourceBarData} margin={{ top: 20, right: 20, bottom: 10, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                    <XAxis dataKey="metric" stroke="#94A3B8" fontSize={11} />
                    <YAxis stroke="#94A3B8" fontSize={11} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="Job A" fill="#4F46E5" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="Job B" fill="#10B981" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className="p-12 text-center text-slate-400 text-xs font-mono">
          Select two jobs from the dropdowns above to compare their empirical performance.
        </div>
      )}
    </div>
  );
};

export default JobComparePage;
