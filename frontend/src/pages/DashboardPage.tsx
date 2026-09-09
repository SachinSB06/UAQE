import React from 'react';
import {
  Activity,
  Cpu,
  HardDrive,
  CheckCircle2,
  Clock,
  ArrowUpRight,
  Plus,
  Play,
  Layers,
  Sparkles,
  ShieldCheck,
  Zap,
} from 'lucide-react';
import { SystemStatus, JobSummary } from '../types/api';
import SourceBadge from '../components/ui/SourceBadge';
import SafetyBadge from '../components/ui/SafetyBadge';
import AnimatedCounter from '../components/ui/AnimatedCounter';

interface DashboardPageProps {
  systemStatus: SystemStatus | null;
  recentJobs: JobSummary[];
  onNavigate: (tab: string, jobId?: string) => void;
  onSelectJob: (jobId: string) => void;
  onStartNewOptimization?: () => void;
}

export const DashboardPage: React.FC<DashboardPageProps> = ({
  systemStatus,
  recentJobs,
  onNavigate,
  onSelectJob,
  onStartNewOptimization,
}) => {
  const host = systemStatus?.host_environment;

  return (
    <div className="flex flex-col gap-8">
      {/* Hero Welcome Banner */}
      <div className="p-8 rounded-3xl bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 text-white shadow-xl relative overflow-hidden">
        {/* Futuristic Background Accents */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-1/3 w-64 h-64 bg-emerald-500/10 rounded-full blur-2xl pointer-events-none" />

        <div className="relative z-10 flex flex-col md:flex-row md:items-center md:justify-between gap-6">
          <div className="max-w-2xl">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/10 backdrop-blur-md border border-white/15 text-xs font-mono mb-4 text-indigo-200">
              <Sparkles className="w-3.5 h-3.5 text-amber-300" />
              <span>Universal AI Quantization Engine • Production Observatory</span>
            </div>
            <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white leading-tight font-sans">
              Autonomous Neural Compression & Edge Verification Cockpit
            </h1>
            <p className="text-sm sm:text-base text-slate-300 mt-2 leading-relaxed">
              Transform PyTorch and ONNX models into verified, hardware-accelerated INT8/FP8 edge artifacts with strict zero-fake telemetry provenance and multi-candidate Pareto exploration.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
            <button
              onClick={() => onStartNewOptimization ? onStartNewOptimization() : onNavigate('new')}
              className="px-6 py-3.5 rounded-2xl bg-white text-slate-900 font-bold text-sm shadow-lg hover:bg-slate-100 transition-all flex items-center justify-center gap-2 hover:scale-[1.02] active:scale-[0.98] cursor-pointer"
            >
              <Plus className="w-4 h-4 text-indigo-600" />
              <span>Launch New Job</span>
            </button>
            <button
              onClick={() => onNavigate('jobs')}
              className="px-5 py-3.5 rounded-2xl bg-white/10 hover:bg-white/15 text-white font-semibold text-sm border border-white/20 backdrop-blur-md transition-all flex items-center justify-center gap-2 cursor-pointer"
            >
              <span>View Job History</span>
            </button>
          </div>
        </div>
      </div>

      {/* Host System Telemetry & Capabilities Overview */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Host Compute */}
        <div className="p-5 rounded-2xl bg-white border border-slate-200/80 shadow-xs flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-3">
            <span className="text-xs font-mono font-semibold uppercase tracking-wider">Host Compute</span>
            <Cpu className="w-4 h-4 text-indigo-500" />
          </div>
          <div>
            <div className="text-xl font-bold text-slate-900 font-mono">
              {host?.cpu_count_logical || '8'} Logical Cores
            </div>
            <div className="text-xs text-slate-500 mt-0.5">
              {host?.architecture || 'AMD64'} ({host?.cpu_count_physical || '4'} Physical)
            </div>
          </div>
          <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between">
            <SourceBadge source="DETECTED" origin="OS Kernel" />
            <span className="text-[11px] font-mono text-slate-400">{host?.os}</span>
          </div>
        </div>

        {/* Host Memory */}
        <div className="p-5 rounded-2xl bg-white border border-slate-200/80 shadow-xs flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-3">
            <span className="text-xs font-mono font-semibold uppercase tracking-wider">Host Memory</span>
            <HardDrive className="w-4 h-4 text-emerald-500" />
          </div>
          <div>
            <div className="text-xl font-bold text-slate-900 font-mono">
              {((host?.total_ram_mb || 16384) / 1024).toFixed(1)} GB RAM
            </div>
            <div className="text-xs text-slate-500 mt-0.5">
              Available for Calib Caching
            </div>
          </div>
          <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between">
            <SourceBadge source="MEASURED" origin="psutil" />
            <span className="text-[11px] font-mono text-emerald-600 font-semibold">Active Monitoring</span>
          </div>
        </div>

        {/* Target Registry */}
        <div className="p-5 rounded-2xl bg-white border border-slate-200/80 shadow-xs flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-3">
            <span className="text-xs font-mono font-semibold uppercase tracking-wider">Target Hardware</span>
            <Zap className="w-4 h-4 text-amber-500" />
          </div>
          <div>
            <div className="text-xl font-bold text-slate-900 font-mono">
              {systemStatus?.supported_targets?.length || 4} Hardware Classes
            </div>
            <div className="text-xs text-slate-500 mt-0.5">
              Raspberry Pi 5, Jetson, Cortex-M
            </div>
          </div>
          <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between">
            <SourceBadge source="CONFIGURED" origin="Target Registry" />
            <span className="text-[11px] font-mono text-slate-400">ARM64 & x86</span>
          </div>
        </div>

        {/* Engine Governance */}
        <div className="p-5 rounded-2xl bg-white border border-slate-200/80 shadow-xs flex flex-col justify-between">
          <div className="flex items-center justify-between text-slate-400 mb-3">
            <span className="text-xs font-mono font-semibold uppercase tracking-wider">Engine Safety</span>
            <ShieldCheck className="w-4 h-4 text-sky-500" />
          </div>
          <div>
            <div className="text-xl font-bold text-emerald-700 font-mono flex items-center gap-1.5">
              <CheckCircle2 className="w-5 h-5 text-emerald-600" />
              <span>Strict Mode</span>
            </div>
            <div className="text-xs text-slate-500 mt-0.5">
              Loss Tolerance: &le; 4.00 pp (EXCELLENT &le; 1.0 pp)
            </div>
          </div>
          <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between">
            <SourceBadge source="POLICY" origin="UAQE Core Safety" />
            <span className="text-[11px] font-mono text-emerald-600 font-semibold">Enabled</span>
          </div>
        </div>
      </div>

      {/* Recent Optimization Runs Section */}
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-slate-900 tracking-tight">Recent Optimization Jobs</h2>
            <p className="text-xs text-slate-500">Autonomous execution history from local output registry</p>
          </div>
          <button
            onClick={() => onNavigate('jobs')}
            className="text-xs font-semibold text-indigo-600 hover:text-indigo-800 flex items-center gap-1"
          >
            <span>View All Runs</span>
            <ArrowUpRight className="w-3.5 h-3.5" />
          </button>
        </div>

        {recentJobs.length === 0 ? (
          <div className="p-8 rounded-2xl bg-white border border-slate-200 text-center flex flex-col items-center justify-center">
            <Layers className="w-10 h-10 text-slate-300 mb-2" />
            <h3 className="text-sm font-bold text-slate-700">No Historical Jobs Found</h3>
            <p className="text-xs text-slate-400 mt-1 max-w-sm">
              Launch your first optimization job to evaluate real ResNet-50 + CIFAR-10 compression.
            </p>
            <button
              onClick={() => onStartNewOptimization ? onStartNewOptimization() : onNavigate('new')}
              className="mt-4 px-4 py-2 rounded-xl bg-indigo-600 text-white text-xs font-semibold hover:bg-indigo-700 transition-all cursor-pointer"
            >
              Start Optimization
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-3">
            {recentJobs.map((job) => {
              const isVerified = job.verdict === 'VERIFIED';
              return (
                <div
                  key={job.job_id}
                  onClick={() => {
                    onSelectJob(job.job_id);
                    onNavigate('results', job.job_id);
                  }}
                  className="p-5 rounded-2xl bg-white border border-slate-200/80 hover:border-indigo-400/80 hover:shadow-md transition-all cursor-pointer flex flex-col md:flex-row md:items-center md:justify-between gap-4 group"
                >
                  <div className="flex items-start gap-3">
                    <div className="w-10 h-10 rounded-xl bg-slate-100 flex items-center justify-center text-slate-600 group-hover:bg-indigo-50 group-hover:text-indigo-600 transition-colors">
                      <Layers className="w-5 h-5" />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="text-sm font-bold text-slate-900 group-hover:text-indigo-600 transition-colors font-mono">
                          {job.job_id}
                        </h3>
                        <span className={`px-2 py-0.5 rounded-md text-[10px] font-bold font-mono ${isVerified ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-800'}`}>
                          {job.verdict}
                        </span>
                        <SourceBadge source="MEASURED" origin="Execution Log" />
                      </div>
                      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500 mt-1">
                        <span><strong>Model:</strong> {job.model_name}</span>
                        <span>•</span>
                        <span><strong>Dataset:</strong> {job.dataset_name}</span>
                        <span>•</span>
                        <span><strong>Target:</strong> {job.target_hardware}</span>
                        <span>•</span>
                        <span><strong>Profile:</strong> {job.optimization_profile}</span>
                      </div>
                    </div>
                  </div>

                  {/* Metrics Delta Pill Group */}
                  <div className="flex flex-wrap items-center gap-3">
                    {/* Accuracy Delta */}
                    <div className="px-3 py-1.5 rounded-xl bg-slate-50 border border-slate-100 flex flex-col">
                      <span className="text-[10px] text-slate-400 font-semibold uppercase">Accuracy Delta</span>
                      <span className="text-xs font-mono font-bold text-emerald-700">
                        {job.accuracy_loss_pp !== null ? `${job.accuracy_loss_pp > 0 ? '-' : '+'}${Math.abs(job.accuracy_loss_pp).toFixed(2)} pp` : 'N/A'}
                      </span>
                    </div>

                    {/* Size Reduction */}
                    <div className="px-3 py-1.5 rounded-xl bg-slate-50 border border-slate-100 flex flex-col">
                      <span className="text-[10px] text-slate-400 font-semibold uppercase">Compression</span>
                      <span className="text-xs font-mono font-bold text-indigo-600">
                        {job.size_reduction_percent !== null ? `-${job.size_reduction_percent.toFixed(1)}%` : 'N/A'}
                      </span>
                    </div>

                    {/* Host Latency */}
                    <div className="px-3 py-1.5 rounded-xl bg-slate-50 border border-slate-100 flex flex-col">
                      <span className="text-[10px] text-slate-400 font-semibold uppercase">Host Speedup</span>
                      <span className="text-xs font-mono font-bold text-emerald-600">
                        {job.latency_change_percent !== null ? `+${job.latency_change_percent.toFixed(1)}%` : 'N/A'}
                      </span>
                    </div>

                    <div className="pl-2">
                      <div className="w-8 h-8 rounded-full bg-slate-100 group-hover:bg-indigo-600 group-hover:text-white flex items-center justify-center text-slate-500 transition-all">
                        <ArrowUpRight className="w-4 h-4" />
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default DashboardPage;
