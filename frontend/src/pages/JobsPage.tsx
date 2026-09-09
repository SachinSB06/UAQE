import React, { useState } from 'react';
import {
  Layers,
  Search,
  CheckCircle2,
  Clock,
  ArrowUpRight,
  Filter,
  Eye,
  Sliders,
  Sparkles,
} from 'lucide-react';
import { JobSummary } from '../types/api';
import SourceBadge from '../components/ui/SourceBadge';

interface JobsPageProps {
  jobs: JobSummary[];
  onSelectJob: (jobId: string) => void;
  onNavigate: (tab: string, jobId?: string) => void;
}

export const JobsPage: React.FC<JobsPageProps> = ({
  jobs,
  onSelectJob,
  onNavigate,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterVerdict, setFilterVerdict] = useState<string>('ALL');

  const filteredJobs = jobs.filter((j) => {
    const matchesSearch =
      j.job_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      j.model_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      j.target_hardware.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesFilter =
      filterVerdict === 'ALL' ||
      (filterVerdict === 'VERIFIED' && j.verdict === 'VERIFIED') ||
      (filterVerdict === 'CAVEAT' && j.verdict.includes('CAVEAT'));

    return matchesSearch && matchesFilter;
  });

  return (
    <div className="flex flex-col gap-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex flex-col gap-1 pb-4 border-b border-slate-200">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            Job History & Execution Archive
          </h1>
          <SourceBadge source="MEASURED" origin="Local Output Job Registry" />
        </div>
        <p className="text-sm text-slate-500">
          Browse, inspect, and replay past autonomous quantization jobs and verification reports
        </p>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 p-4 rounded-2xl bg-white border border-slate-200/80 shadow-xs">
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by Job ID, Model, or Target..."
            className="w-full pl-9 pr-4 py-2 rounded-xl bg-slate-50 border border-slate-200 text-xs font-mono focus:bg-white focus:outline-hidden focus:ring-2 focus:ring-indigo-600"
          />
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          <Filter className="w-4 h-4 text-slate-400" />
          <div className="flex bg-slate-100 p-0.5 rounded-xl text-xs font-semibold">
            <button
              onClick={() => setFilterVerdict('ALL')}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                filterVerdict === 'ALL' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600'
              }`}
            >
              All ({jobs.length})
            </button>
            <button
              onClick={() => setFilterVerdict('VERIFIED')}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                filterVerdict === 'VERIFIED' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600'
              }`}
            >
              Verified
            </button>
          </div>
        </div>
      </div>

      {/* Jobs Table */}
      <div className="rounded-2xl bg-white border border-slate-200/80 shadow-xs overflow-hidden">
        {filteredJobs.length === 0 ? (
          <div className="p-12 text-center flex flex-col items-center justify-center">
            <Layers className="w-10 h-10 text-slate-300 mb-2" />
            <h3 className="text-sm font-bold text-slate-700">No Matching Jobs Found</h3>
            <p className="text-xs text-slate-400 mt-1">Try adjusting your search query or filters.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50/80 border-b border-slate-200 text-[11px] font-mono text-slate-400 uppercase tracking-wider">
                <tr>
                  <th className="py-3 px-4 font-semibold">Job ID</th>
                  <th className="py-3 px-4 font-semibold">Model & Dataset</th>
                  <th className="py-3 px-4 font-semibold">Target & Profile</th>
                  <th className="py-3 px-4 font-semibold">Accuracy</th>
                  <th className="py-3 px-4 font-semibold">Compression</th>
                  <th className="py-3 px-4 font-semibold">Latency</th>
                  <th className="py-3 px-4 font-semibold">Verdict</th>
                  <th className="py-3 px-4 font-semibold text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {filteredJobs.map((j) => {
                  const isVerified = j.verdict === 'VERIFIED';
                  return (
                    <tr
                      key={j.job_id}
                      onClick={() => {
                        onSelectJob(j.job_id);
                        onNavigate('results', j.job_id);
                      }}
                      className="hover:bg-indigo-50/30 transition-colors cursor-pointer"
                    >
                      <td className="py-3.5 px-4 font-mono font-bold text-slate-900">
                        {j.job_id}
                      </td>
                      <td className="py-3.5 px-4">
                        <div className="font-semibold text-slate-800">{j.model_name}</div>
                        <div className="text-[10px] text-slate-400">{j.dataset_name}</div>
                      </td>
                      <td className="py-3.5 px-4">
                        <div className="font-mono text-slate-700">{j.target_hardware}</div>
                        <div className="text-[10px] text-slate-400">{j.optimization_profile}</div>
                      </td>
                      <td className="py-3.5 px-4 font-mono">
                        <div className="font-bold text-slate-800">
                          {j.final_accuracy !== null ? `${(j.final_accuracy * 100).toFixed(2)}%` : 'N/A'}
                        </div>
                        <div className="text-[10px] text-emerald-600">
                          {j.accuracy_loss_pp !== null ? `-${j.accuracy_loss_pp.toFixed(2)} pp` : ''}
                        </div>
                      </td>
                      <td className="py-3.5 px-4 font-mono">
                        <div className="font-bold text-indigo-600">
                          {j.size_reduction_percent !== null ? `-${j.size_reduction_percent.toFixed(1)}%` : 'N/A'}
                        </div>
                        <div className="text-[10px] text-slate-400">
                          {j.optimized_size_bytes ? `${(j.optimized_size_bytes / (1024 * 1024)).toFixed(1)} MB` : ''}
                        </div>
                      </td>
                      <td className="py-3.5 px-4 font-mono">
                        <div className="font-bold text-slate-800">
                          {j.optimized_latency_ms !== null ? `${j.optimized_latency_ms.toFixed(1)} ms` : 'N/A'}
                        </div>
                        <div className="text-[10px] text-emerald-600">
                          {j.latency_change_percent !== null ? `+${j.latency_change_percent.toFixed(1)}%` : ''}
                        </div>
                      </td>
                      <td className="py-3.5 px-4">
                        <span
                          className={`px-2.5 py-1 rounded-md text-[10px] font-bold font-mono border ${
                            isVerified
                              ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                              : 'bg-amber-50 text-amber-800 border-amber-200'
                          }`}
                        >
                          {j.verdict}
                        </span>
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelectJob(j.job_id);
                            onNavigate('results', j.job_id);
                          }}
                          className="px-3 py-1 rounded-lg bg-slate-100 hover:bg-indigo-600 hover:text-white text-slate-700 text-xs font-semibold transition-all inline-flex items-center gap-1"
                        >
                          <Eye className="w-3.5 h-3.5" />
                          <span>Inspect</span>
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default JobsPage;
