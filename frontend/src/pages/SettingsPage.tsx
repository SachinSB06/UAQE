import React from 'react';
import {
  Settings,
  Cpu,
  HardDrive,
  CheckCircle2,
  Server,
  Zap,
  Shield,
  Layers,
  Code2,
} from 'lucide-react';
import { SystemStatus, TargetProfile } from '../types/api';
import SourceBadge from '../components/ui/SourceBadge';

interface SettingsPageProps {
  systemStatus: SystemStatus | null;
}

export const SettingsPage: React.FC<SettingsPageProps> = ({ systemStatus }) => {
  const host = systemStatus?.host_environment;
  const targets: TargetProfile[] = systemStatus?.supported_targets || [];

  return (
    <div className="flex flex-col gap-8 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex flex-col gap-1 pb-4 border-b border-slate-200">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            Engine Configuration & Hardware Registry
          </h1>
          <SourceBadge source="DETECTED" origin="Runtime Target Registry" />
        </div>
        <p className="text-sm text-slate-500">
          Hardware profiles, execution providers, and backend engine status
        </p>
      </div>

      {/* Backend Engine Health */}
      <div className="p-6 rounded-2xl bg-white border border-slate-200/80 shadow-xs flex flex-col gap-4">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <Server className="w-5 h-5 text-indigo-600" />
            <h3 className="text-sm font-bold text-slate-900">UAQE Core Engine Status</h3>
          </div>
          <span className="px-2.5 py-1 rounded-full text-xs font-mono font-bold bg-emerald-50 text-emerald-800 border border-emerald-200 flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>OPERATIONAL (v{systemStatus?.version || '2.0.0'})</span>
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <div className="p-3 rounded-xl bg-slate-50 border border-slate-100 font-mono">
            <span className="text-[10px] text-slate-400 block uppercase">OS Kernel</span>
            <strong className="text-slate-800">{host?.os} {host?.os_release}</strong>
          </div>

          <div className="p-3 rounded-xl bg-slate-50 border border-slate-100 font-mono">
            <span className="text-[10px] text-slate-400 block uppercase">Python Runtime</span>
            <strong className="text-slate-800">{host?.python_version || '3.11.9'}</strong>
          </div>

          <div className="p-3 rounded-xl bg-slate-50 border border-slate-100 font-mono">
            <span className="text-[10px] text-slate-400 block uppercase">Logical Cores</span>
            <strong className="text-slate-800">{host?.cpu_count_logical} Cores</strong>
          </div>

          <div className="p-3 rounded-xl bg-slate-50 border border-slate-100 font-mono">
            <span className="text-[10px] text-slate-400 block uppercase">Host Memory</span>
            <strong className="text-slate-800">{((host?.total_ram_mb || 16384) / 1024).toFixed(1)} GB</strong>
          </div>
        </div>
      </div>

      {/* Target Hardware Profiles Registry */}
      <div className="p-6 rounded-2xl bg-white border border-slate-200/80 shadow-xs flex flex-col gap-4">
        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-amber-500" />
            <div>
              <h3 className="text-sm font-bold text-slate-900">Supported Target Hardware Profiles</h3>
              <p className="text-xs text-slate-500">Registry of embedded, mobile, and microcontroller target deployment specs</p>
            </div>
          </div>
          <span className="text-xs font-mono text-slate-400">{targets.length} Registered Targets</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {targets.map((tgt) => (
            <div key={tgt.target_id} className="p-4 rounded-xl bg-slate-50/70 border border-slate-200 flex flex-col justify-between gap-3">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <h4 className="text-xs font-bold text-slate-900">{tgt.name}</h4>
                  <span className="px-2 py-0.5 rounded-md text-[9px] font-mono font-bold bg-indigo-50 text-indigo-700 border border-indigo-100">
                    {tgt.hardware_class}
                  </span>
                </div>
                <p className="text-[11px] text-slate-500">{tgt.description}</p>
              </div>

              <div className="pt-2 border-t border-slate-200/60 flex flex-wrap items-center justify-between gap-2 text-[10px] font-mono">
                <span className="text-slate-600">
                  Export: <strong>{tgt.preferred_export_format}</strong>
                </span>
                <span className="text-slate-600">
                  Precisions: <strong>{tgt.supported_precisions.join(', ')}</strong>
                </span>
                <span className={tgt.is_physical_measurement ? 'text-emerald-700 font-bold' : 'text-slate-400'}>
                  {tgt.is_physical_measurement ? 'Physical Dev Ready' : 'Host Emulated'}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default SettingsPage;
