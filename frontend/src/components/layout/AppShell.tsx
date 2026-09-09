import React from 'react';
import {
  Cpu,
  PlayCircle,
  Activity,
  History,
  Lightbulb,
  FileCheck,
  Settings,
  Tv,
  Terminal,
  ShieldCheck,
  Sparkles,
  ChevronRight,
  Layers,
  Plus,
  GitCompare,
} from 'lucide-react';
import { SystemStatus } from '../../types/api';

interface AppShellProps {
  children: React.ReactNode;
  activeTab: string;
  onTabChange: (tab: string) => void;
  onStartNewOptimization: () => void;
  isPresentationMode: boolean;
  onTogglePresentationMode: () => void;
  systemStatus: SystemStatus | null;
  selectedJobId?: string | null;
  runningJobId?: string | null;
  viewingJobId?: string | null;
}

export const AppShell: React.FC<AppShellProps> = ({
  children,
  activeTab,
  onTabChange,
  onStartNewOptimization,
  isPresentationMode,
  onTogglePresentationMode,
  systemStatus,
  selectedJobId,
  runningJobId,
  viewingJobId,
}) => {
  const currentViewingJob = viewingJobId || selectedJobId;
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Activity },
    { id: 'new', label: 'New Optimization', icon: PlayCircle },
    { id: 'cockpit', label: 'Autonomous Cockpit', icon: Cpu },
    { id: 'intelligence', label: 'Optimization Intelligence', icon: Lightbulb },
    { id: 'results', label: 'Results & Deployment', icon: FileCheck },
    { id: 'compare', label: 'Compare Jobs', icon: GitCompare },
    { id: 'jobs', label: 'Job History', icon: History },
    { id: 'settings', label: 'System Settings', icon: Settings },
  ];

  if (isPresentationMode) {
    return (
      <div className="min-h-screen bg-slate-950 text-white flex flex-col antialiased">
        {/* Cinematic Presentation Mode Header */}
        <header className="h-16 px-8 border-b border-slate-800 bg-slate-900/80 backdrop-blur-md flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/20 border border-indigo-400/40 flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-indigo-400" />
            </div>
            <div>
              <span className="font-bold tracking-wider text-base text-white">UAQE</span>
              <span className="text-xs text-slate-400 ml-2 font-mono uppercase">Presentation Mode</span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={onTogglePresentationMode}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
            >
              <Terminal className="w-3.5 h-3.5" />
              <span>Switch to Technical Mode</span>
            </button>
          </div>
        </header>

        <main className="flex-1 p-8 max-w-7xl mx-auto w-full">
          {children}
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#FAFAFA] text-slate-900 flex flex-col font-sans">
      {/* Top Luxury Navigation Header */}
      <header className="h-16 px-6 border-b border-slate-200/80 bg-white/90 backdrop-blur-md sticky top-0 z-30 flex items-center justify-between shadow-2xs">
        <div className="flex items-center gap-4">
          <div
            onClick={() => onTabChange('dashboard')}
            className="flex items-center gap-3 cursor-pointer group"
          >
            <div className="w-9 h-9 rounded-xl bg-slate-900 flex items-center justify-center text-white shadow-subtle group-hover:bg-indigo-600 transition-colors duration-200">
              <Cpu className="w-5 h-5 text-indigo-200" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-base tracking-tight text-slate-900">UAQE</span>
                <span className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 border border-slate-200">
                  v1.0.0
                </span>
              </div>
              <p className="text-[11px] text-slate-500 font-medium tracking-wide">
                Universal AI Quantization Engine
              </p>
            </div>
          </div>

          {/* Job Badges: Clearly distinguish Viewing Job vs Running Job */}
          <div className="hidden md:flex items-center gap-3 ml-4 pl-4 border-l border-slate-200 text-xs font-mono">
            {currentViewingJob && (
              <div className="flex items-center gap-1.5 text-slate-600">
                <span className="text-slate-400 font-sans text-[11px]">Viewing Job:</span>
                <span className="font-semibold text-slate-800 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
                  {currentViewingJob}
                </span>
              </div>
            )}
            {runningJobId && (
              <div className="flex items-center gap-1.5 text-indigo-700 bg-indigo-50 border border-indigo-200 px-2 py-0.5 rounded">
                <span className="w-2 h-2 rounded-full bg-indigo-600 animate-pulse" />
                <span className="text-indigo-500 font-sans text-[11px] font-medium">Running Job:</span>
                <span className="font-bold text-indigo-900">
                  {runningJobId}
                </span>
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center gap-3 sm:gap-4">
          {/* Target status pill */}
          <div className="hidden lg:flex items-center gap-2 px-3 py-1 rounded-full bg-slate-50 border border-slate-200 text-xs text-slate-600">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="font-medium">System Ready</span>
            <span className="text-slate-300">|</span>
            <span className="font-mono text-slate-500">Raspberry Pi 5</span>
          </div>

          {/* Mode Switcher */}
          <button
            onClick={onTogglePresentationMode}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-700 bg-white hover:bg-slate-50 border border-slate-200 shadow-2xs transition hover:border-slate-300"
            title="Switch to Cinematic Presentation Mode"
          >
            <Tv className="w-3.5 h-3.5 text-indigo-600" />
            <span className="hidden sm:inline">Presentation Mode</span>
          </button>
        </div>
      </header>

      <div className="flex-1 flex">
        {/* Left Sidebar Navigation */}
        <aside className="w-64 border-r border-slate-200/80 bg-white/70 backdrop-blur-md flex flex-col p-4 space-y-1">
          {/* Prominent New Optimization Action Button */}
          <div className="mb-3 px-1">
            <button
              onClick={onStartNewOptimization}
              className="w-full py-2.5 px-3.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800 text-white text-xs font-bold shadow-sm transition-all flex items-center justify-center gap-2 group cursor-pointer"
            >
              <Plus className="w-4 h-4 text-indigo-200 group-hover:rotate-90 transition-transform duration-200" />
              <span>+ NEW OPTIMIZATION</span>
            </button>
          </div>

          <div className="px-3 py-2 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
            Workspace
          </div>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => {
                  if (item.id === 'new') {
                    onStartNewOptimization();
                  } else {
                    onTabChange(item.id);
                  }
                }}
                className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-xs font-medium transition-all duration-150 cursor-pointer ${
                  isActive
                    ? 'bg-slate-900 text-white shadow-subtle'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100/80'
                }`}
              >
                <div className="flex items-center gap-3">
                  <Icon className={`w-4 h-4 ${isActive ? 'text-indigo-300' : 'text-slate-400'}`} />
                  <span>{item.label}</span>
                </div>
                {isActive && <ChevronRight className="w-3.5 h-3.5 text-slate-400" />}
              </button>
            );
          })}

          <div className="pt-6 mt-auto border-t border-slate-100">
            <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/60 text-xs">
              <div className="flex items-center gap-2 text-slate-800 font-semibold mb-1">
                <ShieldCheck className="w-4 h-4 text-emerald-600" />
                <span>Zero-Fake Policy</span>
              </div>
              <p className="text-[11px] text-slate-500 leading-relaxed">
                All benchmark & telemetry metrics originate directly from genuine Python execution.
              </p>
            </div>
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 p-6 lg:p-8 max-w-7xl mx-auto w-full overflow-x-hidden">
          {children}
        </main>
      </div>
    </div>
  );
};

export default AppShell;
