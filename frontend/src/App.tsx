import React, { useState, useEffect, useRef } from 'react';
import confetti from 'canvas-confetti';
import { apiService } from './services/api';
import {
  SystemStatus,
  JobSummary,
  JobDetail,
  TelemetryResponse,
  OptimizationEvent,
  ModelInspection,
  DatasetInspection,
  OptimizationPlan,
} from './types/api';
import { AppShell } from './components/layout/AppShell';
import DashboardPage from './pages/DashboardPage';
import NewOptimizationPage from './pages/NewOptimizationPage';
import InspectionPlanPage from './pages/InspectionPlanPage';
import AutonomousCockpitPage from './pages/AutonomousCockpitPage';
import OptimizationIntelligencePage from './pages/OptimizationIntelligencePage';
import ResultsPage from './pages/ResultsPage';
import JobsPage from './pages/JobsPage';
import SettingsPage from './pages/SettingsPage';
import JobComparePage from './pages/JobComparePage';
import { TelemetryPoint } from './charts/TaskManagerGraph';
import { AlertTriangle, Play, X } from 'lucide-react';

export const App: React.FC = () => {
  // Navigation & View Mode State
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [isPresentationMode, setIsPresentationMode] = useState<boolean>(false);

  // Global Engine State
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [recentJobs, setRecentJobs] = useState<JobSummary[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [runningJobId, setRunningJobId] = useState<string | null>(null);
  const [jobDetail, setJobDetail] = useState<JobDetail | null>(null);
  const [telemetry, setTelemetry] = useState<TelemetryResponse | null>(null);

  // Live SSE stream state
  const [events, setEvents] = useState<OptimizationEvent[]>([]);
  const [liveTelemetrySamples, setLiveTelemetrySamples] = useState<TelemetryPoint[]>([]);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const sseCleanupRef = useRef<(() => void) | null>(null);

  // Pre-optimization inspection state
  const [modelInspection, setModelInspection] = useState<ModelInspection | null>(null);
  const [datasetInspection, setDatasetInspection] = useState<DatasetInspection | null>(null);
  const [optimizationPlan, setOptimizationPlan] = useState<OptimizationPlan | null>(null);
  const [lastConfig, setLastConfig] = useState<any>(null);

  // Confirmation modal state for running jobs
  const [showConfirmNewJobModal, setShowConfirmNewJobModal] = useState<boolean>(false);

  // Synchronize route with URL path
  const syncRouteFromUrl = () => {
    const pathname = window.location.pathname;
    const resultsMatch = pathname.match(/^\/results(?:\/([^/]+))?/);
    if (resultsMatch) {
      const jobId = resultsMatch[1];
      setActiveTab('results');
      if (jobId) {
        setSelectedJobId(jobId);
        apiService.getJobDetail(jobId).then(setJobDetail).catch(console.error);
        apiService.getTelemetry(jobId).then(setTelemetry).catch(() => setTelemetry(null));
      }
      return;
    }
    const compareMatch = pathname.match(/^\/compare/);
    if (compareMatch) {
      setActiveTab('compare');
      return;
    }
    const cockpitMatch = pathname.match(/^\/cockpit/);
    if (cockpitMatch) {
      setActiveTab('cockpit');
      return;
    }
    const newMatch = pathname.match(/^\/new/);
    if (newMatch) {
      setActiveTab('new');
      return;
    }
    const jobsMatch = pathname.match(/^\/jobs/);
    if (jobsMatch) {
      setActiveTab('jobs');
      return;
    }
  };

  // Load system status, recent jobs, and synchronize with URL on mount
  useEffect(() => {
    loadSystemStatus();
    loadJobs();
    syncRouteFromUrl();

    window.addEventListener('popstate', syncRouteFromUrl);
    return () => window.removeEventListener('popstate', syncRouteFromUrl);
  }, []);

  const loadSystemStatus = async () => {
    try {
      const status = await apiService.getSystemStatus();
      setSystemStatus(status);
    } catch (err) {
      console.warn('System status not yet reachable (backend starting up):', err);
    }
  };

  const loadJobs = async () => {
    try {
      const jobs = await apiService.getJobs();
      setRecentJobs(jobs);
    } catch (err) {
      console.warn('Jobs history not yet reachable:', err);
    }
  };

  /**
   * Complete Frontend State Reset: Starts from zero for a brand new optimization run.
   */
  const handleStartNewOptimization = (force: boolean = false) => {
    if (isStreaming && !force) {
      setShowConfirmNewJobModal(true);
      return;
    }

    // Close any active SSE connection
    if (sseCleanupRef.current) {
      sseCleanupRef.current();
      sseCleanupRef.current = null;
    }

    // Purge all active job and optimization state
    setShowConfirmNewJobModal(false);
    setSelectedJobId(null);
    setJobDetail(null);
    setTelemetry(null);
    setEvents([]);
    setIsStreaming(false);
    setIsProcessing(false);
    setModelInspection(null);
    setDatasetInspection(null);
    setOptimizationPlan(null);
    setLastConfig(null);

    // Route cleanly to New Optimization
    setActiveTab('new');
    window.history.pushState(null, '', '/new');
  };

  const handleSelectJob = async (jobId: string, pushUrl: boolean = true) => {
    // If a live stream was running, close it when explicitly opening a historical job
    if (sseCleanupRef.current) {
      sseCleanupRef.current();
      sseCleanupRef.current = null;
    }
    setIsStreaming(false);

    setSelectedJobId(jobId);
    setLiveTelemetrySamples([]);
    setEvents([]);
    if (pushUrl && window.location.pathname !== `/results/${jobId}`) {
      window.history.pushState(null, '', `/results/${jobId}`);
    }
    try {
      const detail = await apiService.getJobDetail(jobId);
      setJobDetail(detail);

      // Load telemetry if available
      try {
        const telem = await apiService.getTelemetry(jobId);
        setTelemetry(telem);
      } catch (e) {
        setTelemetry(null);
      }

      // If verified, fire celebration confetti!
      if (detail.verdict === 'VERIFIED') {
        confetti({
          particleCount: 80,
          spread: 70,
          origin: { y: 0.6 },
          colors: ['#10b981', '#6366f1', '#3b82f6', '#f59e0b'],
        });
      }
    } catch (err) {
      console.error('Failed to load job details:', err);
    }
  };

  const handleAnalyze = async (config: {
    modelUploadId?: string;
    modelId?: string;
    modelPath?: string;
    datasetUploadId?: string;
    datasetId?: string;
    datasetPath?: string;
    targetHardware: string;
    profile: string;
    calibSamples: number;
    testSamples: number;
  }) => {
    setIsProcessing(true);
    setLastConfig(config);
    try {
      const resp = await apiService.analyzeModel({
        model_upload_id: config.modelUploadId,
        model_id: config.modelId,
        model_path: config.modelPath,
        dataset_upload_id: config.datasetUploadId,
        dataset_id: config.datasetId,
        dataset_path: config.datasetPath,
        target: config.targetHardware,
        profile: config.profile,
        calib_samples: config.calibSamples,
        test_samples: config.testSamples,
      });

      setModelInspection(resp.model_inspection);
      setDatasetInspection(resp.dataset_inspection);
      setOptimizationPlan(resp.optimization_plan);

      setActiveTab('inspect');
    } catch (err: any) {
      alert(`Model Analysis Error: ${err.message || err}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleApproveAndLaunch = async () => {
    if (!lastConfig) return;
    setIsProcessing(true);

    // Close any previous SSE stream
    if (sseCleanupRef.current) {
      sseCleanupRef.current();
      sseCleanupRef.current = null;
    }

    try {
      const resp = await apiService.launchOptimization({
        model_upload_id: lastConfig.modelUploadId,
        model_id: lastConfig.modelId,
        model_path: lastConfig.modelPath,
        dataset_upload_id: lastConfig.datasetUploadId,
        dataset_id: lastConfig.datasetId,
        dataset_path: lastConfig.datasetPath,
        target: lastConfig.targetHardware,
        profile: lastConfig.profile,
        calib_samples: lastConfig.calibSamples,
        test_samples: lastConfig.testSamples,
        auto_approve: true,
      });

      // Set running and viewing job states for fresh execution
      setRunningJobId(resp.job_id);
      setSelectedJobId(resp.job_id);
      setJobDetail(null);
      setTelemetry(null);
      setLiveTelemetrySamples([]);
      setActiveTab('cockpit');
      window.history.pushState(null, '', '/cockpit');
      setEvents([
        {
          type: 'OPTIMIZATION_STARTED',
          stage: 'INITIALIZING',
          job_id: resp.job_id,
          message: `Autonomous optimization initialized for Job ${resp.job_id}`,
          timestamp: Date.now() / 1000,
        },
      ]);
      setIsStreaming(true);

      // Start fresh SSE stream listener
      const cleanupStream = apiService.streamJobEvents(
        resp.job_id,
        async (event) => {
          // Scope SSE events strictly to the running job
          if (event.job_id && event.job_id !== resp.job_id) {
            return;
          }
          setEvents((prev) => [...prev, event]);
          if (event.type === 'telemetry') {
            setLiveTelemetrySamples((prev) => {
              const next = [...prev, event as any];
              return next.length > 120 ? next.slice(next.length - 120) : next;
            });
          }
          const evtType = (event.type || '').toUpperCase();
          if (evtType === 'OPTIMIZATION_COMPLETED' || evtType === 'FINAL_RESULT' || evtType === 'COMPLETE') {
            setIsStreaming(false);
            setRunningJobId(null);
            if (sseCleanupRef.current) {
              sseCleanupRef.current();
              sseCleanupRef.current = null;
            }
            await handleSelectJob(resp.job_id, true);
            await loadJobs();
            setActiveTab('results');
            window.history.pushState(null, '', `/results/${resp.job_id}`);
          } else if (evtType === 'ERROR' || evtType === 'FAILED') {
            setIsStreaming(false);
            setRunningJobId(null);
            if (sseCleanupRef.current) {
              sseCleanupRef.current();
              sseCleanupRef.current = null;
            }
            await handleSelectJob(resp.job_id, true);
            await loadJobs();
          }
        },
        (err) => {
          console.warn('SSE Stream terminated or encountered error:', err);
          setIsStreaming(false);
          setRunningJobId(null);
          if (sseCleanupRef.current) {
            sseCleanupRef.current();
            sseCleanupRef.current = null;
          }
          apiService.getJobDetail(resp.job_id).then(setJobDetail).catch(() => {});
          loadJobs();
        }
      );
      sseCleanupRef.current = cleanupStream;
    } catch (err: any) {
      alert(`Optimization Launch Error: ${err.message || err}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDirectOptimize = async (config: {
    modelUploadId?: string;
    modelId?: string;
    modelPath?: string;
    datasetUploadId?: string;
    datasetId?: string;
    datasetPath?: string;
    targetHardware: string;
    profile: string;
    calibSamples: number;
    testSamples: number;
    maxBudget: number;
  }) => {
    setLastConfig(config);
    setIsProcessing(true);

    // Close any previous SSE stream
    if (sseCleanupRef.current) {
      sseCleanupRef.current();
      sseCleanupRef.current = null;
    }

    try {
      const resp = await apiService.launchOptimization({
        model_upload_id: config.modelUploadId,
        model_id: config.modelId,
        model_path: config.modelPath,
        dataset_upload_id: config.datasetUploadId,
        dataset_id: config.datasetId,
        dataset_path: config.datasetPath,
        target: config.targetHardware,
        profile: config.profile,
        calib_samples: config.calibSamples,
        test_samples: config.testSamples,
        max_budget: config.maxBudget,
        auto_approve: true,
      });

      // Set running and viewing job states for fresh execution
      setRunningJobId(resp.job_id);
      setSelectedJobId(resp.job_id);
      setJobDetail(null);
      setTelemetry(null);
      setLiveTelemetrySamples([]);
      setActiveTab('cockpit');
      window.history.pushState(null, '', '/cockpit');
      setEvents([
        {
          type: 'OPTIMIZATION_STARTED',
          stage: 'INITIALIZING',
          job_id: resp.job_id,
          message: `Autonomous optimization job ${resp.job_id} dispatched.`,
          timestamp: Date.now() / 1000,
        },
      ]);
      setIsStreaming(true);

      const cleanupStream = apiService.streamJobEvents(
        resp.job_id,
        async (event) => {
          // Scope SSE events strictly to the running job
          if (event.job_id && event.job_id !== resp.job_id) {
            return;
          }
          setEvents((prev) => [...prev, event]);
          if (event.type === 'telemetry') {
            setLiveTelemetrySamples((prev) => {
              const next = [...prev, event as any];
              return next.length > 120 ? next.slice(next.length - 120) : next;
            });
          }
          const evtType = (event.type || '').toUpperCase();
          if (evtType === 'OPTIMIZATION_COMPLETED' || evtType === 'FINAL_RESULT' || evtType === 'COMPLETE') {
            setIsStreaming(false);
            setRunningJobId(null);
            if (sseCleanupRef.current) {
              sseCleanupRef.current();
              sseCleanupRef.current = null;
            }
            await handleSelectJob(resp.job_id, true);
            await loadJobs();
            setActiveTab('results');
            window.history.pushState(null, '', `/results/${resp.job_id}`);
          } else if (evtType === 'ERROR' || evtType === 'FAILED') {
            setIsStreaming(false);
            setRunningJobId(null);
            if (sseCleanupRef.current) {
              sseCleanupRef.current();
              sseCleanupRef.current = null;
            }
            await handleSelectJob(resp.job_id, true);
            await loadJobs();
          }
        },
        (err) => {
          console.warn('SSE Stream terminated or encountered error:', err);
          setIsStreaming(false);
          setRunningJobId(null);
          if (sseCleanupRef.current) {
            sseCleanupRef.current();
            sseCleanupRef.current = null;
          }
          apiService.getJobDetail(resp.job_id).then(setJobDetail).catch(() => {});
          loadJobs();
        }
      );
      sseCleanupRef.current = cleanupStream;
    } catch (err: any) {
      alert(`Optimization Launch Error: ${err.message || err}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownloadArtifact = (target: string, candidateId?: string) => {
    if (!selectedJobId) return;
    let url: string;
    if (target.startsWith('http://') || target.startsWith('https://') || target.startsWith('/api/')) {
      url = target;
    } else if (candidateId) {
      url = apiService.getCandidateDownloadUrl(selectedJobId, candidateId);
    } else {
      url = apiService.getArtifactDownloadUrl(selectedJobId, target);
    }
    window.open(url, '_blank');
  };

  const handleNavigate = (tab: string, jobId?: string) => {
    if (tab === 'new' || tab === 'new-opt') {
      handleStartNewOptimization();
      return;
    }
    if (jobId) {
      handleSelectJob(jobId, true);
    }
    setActiveTab(tab);
    if (tab === 'results') {
      const targetJob = jobId || selectedJobId;
      if (targetJob) {
        window.history.pushState(null, '', `/results/${targetJob}`);
      } else {
        window.history.pushState(null, '', '/results');
      }
    } else {
      window.history.pushState(null, '', `/${tab}`);
    }
  };

  return (
    <AppShell
      activeTab={activeTab === 'inspect' ? 'new' : activeTab}
      onTabChange={(tab) => {
        if (tab === 'new') {
          handleStartNewOptimization();
        } else {
          handleNavigate(tab);
        }
      }}
      onStartNewOptimization={() => handleStartNewOptimization()}
      isPresentationMode={isPresentationMode}
      onTogglePresentationMode={() => setIsPresentationMode(!isPresentationMode)}
      systemStatus={systemStatus}
      selectedJobId={selectedJobId}
      runningJobId={runningJobId}
      viewingJobId={selectedJobId}
    >
      {/* Confirmation Modal when trying to start a new job while an active job is running */}
      {showConfirmNewJobModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 backdrop-blur-xs p-4">
          <div className="max-w-md w-full rounded-2xl bg-white p-6 shadow-2xl border border-slate-200 animate-in fade-in zoom-in-95">
            <div className="flex items-center gap-3 text-amber-600 mb-3">
              <div className="w-10 h-10 rounded-xl bg-amber-50 flex items-center justify-center">
                <AlertTriangle className="w-5 h-5" />
              </div>
              <h3 className="text-base font-bold text-slate-900">Optimization In Progress</h3>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed mb-6">
              An autonomous optimization is currently actively executing. Starting a new optimization will disconnect from the active live stream and clear the current workspace.
            </p>
            <div className="flex items-center justify-end gap-3">
              <button
                onClick={() => setShowConfirmNewJobModal(false)}
                className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-600 hover:bg-slate-100 transition"
              >
                Cancel
              </button>
              <button
                onClick={() => handleStartNewOptimization(true)}
                className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold shadow-xs transition"
              >
                Start New Job
              </button>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'dashboard' && (
        <DashboardPage
          systemStatus={systemStatus}
          recentJobs={recentJobs}
          onNavigate={handleNavigate}
          onSelectJob={handleSelectJob}
          onStartNewOptimization={() => handleStartNewOptimization()}
        />
      )}

      {activeTab === 'new' && (
        <NewOptimizationPage
          systemStatus={systemStatus}
          onAnalyze={handleAnalyze}
          onDirectOptimize={handleDirectOptimize}
          isProcessing={isProcessing}
        />
      )}

      {activeTab === 'inspect' && (
        <InspectionPlanPage
          modelInspection={modelInspection}
          datasetInspection={datasetInspection}
          optimizationPlan={optimizationPlan}
          onApproveAndLaunch={handleApproveAndLaunch}
          isLaunching={isProcessing}
          onBack={() => setActiveTab('new')}
        />
      )}

      {activeTab === 'cockpit' && (
        <AutonomousCockpitPage
          jobDetail={jobDetail}
          events={events}
          telemetry={telemetry}
          liveTelemetrySamples={liveTelemetrySamples}
          isStreaming={isStreaming}
          onViewResults={() => setActiveTab('results')}
          onStartNewOptimization={() => handleStartNewOptimization()}
        />
      )}

      {activeTab === 'intelligence' && (
        <OptimizationIntelligencePage
          jobDetail={jobDetail}
          onStartNewOptimization={() => handleStartNewOptimization()}
        />
      )}

      {activeTab === 'results' && (
        <ResultsPage
          jobDetail={jobDetail}
          telemetry={telemetry}
          onDownloadArtifact={handleDownloadArtifact}
          onStartNewOptimization={() => handleStartNewOptimization()}
        />
      )}

      {activeTab === 'compare' && (
        <JobComparePage
          jobs={recentJobs}
          initialJobA={selectedJobId || recentJobs[0]?.job_id}
          initialJobB={recentJobs.find((j) => j.job_id !== selectedJobId)?.job_id || recentJobs[1]?.job_id}
          onSelectJob={handleSelectJob}
        />
      )}

      {activeTab === 'jobs' && (
        <JobsPage
          jobs={recentJobs}
          onSelectJob={handleSelectJob}
          onNavigate={handleNavigate}
        />
      )}

      {activeTab === 'settings' && (
        <SettingsPage systemStatus={systemStatus} />
      )}
    </AppShell>
  );
};

export default App;
