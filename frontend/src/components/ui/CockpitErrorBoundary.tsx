import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertOctagon, RotateCcw } from 'lucide-react';

interface Props {
  jobId?: string | null;
  stage?: string | null;
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class CockpitErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log clearly to console so it is visible in dev/test tools
    console.error('CockpitErrorBoundary caught an unhandled rendering error:', error, errorInfo);
    this.setState({ errorInfo });
  }

  public handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="p-8 rounded-3xl bg-white border-2 border-rose-200 shadow-xl max-w-4xl mx-auto my-8 text-slate-800">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-2xl bg-rose-100 flex items-center justify-center text-rose-600 flex-shrink-0">
              <AlertOctagon className="w-6 h-6" />
            </div>
            <div className="flex-1">
              <div className="flex items-center justify-between pb-2 border-b border-rose-100 mb-3">
                <span className="text-xs font-mono font-bold uppercase tracking-wider text-rose-600 bg-rose-50 px-2.5 py-1 rounded-md border border-rose-200">
                  UI RENDER ERROR
                </span>
                <span className="text-xs font-mono text-slate-400">
                  Job: {this.props.jobId || 'N/A'} | Stage: {this.props.stage || 'EVALUATING'}
                </span>
              </div>
              <h2 className="text-lg font-bold text-slate-900 mb-1">
                Autonomous Cockpit Render Interrupted
              </h2>
              <p className="text-xs text-slate-600 mb-4 leading-relaxed">
                An unhandled rendering exception occurred while formatting evaluation data. The underlying engine execution remains unaffected.
              </p>
              {this.state.error && (
                <div className="p-3 bg-slate-900 text-rose-300 font-mono text-xs rounded-xl overflow-x-auto mb-4 border border-slate-800">
                  {this.state.error.name}: {this.state.error.message}
                </div>
              )}
              <div className="flex items-center gap-3">
                <button
                  onClick={this.handleReset}
                  className="px-4 py-2 rounded-xl bg-rose-600 hover:bg-rose-700 text-white text-xs font-bold shadow-xs transition flex items-center gap-2 cursor-pointer"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>Retry Rendering</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default CockpitErrorBoundary;
