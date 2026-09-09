import React, { useState } from 'react';
import { HelpCircle, X } from 'lucide-react';
import { CandidateSummary } from '../../types/api';
import { formatNumber, safeNumber } from '../../utils/candidateNormalizer';

interface ExplainPopoverProps {
  candidate?: CandidateSummary;
  title?: string;
  reason?: string;
  evidence?: string | Record<string, any>;
  classification?: string;
  actionTaken?: string;
}

export const ExplainPopover: React.FC<ExplainPopoverProps> = ({
  candidate,
  title,
  reason,
  evidence,
  classification,
  actionTaken,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const effTitle = title || (candidate ? `Candidate Decision: ${candidate.candidate_name}` : 'Decision Audit');
  const effReason =
    reason ||
    (candidate?.rejection_reason
      ? candidate.rejection_reason
      : candidate?.is_satisfied
      ? `Candidate satisfied target loss policy with only ${formatNumber(candidate.accuracy_loss_pp, 2)} pp loss (Composite Score: ${formatNumber(candidate.composite_score, 3)})`
      : 'Candidate evaluated against multi-objective Pareto frontier.');
  const effClass = classification || candidate?.safety_classification || 'EXCELLENT';
  const effAction = actionTaken || candidate?.action_taken || (candidate?.is_satisfied ? 'Selected as Optimal Package' : 'Gated by Policy');
  const effEvidence = evidence || candidate?.artifact_metadata;

  const getHeaderColor = () => {
    switch (effClass.toUpperCase()) {
      case 'EXCELLENT':
        return 'text-emerald-700 bg-emerald-50 border-emerald-200';
      case 'ACCEPTABLE':
        return 'text-amber-700 bg-amber-50 border-amber-200';
      case 'CRITICAL':
        return 'text-rose-700 bg-rose-50 border-rose-200';
      default:
        return 'text-slate-800 bg-slate-50 border-slate-200';
    }
  };

  return (
    <div className="relative inline-block">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-semibold text-indigo-700 bg-indigo-50 hover:bg-indigo-100 border border-indigo-200 rounded-lg transition-all shadow-xs"
        title="View Autonomous Decision Evidence"
      >
        <HelpCircle className="w-3.5 h-3.5 text-indigo-600" />
        <span>Why?</span>
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-40 bg-slate-900/10 backdrop-blur-2xs"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute right-0 z-50 mt-2 w-80 sm:w-96 p-4 bg-white rounded-2xl shadow-xl border border-slate-200 text-left">
            <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-100">
              <span className="text-xs font-bold text-slate-900 tracking-wide uppercase">
                {effTitle}
              </span>
              <button
                onClick={() => setIsOpen(false)}
                className="p-1 text-slate-400 hover:text-slate-600 rounded-md hover:bg-slate-100"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-3 text-xs text-slate-700">
              <div className={`p-3 rounded-xl border text-xs leading-relaxed ${getHeaderColor()}`}>
                <p className="font-medium">{effReason}</p>
              </div>

              {effAction && (
                <div>
                  <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                    Action Taken by UAQE
                  </span>
                  <p className="text-slate-800 bg-slate-50 p-2.5 rounded-xl border border-slate-200/60 font-mono text-[11px]">
                    {effAction}
                  </p>
                </div>
              )}

              {effEvidence && (
                <div>
                  <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                    Candidate Metadata Evidence
                  </span>
                  <div className="bg-slate-900 text-slate-200 p-2.5 rounded-xl font-mono text-[10px] overflow-x-auto max-h-36">
                    {typeof effEvidence === 'string' ? effEvidence : JSON.stringify(effEvidence, null, 2)}
                  </div>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default ExplainPopover;
