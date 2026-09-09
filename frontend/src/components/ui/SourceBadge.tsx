import React from 'react';
import { ProvenanceLabel } from '../../types/api';

interface SourceBadgeProps {
  source?: ProvenanceLabel | string;
  origin?: string;
  size?: 'sm' | 'md';
  className?: string;
}

export const SourceBadge: React.FC<SourceBadgeProps> = ({
  source = 'NOT_AVAILABLE',
  origin,
  size = 'sm',
  className = '',
}) => {
  const getBadgeStyle = (label: string) => {
    switch (label.toUpperCase()) {
      case 'DETECTED':
        return 'bg-sky-50 text-sky-700 border-sky-200';
      case 'CONFIGURED':
        return 'bg-indigo-50 text-indigo-700 border-indigo-200';
      case 'MEASURED':
        return 'bg-emerald-50 text-emerald-700 border-emerald-200';
      case 'CALCULATED':
        return 'bg-purple-50 text-purple-700 border-purple-200';
      case 'POLICY':
        return 'bg-amber-50 text-amber-700 border-amber-200';
      case 'PENDING':
        return 'bg-slate-100 text-slate-600 border-slate-300';
      default:
        return 'bg-zinc-100 text-zinc-500 border-zinc-200';
    }
  };

  const sizeStyle = size === 'sm' ? 'text-[10px] px-1.5 py-0.5' : 'text-xs px-2 py-1';

  return (
    <span
      className={`inline-flex items-center font-mono font-semibold uppercase tracking-wider rounded border ${sizeStyle} ${getBadgeStyle(
        source
      )} ${className}`}
      title={origin ? `Origin: ${origin}` : `Data Provenance: ${source}`}
    >
      {source}
    </span>
  );
};

export default SourceBadge;
