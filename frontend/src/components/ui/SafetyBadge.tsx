import React from 'react';
import { ShieldCheck, AlertTriangle, AlertOctagon } from 'lucide-react';
import { safeNumber, safeString } from '../../utils/candidateNormalizer';

interface SafetyBadgeProps {
  classification: 'EXCELLENT' | 'ACCEPTABLE' | 'CRITICAL' | string;
  lossPp?: number | null;
  lossPP?: number | null;
  size?: 'sm' | 'md' | 'lg';
  showIcon?: boolean;
  className?: string;
}

export const SafetyBadge: React.FC<SafetyBadgeProps> = ({
  classification,
  lossPp,
  lossPP,
  size = 'md',
  showIcon = true,
  className = '',
}) => {
  const rawLoss = lossPP !== undefined ? lossPP : lossPp;
  const loss = safeNumber(rawLoss);
  const norm = safeString(classification, 'CRITICAL').toUpperCase();

  const getConfig = () => {
    switch (norm) {
      case 'EXCELLENT':
        return {
          bg: 'bg-emerald-50 text-emerald-700 border-emerald-200',
          icon: ShieldCheck,
          label: 'EXCELLENT',
          desc: 'Loss ≤ 1.0 pp',
        };
      case 'ACCEPTABLE':
        return {
          bg: 'bg-amber-50 text-amber-700 border-amber-200',
          icon: AlertTriangle,
          label: 'ACCEPTABLE',
          desc: '1.0 < Loss ≤ 4.0 pp',
        };
      case 'INVALID_BASELINE':
        return {
          bg: 'bg-rose-100 text-rose-800 border-rose-300',
          icon: AlertOctagon,
          label: 'INVALID BASELINE',
          desc: 'Baseline < Threshold',
        };
      case 'CRITICAL':
      default:
        return {
          bg: 'bg-rose-50 text-rose-700 border-rose-200',
          icon: AlertOctagon,
          label: 'CRITICAL',
          desc: 'Loss > 4.0 pp',
        };
    }
  };

  const { bg, icon: Icon, label } = getConfig();

  const sizeClasses =
    size === 'sm'
      ? 'text-[11px] px-2 py-0.5 gap-1'
      : size === 'lg'
      ? 'text-sm px-3 py-1.5 gap-2 font-semibold'
      : 'text-xs px-2.5 py-1 gap-1.5 font-medium';

  return (
    <span
      className={`inline-flex items-center rounded-full border ${bg} ${sizeClasses} ${className}`}
      title={`Accuracy Safety Classification: ${label}${loss !== undefined && loss !== null ? ` (${loss.toFixed(2)} pp loss)` : ''}`}
    >
      {showIcon && <Icon className={size === 'sm' ? 'w-3 h-3' : size === 'lg' ? 'w-4 h-4' : 'w-3.5 h-3.5'} />}
      <span>{label}</span>
      {loss !== undefined && loss !== null && (
        <span className="font-mono opacity-85">({loss >= 0 ? `-${loss.toFixed(2)}` : `+${Math.abs(loss).toFixed(2)}`} pp)</span>
      )}
    </span>
  );
};

export default SafetyBadge;
