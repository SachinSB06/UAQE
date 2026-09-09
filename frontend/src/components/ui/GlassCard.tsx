import React from 'react';

interface GlassCardProps {
  children: React.ReactNode;
  className?: string;
  elevated?: boolean;
  hoverEffect?: boolean;
  onClick?: () => void;
}

export const GlassCard: React.FC<GlassCardProps> = ({
  children,
  className = '',
  elevated = false,
  hoverEffect = false,
  onClick,
}) => {
  const baseClass = elevated ? 'glass-panel-elevated' : 'glass-panel';
  const hoverClass = hoverEffect
    ? 'transition-all duration-200 hover:shadow-luxury hover:border-slate-300 hover:-translate-y-0.5'
    : '';
  const clickClass = onClick ? 'cursor-pointer' : '';

  return (
    <div
      className={`rounded-2xl p-6 ${baseClass} ${hoverClass} ${clickClass} ${className}`}
      onClick={onClick}
    >
      {children}
    </div>
  );
};

export default GlassCard;
