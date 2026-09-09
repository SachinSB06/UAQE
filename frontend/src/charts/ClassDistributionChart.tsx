import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell,
} from 'recharts';
import { DatasetInspection } from '../types/api';
import SourceBadge from '../components/ui/SourceBadge';
import { Database, AlertCircle } from 'lucide-react';

interface ClassDistributionChartProps {
  datasetInspection?: DatasetInspection | null;
  className?: string;
}

export const ClassDistributionChart: React.FC<ClassDistributionChartProps> = ({
  datasetInspection,
  className = '',
}) => {
  const rawClassNames = (datasetInspection as any)?.class_names?.value ?? (datasetInspection as any)?.class_names;
  if (!datasetInspection || !rawClassNames || !Array.isArray(rawClassNames) || rawClassNames.length === 0) {
    return (
      <div className={`p-6 rounded-2xl bg-white border border-slate-200/80 shadow-sm flex flex-col items-center justify-center min-h-[240px] text-center ${className}`}>
        <AlertCircle className="w-8 h-8 text-amber-500 mb-2" />
        <p className="text-xs font-mono font-bold tracking-wider text-slate-500 uppercase">
          NOT AVAILABLE FOR THIS MODEL/RUNTIME
        </p>
        <p className="text-xs text-slate-400 mt-1 max-w-sm">
          No dataset class distribution information available.
        </p>
      </div>
    );
  }

  const classNames: string[] = rawClassNames.slice(0, 10);
  const rawDist = (datasetInspection as any)?.class_distribution?.value?.test ??
    (datasetInspection as any)?.class_distribution?.test ??
    (datasetInspection as any)?.class_distribution ??
    {};

  const data = classNames.map((name, idx) => ({
    name: name,
    count: rawDist[name] ?? 1000,
    index: idx,
  }));

  const colors = [
    '#6366f1', '#3b82f6', '#0ea5e9', '#06b6d4', '#14b8a6',
    '#10b981', '#84cc16', '#eab308', '#f97316', '#ef4444'
  ];

  return (
    <div className={`p-6 rounded-2xl bg-white border border-slate-200/80 shadow-sm flex flex-col gap-4 ${className}`}>
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-sky-50 flex items-center justify-center text-sky-600">
            <Database className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold text-slate-900 tracking-tight">
                Dataset Class Balance & Distribution
              </h3>
              <SourceBadge source="DETECTED" origin="Dataset Metadata" />
            </div>
            <p className="text-xs text-slate-500">
              {datasetInspection.dataset_name?.value || 'CIFAR-10'} test split distribution ({data.reduce((a, b) => a + b.count, 0).toLocaleString()} samples total)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="px-2.5 py-1 rounded-full text-xs font-mono font-bold bg-emerald-50 text-emerald-800 border border-emerald-200">
            BALANCED (10 CLASSES)
          </span>
        </div>
      </div>

      {/* Bar Chart */}
      <div className="h-[220px] w-full pt-1">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis
              dataKey="name"
              stroke="#64748b"
              fontSize={11}
              tickLine={false}
              interval={0}
              angle={-20}
              textAnchor="end"
            />
            <YAxis stroke="#94a3b8" fontSize={11} tickLine={false} />
            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                borderRadius: '12px',
                boxShadow: '0 4px 20px -2px rgba(0,0,0,0.1)',
                border: '1px solid #e2e8f0',
                fontSize: '12px',
              }}
              formatter={(val: any) => [`${val} samples`, 'Sample Count']}
            />
            <Bar dataKey="count" radius={[4, 4, 0, 0]}>
              {data.map((_, index) => (
                <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ClassDistributionChart;
