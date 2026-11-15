import React from 'react';
import type { BarrenPlateauPoint, TrainingStepMetrics } from '../quantum/qcnnEngine';

interface BarrenPlateauChartProps {
  scalingData: BarrenPlateauPoint[];
  history: TrainingStepMetrics[];
}

export const BarrenPlateauChart: React.FC<BarrenPlateauChartProps> = ({
  scalingData,
  history,
}) => {
  const width = 310;
  const height = 118;
  const padL = 34;
  const padR = 14;
  const padT = 14;
  const padB = 22;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  // Log10 scale from 10^-5 to 10^0
  const minLog = -5;
  const maxLog = 0;
  const mapY = (val: number) => {
    const clamped = Math.max(1e-5, Math.min(1, val));
    const logV = Math.log10(clamped);
    const norm = (logV - minLog) / (maxLog - minLog);
    return padT + plotH * (1 - norm);
  };

  const mapX = (idx: number) => {
    if (scalingData.length <= 1) return padL + plotW / 2;
    return padL + (idx / (scalingData.length - 1)) * plotW;
  };

  const qcnnPoints = scalingData
    .map((pt, i) => `${mapX(i)},${mapY(pt.qcnnVariance)}`)
    .join(' ');
  const heaPoints = scalingData
    .map((pt, i) => `${mapX(i)},${mapY(pt.heaVariance)}`)
    .join(' ');

  return (
    <div className="space-y-3">
      {/* Log-Variance Barren Plateau Benchmark */}
      <div className="rounded-lg border border-slate-800 bg-slate-950/80 p-2.5">
        <div className="mb-1.5 flex items-center justify-between text-[11px]">
          <span className="font-semibold tracking-wider text-slate-300 uppercase">
            Gradient Variance Scaling
          </span>
          <div className="flex items-center gap-3 text-[10px]">
            <span className="flex items-center gap-1 text-cyan-400">
              <span className="inline-block h-2 w-2 rounded-full bg-cyan-400" />
              MERA-QCNN
            </span>
            <span className="flex items-center gap-1 text-rose-400">
              <span className="inline-block h-2 w-2 rounded-full bg-rose-400" />
              HEA
            </span>
          </div>
        </div>

        <svg viewBox={`0 0 ${width} ${height}`} className="w-full overflow-visible">
          {[-1, -3, -5].map((exp) => {
            const y = mapY(Math.pow(10, exp));
            return (
              <g key={exp}>
                <line
                  x1={padL}
                  y1={y}
                  x2={width - padR}
                  y2={y}
                  stroke="#1e293b"
                  strokeDasharray="2,2"
                />
                <text
                  x={padL - 5}
                  y={y + 3}
                  textAnchor="end"
                  fontSize="8.5"
                  fill="#64748b"
                >
                  10^{exp}
                </text>
              </g>
            );
          })}

          {scalingData.map((pt, i) => (
            <text
              key={pt.numQubits}
              x={mapX(i)}
              y={height - 5}
              textAnchor="middle"
              fontSize="9"
              fill="#94a3b8"
            >
              N={pt.numQubits}
            </text>
          ))}

          {scalingData.length > 1 && (
            <>
              <polyline
                fill="none"
                stroke="#22d3ee"
                strokeWidth="2"
                points={qcnnPoints}
              />
              <polyline
                fill="none"
                stroke="#fb7185"
                strokeWidth="2"
                strokeDasharray="3,2"
                points={heaPoints}
              />
            </>
          )}

          {scalingData.map((pt, i) => (
            <g key={`pts-${pt.numQubits}`}>
              <circle
                cx={mapX(i)}
                cy={mapY(pt.qcnnVariance)}
                r="3.2"
                fill="#22d3ee"
              />
              <circle
                cx={mapX(i)}
                cy={mapY(pt.heaVariance)}
                r="3.2"
                fill="#fb7185"
              />
            </g>
          ))}
        </svg>
      </div>

      {/* Live Training Loss Curve */}
      <div className="rounded-lg border border-slate-800 bg-slate-950/80 p-2.5">
        <div className="mb-1 flex items-center justify-between text-[11px]">
          <span className="font-semibold tracking-wider text-slate-300 uppercase">
            Variational Loss Trajectory
          </span>
          <span className="font-mono text-[10px] text-emerald-400">
            {history.length > 0
              ? `L = ${history[history.length - 1].loss.toFixed(4)}`
              : 'UNTRAINED'}
          </span>
        </div>

        <svg viewBox={`0 0 ${width} 74`} className="w-full">
          <line x1={padL} y1={60} x2={width - padR} y2={60} stroke="#1e293b" />
          <line x1={padL} y1={10} x2={width - padR} y2={10} stroke="#1e293b" strokeDasharray="2,2" />
          <text x={padL - 5} y={13} textAnchor="end" fontSize="8" fill="#64748b">
            1.2
          </text>
          <text x={padL - 5} y={62} textAnchor="end" fontSize="8" fill="#64748b">
            0.0
          </text>

          {history.length > 1 ? (
            <polyline
              fill="none"
              stroke="#10b981"
              strokeWidth="2"
              points={history
                .map((m, idx) => {
                  const x = padL + (idx / Math.max(1, history.length - 1)) * plotW;
                  const y = 60 - Math.min(1.2, m.loss) * (50 / 1.2);
                  return `${x},${y}`;
                })
                .join(' ')}
            />
          ) : (
            <text
              x={padL + plotW / 2}
              y={38}
              textAnchor="middle"
              fontSize="9.5"
              fill="#475569"
            >
              Click &quot;Train QCNN&quot; to optimize via QFIM Natural Gradient
            </text>
          )}
        </svg>
      </div>
    </div>
  );
};
