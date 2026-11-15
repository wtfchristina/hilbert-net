import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  BookOpen,
  Compass,
  Cpu,
  Layers,
  Play,
  RotateCcw,
  ShieldAlert,
  Sparkles,
  StepForward,
} from 'lucide-react';
import {
  type QuantumPhaseLabel,
  solveClusterIsingGroundState,
} from './quantum/clusterIsing';
import {
  type AnsatzType,
  computeBarrenPlateauScaling,
  generatePhaseTrainingSet,
  type OptimizerType,
  QCNNEngine,
  type TrainingStepMetrics,
} from './quantum/qcnnEngine';
import { QuantumManifoldScene } from './components/QuantumManifoldScene';
import { BarrenPlateauChart } from './components/BarrenPlateauChart';

const PHASE_BADGE: Record<
  QuantumPhaseLabel,
  { label: string; color: string; desc: string }
> = {
  SPT_CLUSTER: {
    label: 'Z₂ × Z₂ SPT Cluster',
    color: 'text-cyan-400 border-cyan-500/40 bg-cyan-500/10',
    desc: 'Topological phase hidden from local <Z_i> observables; detected via non-local string order & MERA disentangling.',
  },
  PARAMAGNETIC: {
    label: 'Trivial Paramagnetic',
    color: 'text-purple-400 border-purple-500/40 bg-purple-500/10',
    desc: 'Transverse field h₁ dominates; ground state approaches product state |+⟩⊗N with low entanglement.',
  },
  ANTIFERROMAGNETIC: {
    label: 'Symmetry-Broken Ising',
    color: 'text-amber-400 border-amber-500/40 bg-amber-500/10',
    desc: 'Two-body Z_i Z_{i+1} exchange dominates; long-range Neel correlations break Z₂ spin-flip symmetry.',
  },
};

const DOCENT_ACTS = [
  {
    act: 'ACT I // HIDDEN TOPOLOGICAL ORDER',
    title: 'Why Classical Local Observables Fail on SPT Phases',
    body: 'In the Cluster-Ising model H = -Σ Z_i X_{i+1} Z_{i+2} - h₁ Σ X_i - h₂ Σ Z_i Z_{i+1}, the Symmetry-Protected Topological (SPT) phase has zero local magnetization ⟨Z_i⟩ ≈ 0. Only non-local string operators ⟨Z₁ X₂ X₄ ... Z_N⟩ or multi-qubit entangling circuits distinguish it from disorder.',
    preset: { h1: 0.2, h2: 0.2, ansatz: 'MERA_QCNN' as AnsatzType },
  },
  {
    act: 'ACT II // MERA TENSOR RENORMALIZATION',
    title: 'Hierarchical Pooling & Partial Trace Contraction',
    body: 'Watch the 3D Bloch spheres from bottom to top: entangled input qubits have short internal vectors (|r| < 1, high Von Neumann entropy S(ρ)). Each MERA Convolutional + Pooling layer traces out half the qubits (O(log N) depth), concentrating topological invariants into a pure readout qubit.',
    preset: { h1: 0.35, h2: 0.25, ansatz: 'MERA_QCNN' as AnsatzType },
  },
  {
    act: 'ACT III // ESCAPING BARREN PLATEAUS',
    title: 'Polynomial vs. Exponential Gradient Variance',
    body: 'Switch to the Hardware-Efficient Ansatz (HEA). Unstructured deep circuits spread quantum information uniformly across the 2^N Hilbert space (2-design concentration), causing gradient variance Var(∂L) to collapse exponentially as O(2^-N). MERA-QCNN preserves polynomial trainability.',
    preset: { h1: 0.25, h2: 0.25, ansatz: 'HARDWARE_EFFICIENT' as AnsatzType },
  },
  {
    act: 'ACT IV // QUANTUM NATURAL GRADIENT',
    title: 'Fubini-Study Metric Tensor Geodesics',
    body: 'Standard Euclidean SGD ignores the curvature of projective Hilbert space. By inverting the diagonal Quantum Fisher Information Matrix g_ii(θ), QNGD steps along state-space geodesics—converging rapidly across the h₁ = 1 and h₂ = 1 quantum phase transitions.',
    preset: { h1: 1.45, h2: 0.25, ansatz: 'MERA_QCNN' as AnsatzType },
  },
];

export function App() {
  const [numQubits, setNumQubits] = useState<number>(6);
  const [h1, setH1] = useState<number>(0.22);
  const [h2, setH2] = useState<number>(0.22);
  const [ansatz, setAnsatz] = useState<AnsatzType>('MERA_QCNN');
  const [optimizer, setOptimizer] = useState<OptimizerType>('QNGD');
  const [noiseRate, setNoiseRate] = useState<number>(0.0);
  const [autoRotate, setAutoRotate] = useState<boolean>(true);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [actIndex, setActIndex] = useState<number | null>(null);
  const [paramVersion, setParamVersion] = useState<number>(0);
  const [history, setHistory] = useState<TrainingStepMetrics[]>([]);

  const engineRef = useRef<QCNNEngine>(new QCNNEngine(6, 'MERA_QCNN', 0));

  const trainingSet = useMemo(() => generatePhaseTrainingSet(numQubits), [numQubits]);
  const barrenScaling = useMemo(() => computeBarrenPlateauScaling(), []);

  useEffect(() => {
    engineRef.current.reset(numQubits, ansatz);
    engineRef.current.noiseRate = noiseRate;
    setHistory([]);
    setParamVersion((v) => v + 1);
  }, [numQubits, ansatz]);

  useEffect(() => {
    engineRef.current.noiseRate = noiseRate;
    setParamVersion((v) => v + 1);
  }, [noiseRate]);

  const currentSample = useMemo(
    () => solveClusterIsingGroundState({ numQubits, h1, h2 }, 80, 0.08),
    [numQubits, h1, h2]
  );

  const forwardPass = useMemo(() => {
    void paramVersion;
    return engineRef.current.forward(currentSample.state, undefined, true);
  }, [currentSample, paramVersion]);

  const runSingleEpoch = () => {
    const nextEpoch = history.length + 1;
    const metrics = engineRef.current.trainStep(
      trainingSet,
      nextEpoch,
      optimizer,
      optimizer === 'QNGD' ? 0.14 : 0.22
    );
    setHistory((prev) => [...prev.slice(-39), metrics]);
    setParamVersion((v) => v + 1);
  };

  useEffect(() => {
    if (!isTraining) return;
    const id = window.setInterval(() => {
      runSingleEpoch();
    }, 260);
    return () => window.clearInterval(id);
  });

  const handleReset = () => {
    setIsTraining(false);
    engineRef.current.reset(numQubits, ansatz);
    setHistory([]);
    setParamVersion((v) => v + 1);
  };

  const applyDocentAct = (idx: number) => {
    setActIndex(idx);
    const preset = DOCENT_ACTS[idx].preset;
    setH1(preset.h1);
    setH2(preset.h2);
    setAnsatz(preset.ansatz);
  };

  const { result } = forwardPass;
  const trueBadge = PHASE_BADGE[currentSample.truePhase];
  const predBadge = PHASE_BADGE[result.predictedPhase];
  const latestMetrics = history[history.length - 1];

  return (
    <div className="flex h-screen w-screen flex-col bg-[#050811] text-slate-200 select-none">
      <header className="flex h-13 shrink-0 items-center justify-between border-b border-slate-800/90 bg-[#080d1a] px-4">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-cyan-500/40 bg-cyan-500/10 text-cyan-400">
            <Cpu className="h-4 w-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-sm font-bold tracking-wider text-white uppercase">
                HilbertNet
              </h1>
              <span className="rounded border border-cyan-500/30 bg-cyan-500/10 px-1.5 py-0.5 font-mono text-[10px] text-cyan-300">
                MERA-QCNN // 2^{numQubits} HILBERT SPACE
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              Topological Quantum Phase Recognition &amp; Barren Plateau Manifold Studio
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => applyDocentAct(actIndex === null ? 0 : (actIndex + 1) % DOCENT_ACTS.length)}
            className="flex items-center gap-1.5 rounded-md border border-violet-500/40 bg-violet-500/15 px-3 py-1.5 text-xs font-medium text-violet-200 transition hover:bg-violet-500/25"
          >
            <BookOpen className="h-3.5 w-3.5 text-violet-400" />
            {actIndex === null ? '4-Act Guided Walkthrough' : `Act ${actIndex + 1}/4: Next`}
          </button>

          <button
            onClick={() => setAutoRotate((r) => !r)}
            className={`flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs transition ${
              autoRotate
                ? 'border-cyan-500/40 bg-cyan-500/15 text-cyan-300'
                : 'border-slate-700 bg-slate-900 text-slate-400'
            }`}
          >
            <Compass className="h-3.5 w-3.5" />
            Orbit
          </button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <aside className="flex w-80 shrink-0 flex-col gap-3.5 overflow-y-auto border-r border-slate-800/90 bg-[#070b16] p-3.5">
          <section className="rounded-xl border border-slate-800 bg-slate-900/50 p-3">
            <div className="mb-2 flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-xs font-semibold tracking-wider text-cyan-300 uppercase">
                <Sparkles className="h-3.5 w-3.5" />
                Cluster-Ising Hamiltonian
              </span>
              <span className="font-mono text-[11px] text-slate-400">
                E₀ = {currentSample.groundEnergy.toFixed(3)}
              </span>
            </div>

            <div className="mb-3 rounded bg-slate-950/90 px-2.5 py-1.5 font-mono text-[10.5px] text-slate-300">
              H = -Σ ZᵢXᵢ₊₁Zᵢ₊₂ - h₁Σ Xᵢ - h₂Σ ZᵢZᵢ₊₁
            </div>

            <div className="mb-3">
              <div className="mb-1 flex justify-between text-[11px] text-slate-400">
                <span>Hilbert Dimension (N Qubits)</span>
                <span className="font-mono text-cyan-400">
                  {numQubits}q (dim {1 << numQubits})
                </span>
              </div>
              <div className="grid grid-cols-3 gap-1.5">
                {[4, 6, 8].map((n) => (
                  <button
                    key={n}
                    onClick={() => setNumQubits(n)}
                    className={`rounded py-1 font-mono text-xs transition ${
                      numQubits === n
                        ? 'border border-cyan-400/60 bg-cyan-500/20 font-semibold text-cyan-200'
                        : 'border border-slate-800 bg-slate-950 text-slate-400 hover:border-slate-700'
                    }`}
                  >
                    {n} Qubits
                  </button>
                ))}
              </div>
            </div>

            <div className="mb-2.5">
              <div className="mb-1 flex justify-between text-[11px]">
                <span className="text-slate-300">Transverse Field (h₁)</span>
                <span className="font-mono text-purple-300">{h1.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0}
                max={2}
                step={0.05}
                value={h1}
                onChange={(e) => setH1(parseFloat(e.target.value))}
                className="w-full accent-purple-500"
              />
            </div>

            <div className="mb-3">
              <div className="mb-1 flex justify-between text-[11px]">
                <span className="text-slate-300">Ising Exchange (h₂)</span>
                <span className="font-mono text-amber-300">{h2.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min={0}
                max={2}
                step={0.05}
                value={h2}
                onChange={(e) => setH2(parseFloat(e.target.value))}
                className="w-full accent-amber-500"
              />
            </div>

            <div className="grid grid-cols-3 gap-1.5 text-[10px]">
              <button
                onClick={() => {
                  setH1(0.2);
                  setH2(0.2);
                }}
                className="rounded border border-cyan-500/30 bg-cyan-500/10 py-1 text-cyan-300 hover:bg-cyan-500/20"
              >
                SPT Cluster
              </button>
              <button
                onClick={() => {
                  setH1(1.65);
                  setH2(0.25);
                }}
                className="rounded border border-purple-500/30 bg-purple-500/10 py-1 text-purple-300 hover:bg-purple-500/20"
              >
                Paramag
              </button>
              <button
                onClick={() => {
                  setH1(0.25);
                  setH2(1.65);
                }}
                className="rounded border border-amber-500/30 bg-amber-500/10 py-1 text-amber-300 hover:bg-amber-500/20"
              >
                Antiferro
              </button>
            </div>
          </section>

          <section className="rounded-xl border border-slate-800 bg-slate-900/50 p-3">
            <span className="mb-2 flex items-center gap-1.5 text-xs font-semibold tracking-wider text-slate-200 uppercase">
              <Layers className="h-3.5 w-3.5 text-cyan-400" />
              Ansatz &amp; Metric Optimizer
            </span>

            <div className="mb-2.5 grid grid-cols-2 gap-1.5 text-[11px]">
              <button
                onClick={() => setAnsatz('MERA_QCNN')}
                className={`rounded border px-2 py-1.5 text-left transition ${
                  ansatz === 'MERA_QCNN'
                    ? 'border-cyan-400/60 bg-cyan-500/15 text-cyan-200'
                    : 'border-slate-800 bg-slate-950 text-slate-400'
                }`}
              >
                <div className="font-semibold">MERA-QCNN</div>
                <div className="text-[9.5px] opacity-80">O(log N) Pooling</div>
              </button>
              <button
                onClick={() => setAnsatz('HARDWARE_EFFICIENT')}
                className={`rounded border px-2 py-1.5 text-left transition ${
                  ansatz === 'HARDWARE_EFFICIENT'
                    ? 'border-rose-400/60 bg-rose-500/15 text-rose-200'
                    : 'border-slate-800 bg-slate-950 text-slate-400'
                }`}
              >
                <div className="font-semibold">HEA Circuit</div>
                <div className="text-[9.5px] opacity-80">Barren Plateau</div>
              </button>
            </div>

            <div className="mb-3 grid grid-cols-2 gap-1.5 text-[11px]">
              <button
                onClick={() => setOptimizer('QNGD')}
                className={`rounded border py-1 font-medium ${
                  optimizer === 'QNGD'
                    ? 'border-emerald-400/60 bg-emerald-500/15 text-emerald-200'
                    : 'border-slate-800 bg-slate-950 text-slate-400'
                }`}
              >
                QFIM Natural Grad
              </button>
              <button
                onClick={() => setOptimizer('VANILLA_SGD')}
                className={`rounded border py-1 font-medium ${
                  optimizer === 'VANILLA_SGD'
                    ? 'border-sky-400/60 bg-sky-500/15 text-sky-200'
                    : 'border-slate-800 bg-slate-950 text-slate-400'
                }`}
              >
                Vanilla SGD
              </button>
            </div>

            <div className="mb-3">
              <div className="mb-1 flex items-center justify-between text-[11px]">
                <span className="flex items-center gap-1 text-slate-300">
                  <ShieldAlert className="h-3 w-3 text-amber-400" />
                  NISQ Depolarizing Noise (p)
                </span>
                <span className="font-mono text-amber-300">
                  {(noiseRate * 100).toFixed(1)}%
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={0.15}
                step={0.01}
                value={noiseRate}
                onChange={(e) => setNoiseRate(parseFloat(e.target.value))}
                className="w-full accent-amber-400"
              />
            </div>

            <div className="flex gap-1.5">
              <button
                onClick={() => setIsTraining((t) => !t)}
                className={`flex flex-1 items-center justify-center gap-1.5 rounded-md py-1.5 text-xs font-semibold transition ${
                  isTraining
                    ? 'bg-amber-500 text-slate-950'
                    : 'bg-cyan-500 text-slate-950 hover:bg-cyan-400'
                }`}
              >
                <Play className="h-3.5 w-3.5" />
                {isTraining ? 'Pause Training' : 'Train QCNN'}
              </button>
              <button
                onClick={runSingleEpoch}
                title="Single Parameter-Shift Step"
                className="rounded-md border border-slate-700 bg-slate-900 px-2.5 py-1.5 text-slate-300 hover:bg-slate-800"
              >
                <StepForward className="h-3.5 w-3.5" />
              </button>
              <button
                onClick={handleReset}
                title="Reset Circuit Parameters"
                className="rounded-md border border-slate-700 bg-slate-900 px-2.5 py-1.5 text-slate-300 hover:bg-slate-800"
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </button>
            </div>
          </section>
        </aside>

        <main className="relative flex-1 bg-[#050811]">
          <QuantumManifoldScene
            snapshots={result.snapshots}
            numQubits={numQubits}
            h1={h1}
            h2={h2}
            truePhase={currentSample.truePhase}
            predictedPhase={result.predictedPhase}
            autoRotate={autoRotate}
          />

          <div className="pointer-events-none absolute top-3 left-3 flex flex-wrap gap-2">
            <div className={`rounded-lg border px-3 py-1.5 backdrop-blur-md ${trueBadge.color}`}>
              <div className="text-[9.5px] tracking-wider uppercase opacity-80">
                Exact Ground-State Phase
              </div>
              <div className="text-xs font-bold">{trueBadge.label}</div>
            </div>

            <div className={`rounded-lg border px-3 py-1.5 backdrop-blur-md ${predBadge.color}`}>
              <div className="text-[9.5px] tracking-wider uppercase opacity-80">
                QCNN Readout ({(result.confidence * 100).toFixed(1)}%)
              </div>
              <div className="text-xs font-bold">{predBadge.label}</div>
            </div>
          </div>

          {actIndex !== null && (
            <div className="absolute right-4 bottom-4 left-4 rounded-xl border border-violet-500/40 bg-slate-950/90 p-4 shadow-2xl backdrop-blur-md">
              <div className="mb-1 flex items-center justify-between">
                <span className="font-mono text-[10.5px] font-semibold text-violet-400">
                  {DOCENT_ACTS[actIndex].act}
                </span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setActIndex((actIndex + 1) % DOCENT_ACTS.length)}
                    className="rounded bg-violet-500/20 px-2.5 py-0.5 text-[11px] font-medium text-violet-200 hover:bg-violet-500/30"
                  >
                    Next Act →
                  </button>
                  <button
                    onClick={() => setActIndex(null)}
                    className="text-xs text-slate-400 hover:text-white"
                  >
                    ✕
                  </button>
                </div>
              </div>
              <h3 className="mb-1 text-sm font-bold text-white">
                {DOCENT_ACTS[actIndex].title}
              </h3>
              <p className="text-xs leading-relaxed text-slate-300">
                {DOCENT_ACTS[actIndex].body}
              </p>
            </div>
          )}
        </main>

        <aside className="flex w-84 shrink-0 flex-col gap-3.5 overflow-y-auto border-l border-slate-800/90 bg-[#070b16] p-3.5">
          <section className="rounded-xl border border-slate-800 bg-slate-900/50 p-3">
            <span className="mb-2 flex items-center gap-1.5 text-xs font-semibold tracking-wider text-cyan-300 uppercase">
              <Activity className="h-3.5 w-3.5" />
              Order &amp; Entanglement Telemetry
            </span>

            <div className="space-y-2 text-xs">
              <div>
                <div className="flex justify-between text-[11px]">
                  <span className="text-slate-400">Non-Local String Order ⟨Z₁X₂...Z_N⟩</span>
                  <span className="font-mono font-semibold text-cyan-300">
                    {currentSample.stringOrderParameter.toFixed(3)}
                  </span>
                </div>
                <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                  <div
                    className="h-full bg-cyan-400 transition-all"
                    style={{ width: `${Math.min(100, currentSample.stringOrderParameter * 100)}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-[11px]">
                  <span className="text-slate-400">Local Magnetization |⟨Zᵢ⟩| (Fails SPT)</span>
                  <span className="font-mono text-slate-300">
                    {currentSample.localMagnetizationZ.toFixed(3)}
                  </span>
                </div>
                <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                  <div
                    className="h-full bg-amber-400 transition-all"
                    style={{ width: `${Math.min(100, currentSample.localMagnetizationZ * 100)}%` }}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 pt-1">
                <div className="rounded border border-slate-800 bg-slate-950/80 p-2">
                  <div className="text-[10px] text-slate-400">Half-Chain S(ρ_A)</div>
                  <div className="font-mono text-sm font-semibold text-purple-300">
                    {currentSample.halfChainEntropy.toFixed(3)}
                  </div>
                </div>
                <div className="rounded border border-slate-800 bg-slate-950/80 p-2">
                  <div className="text-[10px] text-slate-400">Batch Accuracy</div>
                  <div className="font-mono text-sm font-semibold text-emerald-300">
                    {latestMetrics ? `${(latestMetrics.accuracy * 100).toFixed(0)}%` : '—'}
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="rounded-xl border border-slate-800 bg-slate-900/50 p-3">
            <div className="mb-2 text-xs font-semibold tracking-wider text-slate-200 uppercase">
              Readout Qubit Phase Posteriors
            </div>
            <div className="space-y-1.5 text-[11px]">
              {[
                { label: 'SPT Cluster', prob: result.logits[0], bar: 'bg-cyan-400' },
                { label: 'Paramagnetic', prob: result.logits[1], bar: 'bg-purple-400' },
                { label: 'Antiferromagnetic', prob: result.logits[2], bar: 'bg-amber-400' },
              ].map((item) => (
                <div key={item.label}>
                  <div className="flex justify-between text-slate-300">
                    <span>{item.label}</span>
                    <span className="font-mono">{(item.prob * 100).toFixed(1)}%</span>
                  </div>
                  <div className="mt-0.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                    <div
                      className={`h-full ${item.bar} transition-all`}
                      style={{ width: `${item.prob * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </section>

          <BarrenPlateauChart scalingData={barrenScaling} history={history} />

          <div className="grid grid-cols-2 gap-2 text-[11px]">
            <div className="rounded-lg border border-slate-800 bg-slate-950/80 p-2">
              <div className="text-[10px] text-slate-400">Grad Variance</div>
              <div className="font-mono text-xs font-semibold text-cyan-300">
                {latestMetrics ? latestMetrics.gradVariance.toExponential(2) : '—'}
              </div>
            </div>
            <div className="rounded-lg border border-slate-800 bg-slate-950/80 p-2">
              <div className="text-[10px] text-slate-400">QFIM Cond. κ(g)</div>
              <div className="font-mono text-xs font-semibold text-violet-300">
                {latestMetrics ? latestMetrics.qfimConditionNumber.toFixed(2) : '1.00'}
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;
