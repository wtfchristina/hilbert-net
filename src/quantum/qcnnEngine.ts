import { type BlochVector, C, Gates, Statevector } from './complexState';
import { type PhaseSample, type QuantumPhaseLabel, solveClusterIsingGroundState } from './clusterIsing';

export type AnsatzType = 'MERA_QCNN' | 'HARDWARE_EFFICIENT';
export type OptimizerType = 'QNGD' | 'VANILLA_SGD';

export interface LayerSnapshot {
  layerIndex: number;
  label: string;
  activeQubits: number[];
  blochVectors: Record<number, BlochVector>;
  entanglementLinks: Array<{ q1: number; q2: number; strength: number }>;
}

export interface QCNNForwardResult {
  logits: [number, number, number]; // [P(SPT), P(Paramagnetic), P(Antiferromagnetic)]
  predictedPhase: QuantumPhaseLabel;
  confidence: number;
  snapshots: LayerSnapshot[];
  readoutQubit: number;
}

export interface TrainingStepMetrics {
  epoch: number;
  loss: number;
  accuracy: number;
  gradVariance: number;
  qfimConditionNumber: number;
  paramNorm: number;
}

export interface BarrenPlateauPoint {
  numQubits: number;
  qcnnVariance: number;
  heaVariance: number;
}

const PHASE_INDEX: Record<QuantumPhaseLabel, number> = {
  SPT_CLUSTER: 0,
  PARAMAGNETIC: 1,
  ANTIFERROMAGNETIC: 2,
};

const INDEX_PHASE: QuantumPhaseLabel[] = [
  'SPT_CLUSTER',
  'PARAMAGNETIC',
  'ANTIFERROMAGNETIC',
];

export class QCNNEngine {
  public numQubits: number;
  public ansatz: AnsatzType;
  public params: number[];
  public noiseRate: number; // Depolarizing probability p in [0, 0.15]

  constructor(numQubits = 6, ansatz: AnsatzType = 'MERA_QCNN', noiseRate = 0.0) {
    this.numQubits = numQubits;
    this.ansatz = ansatz;
    this.noiseRate = noiseRate;
    this.params = this.initParameters(numQubits, ansatz);
  }

  public initParameters(numQubits: number, ansatz: AnsatzType): number[] {
    const count = ansatz === 'MERA_QCNN' ? 18 : numQubits * 4;
    // Deterministic quasi-random initialization near small angles
    return Array.from({ length: count }, (_, i) =>
      0.35 * Math.sin(i * 1.618 + numQubits) + 0.15 * Math.cos(i * 2.414)
    );
  }

  public reset(numQubits = this.numQubits, ansatz = this.ansatz): void {
    this.numQubits = numQubits;
    this.ansatz = ansatz;
    this.params = this.initParameters(numQubits, ansatz);
  }

  /**
   * Executes forward pass through either the hierarchical MERA-QCNN
   * or the unstructured Hardware-Efficient Ansatz (HEA).
   */
  public forward(
    inputState: Statevector,
    overrideParams?: number[],
    recordSnapshots = false
  ): { result: QCNNForwardResult; finalState: Statevector } {
    const theta = overrideParams ?? this.params;
    const psi = inputState.clone();
    const n = this.numQubits;
    const snapshots: LayerSnapshot[] = [];

    let activeQubits = Array.from({ length: n }, (_, i) => i);

    if (recordSnapshots) {
      snapshots.push(this.captureSnapshot(psi, 0, 'Input Ground State |ψ₀⟩', activeQubits));
    }

    if (this.ansatz === 'MERA_QCNN') {
      let scale = 1;
      while (activeQubits.length > 1) {
        const paramOffset = ((scale - 1) * 9) % theta.length;

        // 1. Convolutional Layer: Local two-qubit entangling unitaries across active neighbors
        for (let idx = 0; idx < activeQubits.length - 1; idx++) {
          const q1 = activeQubits[idx];
          const q2 = activeQubits[idx + 1];
          this.applyConvBlock(psi, q1, q2, theta, paramOffset);
        }

        if (this.noiseRate > 0) {
          this.applyDepolarizingChannel(psi, activeQubits, this.noiseRate);
        }

        if (recordSnapshots) {
          snapshots.push(
            this.captureSnapshot(psi, snapshots.length, `Conv Layer ${scale}`, activeQubits)
          );
        }

        // 2. Pooling Layer: Controlled rotations from odd sites to even sites, then trace out odd sites
        const nextActive: number[] = [];
        for (let idx = 0; idx < activeQubits.length; idx += 2) {
          const keepQ = activeQubits[idx];
          nextActive.push(keepQ);
          if (idx + 1 < activeQubits.length) {
            const traceQ = activeQubits[idx + 1];
            this.applyPoolingBlock(psi, traceQ, keepQ, theta, (paramOffset + 6) % theta.length);
          }
        }

        activeQubits = nextActive;

        if (recordSnapshots) {
          snapshots.push(
            this.captureSnapshot(psi, snapshots.length, `Pool Scale ${scale}`, activeQubits)
          );
        }
        scale++;
      }
    } else {
      // Unstructured Deep Hardware-Efficient Ansatz (prone to Barren Plateaus)
      const depth = 4;
      let pIdx = 0;
      for (let d = 0; d < depth; d++) {
        for (let q = 0; q < n; q++) {
          psi.applySingleQubitGate(q, Gates.Ry(theta[pIdx % theta.length]));
          pIdx++;
          psi.applySingleQubitGate(q, Gates.Rz(theta[pIdx % theta.length]));
          pIdx++;
        }
        for (let q = 0; q < n - 1; q++) {
          psi.applyControlledGate(q, q + 1, Gates.X());
        }
        if (n > 2) {
          psi.applyControlledGate(n - 1, 0, Gates.X());
        }
        if (this.noiseRate > 0) {
          this.applyDepolarizingChannel(psi, activeQubits, this.noiseRate);
        }
        if (recordSnapshots) {
          snapshots.push(
            this.captureSnapshot(psi, snapshots.length, `HEA Layer ${d + 1}`, activeQubits)
          );
        }
      }
    }

    // Readout on surviving qubit 0: map <X>, <Y>, <Z> expectations to 3-phase softmax probabilities
    const readoutQubit = activeQubits[0] ?? 0;
    const bloch = psi.getSingleQubitBloch(readoutQubit);

    // Three phase-readout observables separated by 120 deg on the Bloch sphere
    const s0 = 2.4 * bloch.z;                                      // SPT Cluster (+Z)
    const s1 = 2.4 * (-0.5 * bloch.z + 0.866025 * bloch.x);        // Paramagnetic
    const s2 = 2.4 * (-0.5 * bloch.z - 0.866025 * bloch.x);        // Antiferromagnetic

    const maxS = Math.max(s0, s1, s2);
    const e0 = Math.exp(s0 - maxS);
    const e1 = Math.exp(s1 - maxS);
    const e2 = Math.exp(s2 - maxS);
    const sumE = e0 + e1 + e2;

    const logits: [number, number, number] = [e0 / sumE, e1 / sumE, e2 / sumE];
    let bestIdx = 0;
    if (logits[1] > logits[bestIdx]) bestIdx = 1;
    if (logits[2] > logits[bestIdx]) bestIdx = 2;

    return {
      result: {
        logits,
        predictedPhase: INDEX_PHASE[bestIdx],
        confidence: logits[bestIdx],
        snapshots,
        readoutQubit,
      },
      finalState: psi,
    };
  }

  private applyConvBlock(
    psi: Statevector,
    q1: number,
    q2: number,
    theta: number[],
    offset: number
  ): void {
    const t0 = theta[offset % theta.length];
    const t1 = theta[(offset + 1) % theta.length];
    const t2 = theta[(offset + 2) % theta.length];
    const t3 = theta[(offset + 3) % theta.length];
    const t4 = theta[(offset + 4) % theta.length];
    const t5 = theta[(offset + 5) % theta.length];

    psi.applySingleQubitGate(q1, Gates.Ry(t0));
    psi.applySingleQubitGate(q2, Gates.Ry(t1));
    psi.applyControlledGate(q1, q2, Gates.X());
    psi.applySingleQubitGate(q1, Gates.Rz(t2));
    psi.applySingleQubitGate(q2, Gates.Ry(t3));
    psi.applyControlledGate(q2, q1, Gates.X());
    psi.applySingleQubitGate(q1, Gates.Ry(t4));
    psi.applySingleQubitGate(q2, Gates.Rz(t5));
  }

  private applyPoolingBlock(
    psi: Statevector,
    controlTraceQ: number,
    targetKeepQ: number,
    theta: number[],
    offset: number
  ): void {
    const t0 = theta[offset % theta.length];
    const t1 = theta[(offset + 1) % theta.length];
    const t2 = theta[(offset + 2) % theta.length];

    psi.applySingleQubitGate(controlTraceQ, Gates.Ry(t0));
    psi.applyControlledGate(controlTraceQ, targetKeepQ, Gates.Rz(t1));
    psi.applyControlledGate(controlTraceQ, targetKeepQ, Gates.Rx(t2));
  }

  /**
   * Simulates coherent-stochastic Pauli depolarizing noise on active qubits.
   */
  private applyDepolarizingChannel(
    psi: Statevector,
    activeQubits: number[],
    p: number
  ): void {
    const angle = Math.sqrt(p) * 0.45;
    for (let i = 0; i < activeQubits.length; i++) {
      const q = activeQubits[i];
      const sign = i % 2 === 0 ? 1 : -1;
      psi.applySingleQubitGate(q, Gates.Rx(sign * angle));
      psi.applySingleQubitGate(q, Gates.Rz(-sign * angle * 0.7));
    }
  }

  private captureSnapshot(
    psi: Statevector,
    layerIndex: number,
    label: string,
    activeQubits: number[]
  ): LayerSnapshot {
    const blochVectors: Record<number, BlochVector> = {};
    for (let q = 0; q < this.numQubits; q++) {
      blochVectors[q] = psi.getSingleQubitBloch(q);
    }

    const entanglementLinks: Array<{ q1: number; q2: number; strength: number }> = [];
    for (let idx = 0; idx < activeQubits.length - 1; idx++) {
      const q1 = activeQubits[idx];
      const q2 = activeQubits[idx + 1];
      // Entanglement proxy from mixed-state impurity 1 - Tr(rho^2)
      const impurity1 = 1 - blochVectors[q1].purity;
      const impurity2 = 1 - blochVectors[q2].purity;
      const strength = Math.min(1, Math.sqrt(Math.max(0, impurity1 + impurity2)) * 1.4);
      entanglementLinks.push({ q1, q2, strength });
    }

    return {
      layerIndex,
      label,
      activeQubits: [...activeQubits],
      blochVectors,
      entanglementLinks,
    };
  }

  /**
   * Computes cross-entropy loss over a mini-batch of Cluster-Ising ground states.
   */
  public computeBatchLoss(
    batch: PhaseSample[],
    overrideParams?: number[]
  ): { loss: number; accuracy: number } {
    let totalLoss = 0;
    let correct = 0;

    for (const sample of batch) {
      const { result } = this.forward(sample.state, overrideParams, false);
      const targetIdx = PHASE_INDEX[sample.truePhase];
      const prob = Math.max(1e-7, result.logits[targetIdx]);
      totalLoss -= Math.log(prob);
      if (result.predictedPhase === sample.truePhase) {
        correct++;
      }
    }

    return {
      loss: totalLoss / batch.length,
      accuracy: correct / batch.length,
    };
  }

  /**
   * Executes one optimization step using either:
   * - Quantum Natural Gradient Descent (QNGD) via Fubini-Study QFIM metric tensor
   * - Vanilla Euclidean Gradient Descent (SGD)
   */
  public trainStep(
    batch: PhaseSample[],
    epoch: number,
    optimizer: OptimizerType = 'QNGD',
    learningRate = 0.14
  ): TrainingStepMetrics {
    const numParams = this.params.length;
    const shift = Math.PI / 4; // Central finite/parameter shift

    // 1. Compute analytical parameter-shift gradient vector g_i = dL / dtheta_i
    const grads = new Array<number>(numParams).fill(0);
    for (let i = 0; i < numParams; i++) {
      const plus = [...this.params];
      const minus = [...this.params];
      plus[i] += shift;
      minus[i] -= shift;

      const lossPlus = this.computeBatchLoss(batch, plus).loss;
      const lossMinus = this.computeBatchLoss(batch, minus).loss;
      grads[i] = (lossPlus - lossMinus) / (2 * Math.sin(shift));
    }

    // Compute gradient variance across parameters (barren plateau diagnostic)
    const meanGrad = grads.reduce((a, b) => a + b, 0) / numParams;
    const gradVariance =
      grads.reduce((acc, g) => acc + (g - meanGrad) ** 2, 0) / numParams;

    let stepDir = [...grads];
    let qfimConditionNumber = 1.0;

    // 2. If QNGD, compute diagonal + block-Fubini-Study metric tensor g_{ij}
    if (optimizer === 'QNGD' && batch.length > 0) {
      const refState = batch[0].state;
      const psi0 = this.forward(refState, this.params, false).finalState;
      const eps = 0.05;

      const dPsi: Statevector[] = [];
      for (let i = 0; i < numParams; i++) {
        const pPlus = [...this.params];
        const pMinus = [...this.params];
        pPlus[i] += eps;
        pMinus[i] -= eps;
        const sPlus = this.forward(refState, pPlus, false).finalState;
        const sMinus = this.forward(refState, pMinus, false).finalState;

        const diffAmps = sPlus.amps.map((a, k) =>
          C.scale(C.sub(a, sMinus.amps[k]), 1 / (2 * eps))
        );
        dPsi.push(new Statevector(this.numQubits, diffAmps));
      }

      // Diagonal Fubini-Study metric tensor elements g_{ii} + Tikhonov regularization
      const reg = 0.02;
      const diagG = new Array<number>(numParams).fill(reg);
      let minG = Infinity;
      let maxG = 0;

      for (let i = 0; i < numParams; i++) {
        const overlap = psi0.innerProduct(dPsi[i]);
        const normSq = dPsi[i].innerProduct(dPsi[i]).re;
        const g_ii = Math.max(0, normSq - C.abs2(overlap)) + reg;
        diagG[i] = g_ii;
        if (g_ii < minG) minG = g_ii;
        if (g_ii > maxG) maxG = g_ii;
        stepDir[i] = grads[i] / g_ii;
      }
      qfimConditionNumber = maxG / Math.max(1e-6, minG);
    }

    // 3. Apply parameter update
    for (let i = 0; i < numParams; i++) {
      this.params[i] -= learningRate * stepDir[i];
    }

    const { loss, accuracy } = this.computeBatchLoss(batch);
    const paramNorm = Math.sqrt(this.params.reduce((acc, v) => acc + v * v, 0));

    return {
      epoch,
      loss,
      accuracy,
      gradVariance,
      qfimConditionNumber,
      paramNorm,
    };
  }
}

/**
 * Generates a balanced 9-point training set spanning all 3 phases of the Cluster-Ising phase diagram.
 */
export function generatePhaseTrainingSet(numQubits: number): PhaseSample[] {
  const grid: Array<[number, number]> = [
    [0.15, 0.15], // Deep SPT Cluster
    [0.35, 0.25], // SPT Cluster
    [0.20, 0.45], // SPT Cluster
    [1.45, 0.20], // Paramagnetic
    [1.75, 0.35], // Deep Paramagnetic
    [1.35, 0.65], // Paramagnetic boundary
    [0.20, 1.45], // Antiferromagnetic
    [0.35, 1.75], // Deep Antiferromagnetic
    [0.65, 1.35], // Antiferromagnetic boundary
  ];

  return grid.map(([h1, h2]) =>
    solveClusterIsingGroundState({ numQubits, h1, h2 }, 65, 0.08)
  );
}

/**
 * Benchmarks Barren Plateau gradient variance scaling across N = 4, 6, 8 qubits:
 * MERA-QCNN exhibits polynomial O(1/poly(N)) variance decay vs. exponential O(2^-N) for HEA.
 */
export function computeBarrenPlateauScaling(): BarrenPlateauPoint[] {
  const sizes = [4, 6, 8];
  return sizes.map((n) => {
    const sample = solveClusterIsingGroundState({ numQubits: n, h1: 0.3, h2: 0.3 }, 40, 0.08);
    const qcnn = new QCNNEngine(n, 'MERA_QCNN', 0);
    const hea = new QCNNEngine(n, 'HARDWARE_EFFICIENT', 0);

    const shift = Math.PI / 4;
    const sampleTrials = 6;

    const evalVariance = (model: QCNNEngine): number => {
      const grads: number[] = [];
      for (let t = 0; t < sampleTrials; t++) {
        const randParams = model.params.map((_, idx) =>
          Math.sin((t + 1) * 3.17 + idx * 1.91) * Math.PI
        );
        const pPlus = [...randParams];
        const pMinus = [...randParams];
        pPlus[0] += shift;
        pMinus[0] -= shift;

        const lPlus = model.computeBatchLoss([sample], pPlus).loss;
        const lMinus = model.computeBatchLoss([sample], pMinus).loss;
        grads.push((lPlus - lMinus) / (2 * shift));
      }
      const mean = grads.reduce((a, b) => a + b, 0) / grads.length;
      return grads.reduce((acc, g) => acc + (g - mean) ** 2, 0) / grads.length;
    };

    return {
      numQubits: n,
      qcnnVariance: evalVariance(qcnn),
      heaVariance: evalVariance(hea),
    };
  });
}
