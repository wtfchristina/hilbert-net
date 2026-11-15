import { C, type Complex, Statevector } from './complexState';

export type QuantumPhaseLabel = 'SPT_CLUSTER' | 'PARAMAGNETIC' | 'ANTIFERROMAGNETIC';

export interface HamiltonianParams {
  numQubits: number;
  h1: number;
  h2: number;
}

export interface PhaseSample {
  params: HamiltonianParams;
  state: Statevector;
  groundEnergy: number;
  localMagnetizationZ: number;
  transversePolarizationX: number;
  stringOrderParameter: number;
  halfChainEntropy: number;
  truePhase: QuantumPhaseLabel;
}

export function applyClusterIsingHamiltonian(
  state: Statevector,
  h1: number,
  h2: number
): Statevector {
  const n = state.numQubits;
  const dim = state.dim;
  const out = Array.from({ length: dim }, () => C.zero());

  for (let basis = 0; basis < dim; basis++) {
    const amp = state.amps[basis];
    if (amp.re === 0 && amp.im === 0) continue;

    let diagCoeff = 0;
    if (h2 !== 0) {
      for (let i = 0; i < n - 1; i++) {
        const b1 = (basis >> (n - 1 - i)) & 1;
        const b2 = (basis >> (n - 2 - i)) & 1;
        const zz = b1 === b2 ? 1 : -1;
        diagCoeff -= h2 * zz;
      }
    }
    if (diagCoeff !== 0) {
      out[basis] = C.add(out[basis], C.scale(amp, diagCoeff));
    }

    if (h1 !== 0) {
      for (let i = 0; i < n; i++) {
        const flipMask = 1 << (n - 1 - i);
        const targetBasis = basis ^ flipMask;
        out[targetBasis] = C.add(out[targetBasis], C.scale(amp, -h1));
      }
    }

    for (let i = 0; i < n - 2; i++) {
      const bLeft = (basis >> (n - 1 - i)) & 1;
      const bRight = (basis >> (n - 3 - i)) & 1;
      const zSign = bLeft === bRight ? 1 : -1;
      const midFlipMask = 1 << (n - 2 - i);
      const targetBasis = basis ^ midFlipMask;

      out[targetBasis] = C.add(out[targetBasis], C.scale(amp, -zSign));
    }
  }

  return new Statevector(n, out);
}

export function solveClusterIsingGroundState(
  params: HamiltonianParams,
  steps = 85,
  dt = 0.08
): PhaseSample {
  const { numQubits, h1, h2 } = params;
  const dim = 1 << numQubits;

  const initAmps: Complex[] = Array.from({ length: dim }, (_, idx) =>
    C.from(1 + 0.03 * Math.sin(idx * 1.7), 0.01 * Math.cos(idx * 2.3))
  );
  let psi = new Statevector(numQubits, initAmps).normalize();

  for (let s = 0; s < steps; s++) {
    const hPsi = applyClusterIsingHamiltonian(psi, h1, h2);
    const nextAmps = psi.amps.map((a, idx) => C.sub(a, C.scale(hPsi.amps[idx], dt)));
    psi = new Statevector(numQubits, nextAmps).normalize();
  }

  const hPsiFinal = applyClusterIsingHamiltonian(psi, h1, h2);
  const groundEnergy = psi.innerProduct(hPsiFinal).re;

  const { localMagnetizationZ, transversePolarizationX, stringOrderParameter } =
    measureClusterIsingObservables(psi);
  const halfChainEntropy = psi.getHalfChainEntropy();

  let truePhase: QuantumPhaseLabel = 'SPT_CLUSTER';
  if (h1 > 1.0 && h1 >= h2) {
    truePhase = 'PARAMAGNETIC';
  } else if (h2 > 1.0 && h2 > h1) {
    truePhase = 'ANTIFERROMAGNETIC';
  }

  return {
    params,
    state: psi,
    groundEnergy,
    localMagnetizationZ,
    transversePolarizationX,
    stringOrderParameter,
    halfChainEntropy,
    truePhase,
  };
}

export function measureClusterIsingObservables(state: Statevector): {
  localMagnetizationZ: number;
  transversePolarizationX: number;
  stringOrderParameter: number;
} {
  const n = state.numQubits;
  const dim = state.dim;

  let sumZ = 0;
  let sumX = 0;
  let stringOrder = 0;

  let xStringMask = 0;
  for (let i = 1; i < n - 1; i += 2) {
    xStringMask |= 1 << (n - 1 - i);
  }

  for (let basis = 0; basis < dim; basis++) {
    const amp = state.amps[basis];
    const prob = C.abs2(amp);

    for (let i = 0; i < n; i++) {
      const bit = (basis >> (n - 1 - i)) & 1;
      sumZ += prob * (bit === 0 ? 1 : -1);
    }

    for (let i = 0; i < n; i++) {
      const flipped = basis ^ (1 << (n - 1 - i));
      sumX += C.mul(C.conj(amp), state.amps[flipped]).re;
    }

    const z0 = ((basis >> (n - 1)) & 1) === 0 ? 1 : -1;
    const zLast = (basis & 1) === 0 ? 1 : -1;
    const flippedString = basis ^ xStringMask;
    stringOrder += z0 * zLast * C.mul(C.conj(amp), state.amps[flippedString]).re;
  }

  return {
    localMagnetizationZ: Math.abs(sumZ / n),
    transversePolarizationX: sumX / n,
    stringOrderParameter: Math.abs(stringOrder),
  };
}
