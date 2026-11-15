export interface Complex {
  re: number;
  im: number;
}

export type ComplexMatrix = Complex[][];

export interface BlochVector {
  x: number;
  y: number;
  z: number;
  purity: number;
  entropy: number;
  radius: number;
}

export const C = {
  zero: (): Complex => ({ re: 0, im: 0 }),
  one: (): Complex => ({ re: 1, im: 0 }),
  i: (): Complex => ({ re: 0, im: 1 }),
  from: (re: number, im = 0): Complex => ({ re, im }),
  add: (a: Complex, b: Complex): Complex => ({ re: a.re + b.re, im: a.im + b.im }),
  sub: (a: Complex, b: Complex): Complex => ({ re: a.re - b.re, im: a.im - b.im }),
  mul: (a: Complex, b: Complex): Complex => ({
    re: a.re * b.re - a.im * b.im,
    im: a.re * b.im + a.im * b.re,
  }),
  scale: (a: Complex, s: number): Complex => ({ re: a.re * s, im: a.im * s }),
  conj: (a: Complex): Complex => ({ re: a.re, im: -a.im }),
  abs2: (a: Complex): number => a.re * a.re + a.im * a.im,
};

export const Gates = {
  I: (): ComplexMatrix => [
    [C.one(), C.zero()],
    [C.zero(), C.one()],
  ],
  X: (): ComplexMatrix => [
    [C.zero(), C.one()],
    [C.one(), C.zero()],
  ],
  Y: (): ComplexMatrix => [
    [C.zero(), C.from(0, -1)],
    [C.from(0, 1), C.zero()],
  ],
  Z: (): ComplexMatrix => [
    [C.one(), C.zero()],
    [C.zero(), C.from(-1, 0)],
  ],
  H: (): ComplexMatrix => {
    const s = 1 / Math.SQRT2;
    return [
      [C.from(s), C.from(s)],
      [C.from(s), C.from(-s)],
    ];
  },
  Rx: (theta: number): ComplexMatrix => {
    const c = Math.cos(theta / 2);
    const s = Math.sin(theta / 2);
    return [
      [C.from(c, 0), C.from(0, -s)],
      [C.from(0, -s), C.from(c, 0)],
    ];
  },
  Ry: (theta: number): ComplexMatrix => {
    const c = Math.cos(theta / 2);
    const s = Math.sin(theta / 2);
    return [
      [C.from(c, 0), C.from(-s, 0)],
      [C.from(s, 0), C.from(c, 0)],
    ];
  },
  Rz: (theta: number): ComplexMatrix => {
    const c = Math.cos(theta / 2);
    const s = Math.sin(theta / 2);
    return [
      [C.from(c, -s), C.zero()],
      [C.zero(), C.from(c, s)],
    ];
  },
};

export class Statevector {
  public readonly numQubits: number;
  public readonly dim: number;
  public amps: Complex[];

  constructor(numQubits: number, initialAmps?: Complex[]) {
    this.numQubits = numQubits;
    this.dim = 1 << numQubits;
    if (initialAmps) {
      if (initialAmps.length !== this.dim) {
        throw new Error(`Expected ${this.dim} amplitudes, received ${initialAmps.length}`);
      }
      this.amps = initialAmps.map((a) => ({ ...a }));
    } else {
      this.amps = Array.from({ length: this.dim }, (_, idx) =>
        idx === 0 ? C.one() : C.zero()
      );
    }
  }

  public clone(): Statevector {
    return new Statevector(this.numQubits, this.amps);
  }

  public normalize(): this {
    let normSq = 0;
    for (let i = 0; i < this.dim; i++) {
      normSq += C.abs2(this.amps[i]);
    }
    const invNorm = normSq > 1e-16 ? 1 / Math.sqrt(normSq) : 0;
    for (let i = 0; i < this.dim; i++) {
      this.amps[i] = C.scale(this.amps[i], invNorm);
    }
    return this;
  }

  public innerProduct(other: Statevector): Complex {
    let sum = C.zero();
    for (let i = 0; i < this.dim; i++) {
      sum = C.add(sum, C.mul(C.conj(this.amps[i]), other.amps[i]));
    }
    return sum;
  }

  public applySingleQubitGate(target: number, gate: ComplexMatrix): this {
    const bit = this.numQubits - 1 - target;
    const stride = 1 << bit;

    for (let base = 0; base < this.dim; base += stride << 1) {
      for (let offset = 0; offset < stride; offset++) {
        const i0 = base + offset;
        const i1 = i0 + stride;
        const a0 = this.amps[i0];
        const a1 = this.amps[i1];

        this.amps[i0] = C.add(C.mul(gate[0][0], a0), C.mul(gate[0][1], a1));
        this.amps[i1] = C.add(C.mul(gate[1][0], a0), C.mul(gate[1][1], a1));
      }
    }
    return this;
  }

  public applyControlledGate(control: number, target: number, gate: ComplexMatrix): this {
    const cBit = this.numQubits - 1 - control;
    const tBit = this.numQubits - 1 - target;
    const cMask = 1 << cBit;
    const tMask = 1 << tBit;

    for (let i = 0; i < this.dim; i++) {
      if ((i & cMask) !== 0 && (i & tMask) === 0) {
        const i0 = i;
        const i1 = i | tMask;
        const a0 = this.amps[i0];
        const a1 = this.amps[i1];

        this.amps[i0] = C.add(C.mul(gate[0][0], a0), C.mul(gate[0][1], a1));
        this.amps[i1] = C.add(C.mul(gate[1][0], a0), C.mul(gate[1][1], a1));
      }
    }
    return this;
  }

  public applyTwoQubitGate(q1: number, q2: number, u4: ComplexMatrix): this {
    const b1 = this.numQubits - 1 - q1;
    const b2 = this.numQubits - 1 - q2;
    const m1 = 1 << b1;
    const m2 = 1 << b2;

    for (let i = 0; i < this.dim; i++) {
      if ((i & m1) === 0 && (i & m2) === 0) {
        const idx = [i, i | m2, i | m1, i | m1 | m2];
        const v = idx.map((k) => this.amps[k]);

        for (let r = 0; r < 4; r++) {
          let acc = C.zero();
          for (let c = 0; c < 4; c++) {
            acc = C.add(acc, C.mul(u4[r][c], v[c]));
          }
          this.amps[idx[r]] = acc;
        }
      }
    }
    return this;
  }

  public getSingleQubitBloch(target: number): BlochVector {
    const bit = this.numQubits - 1 - target;
    const mask = 1 << bit;

    let rho00 = 0;
    let rho11 = 0;
    let rho01 = C.zero();

    for (let i = 0; i < this.dim; i++) {
      if ((i & mask) === 0) {
        const i0 = i;
        const i1 = i | mask;
        const a0 = this.amps[i0];
        const a1 = this.amps[i1];

        rho00 += C.abs2(a0);
        rho11 += C.abs2(a1);
        rho01 = C.add(rho01, C.mul(a0, C.conj(a1)));
      }
    }

    const x = 2 * rho01.re;
    const y = -2 * rho01.im;
    const z = rho00 - rho11;
    const radius = Math.min(1, Math.sqrt(x * x + y * y + z * z));
    const purity = 0.5 * (1 + radius * radius);

    const l1 = Math.max(1e-15, (1 + radius) / 2);
    const l2 = Math.max(1e-15, (1 - radius) / 2);
    const entropy = -(l1 * Math.log(l1) + l2 * Math.log(l2));

    return { x, y, z, purity, entropy, radius };
  }

  public getHalfChainEntropy(): number {
    const nA = Math.floor(this.numQubits / 2);
    const nB = this.numQubits - nA;
    const dimA = 1 << nA;
    const dimB = 1 << nB;

    const rhoA: ComplexMatrix = Array.from({ length: dimA }, () =>
      Array.from({ length: dimA }, () => C.zero())
    );

    for (let a1 = 0; a1 < dimA; a1++) {
      for (let a2 = a1; a2 < dimA; a2++) {
        let sum = C.zero();
        for (let b = 0; b < dimB; b++) {
          const i1 = (a1 << nB) | b;
          const i2 = (a2 << nB) | b;
          sum = C.add(sum, C.mul(this.amps[i1], C.conj(this.amps[i2])));
        }
        rhoA[a1][a2] = sum;
        rhoA[a2][a1] = C.conj(sum);
      }
    }

    const eigvals = hermitianEigenvalues(rhoA);
    let entropy = 0;
    for (const lam of eigvals) {
      if (lam > 1e-12) {
        entropy -= lam * Math.log(lam);
      }
    }
    return entropy;
  }
}

function hermitianEigenvalues(mat: ComplexMatrix): number[] {
  const n = mat.length;
  const A = mat.map((row) => row.map((v) => ({ ...v })));
  const eigenvalues: number[] = [];

  for (let k = 0; k < n; k++) {
    let vec = Array.from({ length: n }, (_, i) => C.from(Math.cos(i + 1.3), Math.sin(i + 0.7)));
    let norm = Math.sqrt(vec.reduce((acc, v) => acc + C.abs2(v), 0));
    vec = vec.map((v) => C.scale(v, 1 / norm));

    let lambda = 0;
    for (let iter = 0; iter < 40; iter++) {
      const next = Array.from({ length: n }, (_, r) => {
        let s = C.zero();
        for (let c = 0; c < n; c++) {
          s = C.add(s, C.mul(A[r][c], vec[c]));
        }
        return s;
      });

      let rayleigh = 0;
      for (let i = 0; i < n; i++) {
        rayleigh += C.mul(C.conj(vec[i]), next[i]).re;
      }
      lambda = rayleigh;

      const nextNorm = Math.sqrt(next.reduce((acc, v) => acc + C.abs2(v), 0));
      if (nextNorm < 1e-14) break;
      vec = next.map((v) => C.scale(v, 1 / nextNorm));
    }

    eigenvalues.push(Math.max(0, lambda));
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const outer = C.scale(C.mul(vec[r], C.conj(vec[c])), lambda);
        A[r][c] = C.sub(A[r][c], outer);
      }
    }
  }

  return eigenvalues;
}
