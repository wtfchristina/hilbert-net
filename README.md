# HilbertNet — MERA-QCNN & Quantum Phase Manifold Studio

An interactive 3D WebGL & TypeScript quantum machine learning laboratory modeling **Multi-scale Entanglement Renormalization Ansatz (MERA) Quantum Convolutional Neural Networks** on the 1D **Cluster-Ising Hamiltonian**.

## Scientific & Architectural Highlights

* **Exact 2^N-Dimensional Complex Statevector Engine:** Client-side matrix-free Imaginary Time Evolution solver computing ground states |psi_0(h1, h2)> across N in {4, 6, 8} qubits (dim <= 256 complex amplitudes) for H(h1, h2) = -sum Z_i X_{i+1} Z_{i+2} - h1 sum X_i - h2 sum Z_i Z_{i+1}.
* **Non-Local Topological Phase Recognition:** Demonstrates why single-qubit local magnetization |<Z_i>| ~ 0 fails to distinguish the Z2 x Z2 Symmetry-Protected Topological (SPT) phase from paramagnetic disorder, while non-local string operators <Z_1 X_2 X_4 ... Z_N> and hierarchical MERA disentangling unitaries isolate the topological invariant.
* **Barren Plateau Scaling & Quantum Natural Gradient Descent (QNGD):** Contrasts O(log N)-depth MERA-QCNN polynomial gradient variance against exponential O(2^-N) gradient collapse in unstructured Hardware-Efficient Ansatz (HEA) circuits, optimizing parameters via the Fubini-Study Quantum Fisher Information Matrix (QFIM).
* **3D Three.js Tensor-Network & Reduced Density Matrix Telemetry:** Renders layer-by-layer partial trace contraction, Von Neumann entanglement entropy S(rho_q), and Bloch vector purity |r| above an interactive (h1, h2) quantum phase manifold.

## Local Development

    npm install
    npm run dev

## Research Companion (benchmarks/pennylane_qcnn.py)

Includes a hybrid PyTorch + PennyLane parameter-shift implementation reproducing the exact Cluster-Ising MERA-QCNN architecture for offline scaling experiments.
