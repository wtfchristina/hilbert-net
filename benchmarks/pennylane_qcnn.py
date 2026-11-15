"""
HilbertNet Research Companion:
MERA-Structured Quantum Convolutional Neural Network (QCNN) vs. Hardware-Efficient Ansatz (HEA)
for 1D Cluster-Ising Symmetry-Protected Topological (SPT) Phase Recognition.

Requirements:
    pip install pennylane torch numpy scipy
"""

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


def build_cluster_ising_hamiltonian(num_qubits: int, h1: float, h2: float) -> qml.Hamiltonian:
    """
    Constructs H(h1, h2) = -sum Z_i X_{i+1} Z_{i+2} - h1 sum X_i - h2 sum Z_i Z_{i+1}.
    """
    coeffs = []
    obs = []

    # Three-body cluster stabilizers
    for i in range(num_qubits - 2):
        coeffs.append(-1.0)
        obs.append(qml.PauliZ(i) @ qml.PauliX(i + 1) @ qml.PauliZ(i + 2))

    # Transverse field X_i
    if abs(h1) > 1e-8:
        for i in range(num_qubits):
            coeffs.append(-h1)
            obs.append(qml.PauliX(i))

    # Two-body Ising exchange Z_i Z_{i+1}
    if abs(h2) > 1e-8:
        for i in range(num_qubits - 1):
            coeffs.append(-h2)
            obs.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))

    return qml.Hamiltonian(coeffs, obs)


def exact_ground_state(num_qubits: int, h1: float, h2: float) -> tuple[np.ndarray, int]:
    """
    Computes exact ground state |psi_0(h1, h2)> and phase label:
      0: Z2 x Z2 SPT Cluster
      1: Trivial Paramagnetic (h1 > 1, h1 >= h2)
      2: Antiferromagnetic Ising (h2 > 1, h2 > h1)
    """
    ham = build_cluster_ising_hamiltonian(num_qubits, h1, h2)
    mat = qml.matrix(ham, wire_order=range(num_qubits))
    eigvals, eigvecs = np.linalg.eigh(mat)
    ground_vec = eigvecs[:, 0]

    if h1 > 1.0 and h1 >= h2:
        label = 1
    elif h2 > 1.0 and h2 > h1:
        label = 2
    else:
        label = 0
    return ground_vec, label


def conv_block(params: torch.Tensor, wires: list[int]) -> None:
    q0, q1 = wires
    qml.RY(params[0], wires=q0)
    qml.RY(params[1], wires=q1)
    qml.CNOT(wires=[q0, q1])
    qml.RZ(params[2], wires=q0)
    qml.RY(params[3], wires=q1)
    qml.CNOT(wires=[q1, q0])
    qml.RY(params[4], wires=q0)
    qml.RZ(params[5], wires=q1)


def pool_block(params: torch.Tensor, control_wire: int, target_wire: int) -> None:
    qml.RY(params[0], wires=control_wire)
    qml.CRZ(params[1], wires=[control_wire, target_wire])
    qml.CRX(params[2], wires=[control_wire, target_wire])


def create_mera_qcnn_qnode(num_qubits: int):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(state: torch.Tensor, weights: torch.Tensor):
        qml.StatePrep(state, wires=range(num_qubits))
        active = list(range(num_qubits))
        scale = 0

        while len(active) > 1:
            offset = (scale * 9) % weights.shape[0]
            # Convolutional disentangling layer
            for idx in range(len(active) - 1):
                conv_block(weights[offset : offset + 6], [active[idx], active[idx + 1]])

            # Pooling partial-trace contraction layer
            next_active = []
            for idx in range(0, len(active), 2):
                keep = active[idx]
                next_active.append(keep)
                if idx + 1 < len(active):
                    pool_block(weights[offset + 6 : offset + 9], active[idx + 1], keep)

            active = next_active
            scale += 1

        readout = active[0]
        return [qml.expval(qml.PauliZ(readout)), qml.expval(qml.PauliX(readout))]

    return circuit


class HybridMERAQCNN(nn.Module):
    def __init__(self, num_qubits: int = 6):
        super().__init__()
        self.qnode = create_mera_qcnn_qnode(num_qubits)
        self.weights = nn.Parameter(0.25 * torch.randn(18, dtype=torch.float64))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        logits = []
        for state in states:
            ez, ex = self.qnode(state, self.weights)
            s0 = 2.4 * ez
            s1 = 2.4 * (-0.5 * ez + 0.866025 * ex)
            s2 = 2.4 * (-0.5 * ez - 0.866025 * ex)
            logits.append(torch.stack([s0, s1, s2]))
        return torch.stack(logits)


if __name__ == "__main__":
    n_qubits = 6
    grid = [
        (0.15, 0.15), (0.35, 0.25), (0.20, 0.45),
        (1.45, 0.20), (1.75, 0.35), (1.35, 0.65),
        (0.20, 1.45), (0.35, 1.75), (0.65, 1.35),
    ]
    dataset = [exact_ground_state(n_qubits, h1, h2) for h1, h2 in grid]
    x_train = torch.tensor(np.array([d[0] for d in dataset]), dtype=torch.complex128)
    y_train = torch.tensor([d[1] for d in dataset], dtype=torch.long)

    model = HybridMERAQCNN(num_qubits=n_qubits)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.08)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 26):
        optimizer.zero_grad()
        out = model(x_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0 or epoch == 1:
            acc = (out.argmax(dim=1) == y_train).float().mean().item()
            print(f"Epoch {epoch:02d} | Cross-Entropy Loss: {loss.item():.4f} | Accuracy: {acc*100:.1f}%")
