from qiskit import Aer
backend = Aer.get_backend('aer_simulator')
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import Operator
from qiskit.circuit.library.standard_gates import HGate, PhaseGate
from qft import qft, cR
from simulation import simulate
import numpy as np

class Oracle:
    def __init__(self, U: Gate) -> None:
        self.label = 'U'
        self.index = 0
        self.U: Gate = UnitaryGate(U, label='${}^{{2^{}}}$'.format(self.label, self.index))
    def apply(self, ckt: QuantumCircuit, c: int, t: list[int]):
        ckt.append(self.U.control(1), [c, *t])
        self.index += 1
        self.U = UnitaryGate(np.linalg.matrix_power(self.U.to_matrix(), 2), label='${}^{{2^{{{}}}}}$'.format(self.label, self.index))

def estimate_phase(U: Gate, evec: np.ndarray[np.ndarray], t: int):
    ckt = QuantumCircuit(t + len(evec), t)

    # initialize state |u\rangle
    for i in range(len(evec)):
        ckt.initialize(evec[i], t + i)
    for i in range(t):
        ckt.h(i)

    # initialize the oracle
    oracle = Oracle(U)

    for i in range(t):
        oracle.apply(ckt, t-1-i, range(t, t + len(evec)))    
    
    # apply ift to get phase phi
    gate = qft(t, inverse=True)
    ckt.append(gate, range(t))
    ckt.measure(range(t), range(t))
    ckt.draw('mpl', filename='phase_est_ckt')

    shots = 2000
    counts = simulate(ckt, backend=backend, shots=shots)
    phase = max(counts.items(), key=lambda item: item[1])[0]
    print("Predicted phase:", int(phase[::-1], 2)/(2**t), end=" ") # phase is |q0..qt>, qiskit stores as |qt..q0> in counts
    print("with confidence =", int(np.round(counts[phase]/shots * 100)), "\b%")
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)

if __name__ == '__main__':
    n = 5 # bits of accuracy
    eps = 0.01 # chance of failure
    t = n + np.ceil(np.log2(2 + 1/(2*eps))); t = int(t)
    counts = estimate_phase(PhaseGate(theta=2*np.pi*0.6772), [[0, 1]], t)
    print("Qubits used:", t)
    print("Top 5:", [(int(p[::-1], 2)/(2**t), c)for p, c in counts[:5]])
