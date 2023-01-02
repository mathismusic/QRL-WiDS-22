from qiskit import *
from qiskit.circuit.library.standard_gates import XGate, ZGate
from simulation import simulate
import numpy as np

def PhaseShift(n: int):
    ckt = QuantumCircuit(n)
    for i in range(n): ckt.x(i)
    ckt.append(ZGate().control(n-1), range(n))
    for i in range(n): ckt.x(i)
    return ckt.to_gate(label=' 2|0⟩⟨0| - I ')

def ApplyOracle(n: int, s: str, bits_to_check: list[int]=None):
    ckt = QuantumCircuit(n+1)
    if bits_to_check is None: bits_to_check = range(n)
    for i in bits_to_check:
        if s[i] == "0": ckt.x(i)
    ckt.append(XGate().control(num_ctrl_qubits=len(bits_to_check)), bits_to_check + [n]) # check if the x matches s perfectly wherever to be checked
    for i in bits_to_check:
        if s[i] == "0": ckt.x(i) # get the X-ed qubits back to old state
    return ckt.to_gate(label="   Oracle   ")

def GroverIteration(n: int, s: str, bits_to_check: list[int] = None):
    ckt = QuantumCircuit(n+1)
    ckt.append(ApplyOracle(n, s, bits_to_check), range(n+1))
    for i in range(n): ckt.h(i)
    ckt.append(PhaseShift(n), range(n))
    for i in range(n): ckt.h(i)
    return ckt.to_gate(label='G')

def GroverSolver(N: int, M: int, s: str, bits_to_check: list[int] = None):
    n = len(s) # or len(s) = log N since N = 2**len(s)
    if M >= N/2: 
        N *= 2
        n += 1 # N -> 2*N.
        s += '0'
        if bits_to_check: bits_to_check += [n-1]

    # now treat like a usual Grover with M <= N/2
    ckt = QuantumCircuit(n+1, n) # n plus ancilla in |-\rangle
    ckt.x(n)
    for i in range(n+1): ckt.h(i)

    theta = 2*np.arccos(np.sqrt((N-M)/N))
    print('θ/2 ≈', theta/2 * 180/np.pi)
    iters = int(np.pi/2/theta) # = round(pi/2theta - 0.5)
    print('iterations required:', iters)
    for _ in range(iters):
        ckt.append(GroverIteration(n, s, bits_to_check), range(n+1))

    ckt.measure(range(n), range(n))
    print(ckt)
    ckt.draw('mpl', filename='grover_example')

    # run and return results
    counts = simulate(ckt, backend=backend)
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)

if __name__ == '__main__':
    s = "10111"
    bits_to_check = [1, 4, 2, 3, 0]
    n = len(s)
    backend = Aer.get_backend('aer_simulator')
    counts = GroverSolver(2**n, 2**(n - len(bits_to_check)), s, bits_to_check)
    print([(k[::-1], v) for k, v in counts])