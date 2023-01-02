from qiskit import *

def simulate(ckt: QuantumCircuit, backend, shots=2000):
    transpiled_ckt = transpile(ckt, backend=backend) # convert (non-standard) gates so that the backend understands them.
    qobj = assemble(transpiled_ckt) # now assemble to a qobj and run
    return backend.run(qobj, shots=shots).result().get_counts()
