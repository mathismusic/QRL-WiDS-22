from qiskit import *
from math import pi

backend = Aer.get_backend('aer_simulator')

def cR(k: int, inverse=False):
    """returns the controlled R_k gate. returns its inverse if the inverse option is set.

    :param inverse: whether or not to invert, defaults to False
    :type inverse: bool, optional
    :return: the requested controlled gate
    :rtype: ControlledGate
    """
    ckt = QuantumCircuit(1)
    ckt.p((-1 if inverse else 1)*pi/(2**(k-1)), 0)
    ckt.draw('mpl')
    return ckt.to_gate(label='$R_{}^*$'.format(k) if inverse else f'$R_{k}$').control(1)

def qft(n: int, inverse=False):
    """returns the gate performing an n-qubit ft or ift.

    :param n: the size of the gate
    :type n: int
    :param inverse: is the inverse gate requested, defaults to False
    :type inverse: bool, optional
    """
    ckt = QuantumCircuit(n)
    if not inverse:
        for i in range(n):
            ckt.h(i)
            for k in range(2, n-i+1):
                # print([k+i-1, i])
                ckt.append(cR(k), [k+i-1, i])
        for i in range(n//2):
            ckt.swap(i, n-1-i)
    else:
        for i in range(n//2): 
            ckt.swap(i, n-1-i)
        for i in reversed(range(n)):
            for k in reversed(range(2, n-i+1)):
                ckt.append(cR(k, inverse=True), [k+i-1, i])
            ckt.h(i)

    return ckt.to_gate(label=('  QFT*  ' if inverse else '  QFT  '))

if __name__ == '__main__':
    ckt = QuantumCircuit(5)
    ckt.append(qft(5, inverse=True), range(5))
    ckt.decompose().draw(initial_state=True)