import itertools
import math

import cirq
import cirq_google
import numpy as np
from typing import Literal

from sympy import N, Q

#Part A
def ghz_part_a(qubits, low, high):
    n = high - low

    if n == 1:
        return cirq.Circuit(cirq.H(qubits[low]))

    mid = low + (n + 1) // 2
    circuit =  ghz_part_a(qubits, low, mid) 
    for i in range(high - mid):
        circuit.append([cirq.H(qubits[mid + i]),
            cirq.CZ(qubits[low + i], qubits[mid + i]),
            cirq.H(qubits[mid + i])])

    circuit = cirq.add_dynamical_decoupling(circuit)
    return circuit

#Part B
def ghz_part_b(qubits):
    circuit = cirq.Circuit()
    linked_qubits = []
    circuit.append(cirq.H.on_each(*qubits))
    root = sorted(qubits)[len(qubits)//2]
    linked_qubits.append(root)
    for q in linked_qubits:
        for n in q.neighbors(qubits):
            if n not in linked_qubits:
                circuit.append(cirq.CZ(q,n))
                circuit.append(cirq.H(n))
                linked_qubits.append(n)
    
    circuit = cirq.add_dynamical_decoupling(circuit)
    return circuit

#Part C
def ghz_part_c(qubits):
    circuit = cirq.Circuit()
    linked_qubits = []
    root = sorted(qubits)[len(qubits)//2]
    circuit.append(cirq.H(root))
    linked_qubits.append(root)
    for q in linked_qubits:
        for n in q.neighbors(qubits):
            if n not in linked_qubits:
                circuit.append(cirq.CNOT(q,n))
                linked_qubits.append(n)
    
    circuit = cirq.add_dynamical_decoupling(circuit)
    return circuit


def create_ghz_state(qubits: set[cirq.GridQubit], part: Literal['a','b','c']) -> cirq.Circuit:
    """Create a GHZ state on the specified qubits.

    Args:
        qubits: The qubits on which to prepare the state
        part: Which part of the question this is meant to solve.

    Returns:
        A circuit that creates a GHZ state on these qubits.
    """
    num_qubits = len(qubits)

    if part == 'a':
        q = sorted(qubits)
        return ghz_part_a(q, 0, len(q))
    elif part == 'b':
        q = set(qubits)
        return ghz_part_b(q)
    elif part == 'c':
        q = set(qubits)
        return ghz_part_c(q)
    else:
        return -1; # error

def check_ghz_statevector(
        circuit: cirq.Circuit,
        qubits: set[cirq.GridQubit],
        num_trials: int = 10, # this can be 1 unless the circuit is adaptive
        sampler: cirq.SimulatesFinalState = cirq.Simulator()
        ) -> bool:
    all_qubits = circuit.all_qubits()
    ancilla_qubits = all_qubits.difference(qubits)
    ordered_qubits = sorted(qubits) + sorted(ancilla_qubits)
    keep_indices = range(len(qubits))
    num_qubits = len(qubits)
    ideal_state = np.zeros(2**num_qubits)
    ideal_state[0] = 1 / np.sqrt(2)
    ideal_state[-1] = 1 / np.sqrt(2)
    for _ in range(num_trials):
        psi_circuit_full = sampler.simulate(circuit, qubit_order=ordered_qubits).final_state_vector
        psi_circuit = cirq.sub_state_vector(
            state_vector=psi_circuit_full,
            keep_indices=keep_indices
        ) if len(all_qubits) > len(qubits) else psi_circuit_full
        if not np.isclose(np.abs(ideal_state @ psi_circuit) ** 2, 1.0):
            return False
    return True

def test_operations_on_adjacent_qubits(circuit: cirq.Circuit):
    for op in circuit.all_operations():
        assert len(op.qubits) <= 2
        if len(op.qubits) == 2:
            assert op.gate == cirq.CZ
            dy = abs(op.qubits[0].row - op.qubits[1].row)
            dx = abs(op.qubits[0].col - op.qubits[1].col)
            assert dx in {0, 1}
            assert dy in {0, 1}
            assert dx + dy == 1

if __name__ == "__main__":
# we may add additional qubit choices beyond these
    qubit_choices = [
        cirq.GridQubit.rect(1, 20),
        cirq.GridQubit.rect(4, 5),
        cirq_google.Willow105.metadata.qubit_set,
        cirq.GridQubit.rect(5,5)
    ]

    for part in 'abc':
        score = 0.0;
        for qubits in qubit_choices:
            circuit = create_ghz_state(qubits, part)
            n = len(qubits)
            if n <= 22:
                ghz = check_ghz_statevector(circuit, qubits, 1)
            else:
                ghz = "pass"
            if part in 'b':
                test_operations_on_adjacent_qubits(circuit)

            score += len(qubits) ** 2 / len(create_ghz_state(qubits, part))
            print(f'Part: {part}, GHZ? {ghz}, Depth: {len(circuit)}, Score: {score}')
        
        print(circuit)
        print(f'Total Score: {score}')
