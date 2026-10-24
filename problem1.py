import itertools
import math

import cirq
import cirq_google
import numpy as np
from typing import Literal


def create_ghz_state(qubits: set[cirq.GridQubit], part: Literal['a','b','c']) -> cirq.Circuit:
    """Create a GHZ state on the specified qubits.

    Args:
        qubits: The qubits on which to prepare the state
        part: Which part of the question this is meant to solve.

    Returns:
        A circuit that creates a GHZ state on these qubits.
    """
    num_qubits = len(qubits)
    qs = sorted(qubits)
    if part == 'a':
        return ghz_state_helper(qs, num_qubits, 0, len(qubits))
    
    elif part == 'b':
        return ghz_state_helper(qs, num_qubits, 0, len(qubits))
    
    elif part == 'c':
        return ghz_state_helper(qs, num_qubits, 0, len(qubits))
    
    else:
        return -1; # error

def ghz_state_helper(qubits: set[cirq.GridQubit], num_qubits: int, low: int, high: int) -> cirq.Circuit:
    mid = low + (high - low) // 2

    if num_qubits == 1:
        return cirq.Circuit(cirq.H(qubits[low]))

    return (
        ghz_state_helper(qubits, num_qubits//2, low, mid) 
        + ghz_state_helper(qubits, num_qubits//2, mid, high) 
        + cirq.CZ(qubits[low], qubits[mid]) 
        + cirq.H(qubits[mid])
        )


def check_ghz_statevector(
        circuit: cirq.Circuit,
        qubits: set[cirq.GridQubit],
        num_trials: int = 10, # this can be 1 unless the circuit is adaptive
        sampler: cirq.SimulatesFinalState = cirq.Simulator()
        ) -> bool:
    all_qubits = circuit.all_qubits()
    ancilla_qubits = all_qubits.difference(qubits)
    ordered_qubits = sorted(qubits) + sorted(ancilla_qubits)
    keep_indices = range(qubits)
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
    ]
    score = sum(
        len(qubits) ** 2 / len(create_ghz_state(qubits, 'a')) for qubits in qubit_choices
    )
    print(score)
