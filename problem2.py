import itertools
import math
import time

import cirq
import cirq_google
import numpy as np
from typing import Literal

def create_w_state(
    num_qubits: int,
    cz_pauli_error_rate: dict[
        tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]
    ],
    sq_pauli_error_rate: dict[cirq.GridQubit, tuple[float, float]],
    zero_errors: dict[cirq.GridQubit, tuple[float, float]],
    one_errors: dict[cirq.GridQubit, tuple[float, float]],
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Create a Cirq circuit that produces a W state.

    Args:
        num_qubits: The number, N, of qubits in the W state.
        cz_pauli_error_rate: Measured CZ Pauli error rates and their statistical uncertainties.
        sq_pauli_error_rate: Measured single-qubit Pauli error rates and their statistical uncertainties.
        zero_errors: Measured parallel 0 -> 1 readout error rates and their statistical uncertainties.
        one_errors: Measured 1 -> 0 readout error rates and their statistical uncertainties.

    Returns:
        A Cirq circuit to produce W state and a set indicating which qubits the resulting state lives on.

    Raises:
        NotImplementedError: The requested num_qubits is not implemented.
    """

    circuit = cirq.Circuit()
    mid = sorted(sq_pauli_error_rate)[len(sq_pauli_error_rate) // 2]
    qubits = cirq.GridQubit.rect(1, num_qubits, top=mid.row, left=mid.col)
    qubits = sorted(qubits)
    circuit.append(cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0).on(min(qubits)))
    theta = 2 * math.acos((1/math.sqrt(num_qubits)))
    circuit.append(cirq.PhasedXZGate(x_exponent=theta/math.pi, z_exponent=0, axis_phase_exponent=0.5).on(qubits[1]))
    for q in range(len(qubits) - 2):
        theta = 2 * math.acos((1/math.sqrt(len(qubits) - q - 1)))
        circuit.append(cirq.PhasedXZGate(x_exponent=0.5, z_exponent=-1, axis_phase_exponent=-0.5).on(qubits[q]))
        circuit.append(cirq.CZ(qubits[q + 1], qubits[q]))
        circuit.append(cirq.PhasedXZGate(x_exponent=0.5, z_exponent=-1, axis_phase_exponent=-0.5).on(qubits[q]))
        circuit.append(cirq.PhasedXZGate(x_exponent=(theta / 2) / math.pi, z_exponent=0, axis_phase_exponent=0.5).on(qubits[q + 2]))
        circuit.append(cirq.CZ(qubits[q + 1], qubits[q + 2]))
        circuit.append(cirq.PhasedXZGate(x_exponent=-1 * (theta / 2) / math.pi, z_exponent=0, axis_phase_exponent=0.5).on(qubits[q + 2]))
        circuit.append(cirq.CZ(qubits[q + 1], qubits[q + 2]))
    circuit.append(cirq.PhasedXZGate(x_exponent=0.5, z_exponent=-1, axis_phase_exponent=-0.5).on(qubits[num_qubits - 2]))
    circuit.append(cirq.CZ(qubits[num_qubits - 1], qubits[num_qubits - 2]))
    circuit.append(cirq.PhasedXZGate(x_exponent=0.5, z_exponent=-1, axis_phase_exponent=-0.5).on(qubits[num_qubits - 2]))

    return circuit, set(qubits)

def add_depolarizing_noise(circuit, qubits, cz_pauli_error_rate, sq_pauli_error_rate):
    noisy = cirq.Circuit()
    for op in circuit.all_operations():
        noisy.append(op)
        if len(op.qubits) == 1:
            q = op.qubits[0]
            p, w = sq_pauli_error_rate[q]
            noisy.append(cirq.depolarize(p).on(q))
        elif len(op.qubits) == 2:
            a, b = op.qubits
            p, w = cz_pauli_error_rate.get((a, b)) or cz_pauli_error_rate[(b, a)]
            noisy.append(cirq.depolarize(p, n_qubits=2).on(a, b))
    return noisy


def create_w_statevector(num_qubits: int) -> np.ndarray:
    psi = np.zeros(2**num_qubits)
    normalization = 1 / np.sqrt(num_qubits)
    for qubit_number in range(num_qubits):
        psi[2**qubit_number] = normalization
    return psi


def test_circuit(circuit: cirq.Circuit, qubits: set[cirq.GridQubit], sampler = cirq.Simulator()):
    all_qubits = circuit.all_qubits()
    ancilla_qubits = all_qubits.difference(qubits)
    ordered_qubits = sorted(qubits) + sorted(ancilla_qubits)
    keep_indices = range(len(qubits))
    num_qubits = len(qubits)
    psi_circuit_full = sampler.simulate(circuit, qubit_order=ordered_qubits).final_state_vector
    psi_circuit = cirq.sub_state_vector(
        state_vector=psi_circuit_full,
        keep_indices=keep_indices
    ) if len(all_qubits) > len(qubits) else psi_circuit_full
    assert np.isclose(np.abs(psi_circuit @ create_w_statevector(num_qubits)) ** 2, 1.0)

def check_z_bitstring_distribution(bitstrings: np.ndarray) -> float:
    """Measure how close the observed Z-basis distribution of bitstrings is to
    the ideal one. The lower the score, the better.

    Args:
        bitstrings: An array of shape (n_shots, n_qubits) containing the n_shots Z-basis bitstrings.

    Returns:
        A measure of how close the distribution is to the ideal one.
    """
    # first, let's see how many bitstrings have one excitation
    expected_excitations = 1
    n_shots, n_qubits = bitstrings.shape
    measured_excitations = np.sum(bitstrings, axis=1)
    fraction_in_subspace = np.mean(measured_excitations == expected_excitations)

    # now, let's check how uniform the distribution is over states with 1 excitation
    chi2 = 0
    ideal_prob = 1 / n_qubits
    for q in range(n_qubits):
        ideal_state = np.zeros(n_qubits)
        ideal_state[q] = 1
        match_count = np.sum(np.all(bitstrings == ideal_state, axis=1))
        expected_count = n_shots * fraction_in_subspace * ideal_prob
        chi2 += (match_count - expected_count) ** 2 / expected_count

    return ideal_prob * chi2 / fraction_in_subspace


def get_correlator_average(bitstrings: np.ndarray, correlator_size: int) -> float:
    """Get the average of all weight-correlator_size correlators in `bitstrings`.

    Args:
        bitstrings: The input bitstrings, shape (n_shots, n_qubits).
        correlator_size: Compute all correlators of this weight.

    Returns:
        The average of all weight-k correlators.
    """
    correlator_sum = 0.0
    n_shots, n_qubits = bitstrings.shape
    for indices in itertools.combinations(range(n_qubits), correlator_size):
        correlator_sum += 1 - 2 * np.mean(np.sum(bitstrings[:, indices], axis=1) % 2)
    return correlator_sum / math.comb(n_qubits, correlator_size)


def check_x_bitstring_distribution(
    bitstrings: np.ndarray, max_correlator_size: int = 2
) -> float:
    """Check expected correlations in the X basis. The lower the score, the better.

    Args:
        bitstrings: An array of shape (n_shots, n_qubits) containing the n_shots X-basis bitstrings.
        max_correlator_size: The maximum weight of correlators to consider.

    Returns:
        A measure of how the correlations in the X-basis measurements compare to expectation.
    """
    score = 0.0
    n_shots, n_qubits = bitstrings.shape
    for correlator_size in range(1, max_correlator_size + 1):
        expected_correlator = 2 / n_qubits if correlator_size == 2 else 0.0
        measured_correlator = get_correlator_average(bitstrings, correlator_size)
        score += (expected_correlator - measured_correlator) ** 2 / (2 / n_qubits) ** 2
    return score


def overall_score(
    z_bitstrings: np.ndarray, x_bitstrings: np.ndarray, max_correlator_size: int = 2
) -> float:
    """An overall score that checks how the Z and X basis bitstrings compare to their ideal
    distributions. The higher the score, the better.

    Args:
        z_bitstrings: The bitstrings measured in the Z basis
        x_bitstrings: The bitstrings measured in the X basis
        max_correlator_size: The maximum weight of correlators to consider

    Returns:
        A measure of how similar the bitstrings are to their ideal distributions.
    """
    return 1 / (
        check_z_bitstring_distribution(z_bitstrings)
        + check_x_bitstring_distribution(x_bitstrings, max_correlator_size)
    )

def pick_single_qubit_dropouts(
    dropout_rate: float, rng: np.random.Generator = np.random.default_rng()
) -> set[cirq.GridQubit]:
    """Pick some qubits to be broken.

    Args:
        dropout_rate: The probability that a qubit is broken.
        rng: The pseudo-random number generator.

    Returns:
        The broken qubits.
    """
    all_qubits = np.array(list(cirq_google.Willow105.metadata.qubit_set))
    is_dropout = rng.random(len(all_qubits)) < dropout_rate
    return set(all_qubits[is_dropout])


def generate_qubit_error_rates(
    median: float,
    spread: float,
    dropouts: set[cirq.GridQubit],
    uncertainty: float,
    rng: np.random.Generator = np.random.default_rng(),
) -> dict[cirq.GridQubit, tuple[float, float]]:
    """Generate fake single-qubit error rates.

    Args:
        median: The median error rate.
        spread: The spread of the lognormal distribution
        dropouts: Qubits to exclude
        uncertainty: The uncertainty of the measured error rates.
        rng: The pseudo-random number generator

    Returns:
        A dictionary from qubits to the measured error rate and statistical uncertainty.
    """

    all_qubits = np.array(list(cirq_google.Willow105.metadata.qubit_set))
    is_working = np.array([q not in dropouts for q in all_qubits])
    working_qubits = all_qubits[is_working]
    error_rate = rng.lognormal(np.log(median), spread, len(working_qubits))
    return dict(
        zip(working_qubits, zip(error_rate, uncertainty * np.ones(len(working_qubits))))
    )


def generate_two_qubit_pauli_error_rates(
    median: float = 2.5e-3,
    spread: float = 0.5,
    dropout_rate: float = 0.01,
    uncertainty: float = 5e-4,
    rng: np.random.Generator = np.random.default_rng(),
    single_qubit_dropouts: set[cirq.GridQubit] = set(),
) -> dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]]:
    """Generate fake two-qubit error rates.

    Args:
        median: The median error rate.
        spread: The spread of the lognormal distribution
        dropout_rate: The probability of a broken coupler.
        uncertainty: The uncertainty of the measured error rates.
        rng: The pseudo-random number generator
        single_qubit_dropouts: The single-qubit dropouts to skip.

    Returns:
        A dictionary from qubits to the measured error rate and statistical uncertainty.
    """
    all_pairs = np.array(list(cirq_google.Willow105.metadata.qubit_pairs))
    no_qubit_dropouts = np.array(
        [np.all([q not in single_qubit_dropouts for q in pair]) for pair in all_pairs]
    )
    is_working = (rng.random(len(all_pairs)) > dropout_rate) * no_qubit_dropouts
    working_pairs = [tuple(pair) for pair in all_pairs[is_working]]
    error_rate = rng.lognormal(np.log(median), spread, len(working_pairs))
    return dict(
        zip(working_pairs, zip(error_rate, uncertainty * np.ones(len(working_pairs))))
    )

if __name__ == "__main__":
    single_qubit_dropouts = pick_single_qubit_dropouts(0.01)
    sq_pauli_error_rate = generate_qubit_error_rates(4e-4, 0.5, single_qubit_dropouts, 2e-4)
    cz_pauli_error_rate = generate_two_qubit_pauli_error_rates(
        single_qubit_dropouts=single_qubit_dropouts
    )
    zero_errors = generate_qubit_error_rates(5e-3, 0.5, single_qubit_dropouts, 2e-4)
    one_errors = generate_qubit_error_rates(1e-2, 0.5, single_qubit_dropouts, 3e-4)

    startTime=time.time()
    circuit, qubits = create_w_state(4, cz_pauli_error_rate, sq_pauli_error_rate, zero_errors, one_errors)
    test_circuit(circuit, qubits)
    print("state check passed")

    qs = sorted(qubits)
    z_bits = cirq.Simulator().run(circuit + [cirq.measure(*qs, key="m")], repetitions=5000).measurements["m"]
    x_bits = cirq.Simulator().run(circuit + [cirq.H.on_each(*qs), cirq.measure(*qs, key="m")], repetitions=5000).measurements["m"]
    rng = np.random.default_rng()
    z_bits, x_bits = z_bits.copy(), x_bits.copy()
    for bits in (z_bits, x_bits):
        for j, q in enumerate(qs):
            r = rng.random(len(bits))
            flip = np.where(bits[:, j] == 0, r < zero_errors[q][0], r < one_errors[q][0])
            bits[:, j] = np.where(flip, 1 - bits[:, j], bits[:, j])
    print("score (ideal):", overall_score(z_bits, x_bits))

    circuit = add_depolarizing_noise(circuit, sorted(qubits), cz_pauli_error_rate, sq_pauli_error_rate)
    elapsed=time.time() - startTime
    if elapsed>60:
        raise NotImplementedError(f'Execution took {elapsed:.2f}s, exceeding the 60 second limit.')

    
    qs = sorted(qubits)
    z_bits = cirq.Simulator().run(circuit + [cirq.measure(*qs, key="m")], repetitions=5000).measurements["m"]
    x_bits = cirq.Simulator().run(circuit + [cirq.H.on_each(*qs), cirq.measure(*qs, key="m")], repetitions=5000).measurements["m"]
    z_bits, x_bits = z_bits.copy(), x_bits.copy()
    for bits in  (z_bits, x_bits):
        for j, q in enumerate(qs):
            r = rng.random(len(bits))
            flip = np.where(bits[:, j] == 0, r < zero_errors[q][0], r < one_errors[q][0])
            bits[:, j] = np.where(flip, 1 - bits[:, j], bits[:, j])
    print("score (error):", overall_score(z_bits, x_bits))


    