import cirq
from collections import deque

def bfs_tree(root, qset):
    parent = {root: None}
    order = [root]
    dist = {root: 0}
    dq = deque([root])
    while dq:
        q = dq.popleft()
        for n in sorted(q.neighbors()):
            if n in qset and n not in parent:
                parent[n] = q
                dist[n] = dist[q] + 1
                order.append(n)
                dq.append(n)
    return order, parent, max(dist.values())

# part a
def ghz_part_a(qubits) -> cirq.Circuit:
    qset = set(qubits)
    best = min((bfs_tree(q, qset) + (q,) for q in qset), key=lambda t: t[2])
    order, parent, _, root = best

    ops = [cirq.H(root)]
    for q in  order[1:]:
        p = parent[q]
        ops += [cirq.H(q), cirq.CZ(p, q), cirq.H(q)]
        return cirq.Circuit(ops)


def ghz_state_helper(num_qubits: int, low: int, high: int) -> cirq.Circuit:
    if num_qubits == 1:
        return cirq.Circuit()
    
    if num_qubits == -1:
        return -1; # error
    
    mid = low + (high - low) // 2
    circuit = cirq.Circuit(cirq.CZ(qubits[low], qubits[mid]), cirq.H(qubits[mid]))
    circuit += cirq.Circuit.zip(
        ghz_state_helper(mid - low, low, mid),
        ghz_state_helper(high - mid, mid, high),
    )
    if num_qubits % 2 == 1:
        circuit += cirq.CZ(qubits[low], qubits[high - 1])
    return circuit

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


qubits = cirq.LineQubit.range(5)

circuit = cirq.Circuit(cirq.H.on_each(*qubits)) + ghz_state_helper(len(qubits), 0, len(qubits)) + cirq.measure(*qubits, key="result")

print(circuit)
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1000)
print("Counts: ", result.histogram(key="result"))





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

def prob1b(qubits):
    qs = sorted(qubits)
    circuit = cirq.Circuit()
    for i in range(len(qubits) - 1):
        circuit.append(cirq.H(qs[i]))
    for i in range(len(qubits) - 1):
        circuit.append(cirq.CZ(qs[i], qs[i + 1]))
        circuit.append(cirq.H(qs[i + 1]))
    return circuit

def prob1c(qubits):
    circuit = cirq.Circuit(cirq.H(cirq.GridQubit(0,0)))
    row=len(set(q.row for q in qubits))
    col=len(set(q.col for q in qubits))
    measured = []

    for r in range(row):
        for c in range(col):
            if r==0 or c==0:
                measured.append(cirq.GridQubit(r,c))
    q0=measured[0]
    circuit.append(cirq.Moment(
        cirq.measure(q0, key='control'),
        cirq.measure(*measured[1:], key='others')
    ))
    for q in qubits:
        if q!=q0:
            circuit.append(cirq.X(q).with_classical_controls('control'))

    circuit.append(cirq.measure(*qubits, key="result"))
    return circuit