import cirq

# part a
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


qubits = cirq.LineQubit.range(5)

circuit = cirq.Circuit(cirq.H.on_each(*qubits)) + ghz_state_helper(len(qubits), 0, len(qubits)) + cirq.measure(*qubits, key="result")

print(circuit)
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1000)
print("Counts: ", result.histogram(key="result"))
