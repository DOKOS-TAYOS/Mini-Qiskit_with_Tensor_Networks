"""
Example of canonical circuits as templates to be used
"""

import numpy as np
from .qiskit import QuantumCircuit, QuantumRegister


def Deustch_Jozsa_circuit(n_input_qubits: int, oracle) -> QuantumCircuit:
    """
    Creates a Deutsch-Jozsa quantum circuit for n input qubits.

    Args:
        n_input_qubits: Number of input qubits for the oracle function f(x)
        oracle: Oracle gate implementing the function to be tested

    Returns:
        QuantumCircuit configured for the Deutsch-Jozsa algorithm
    """
    # Create quantum registers
    qx = QuantumRegister(n_input_qubits, name="qx")  # Input register
    qy = QuantumRegister(1, name="qy")  # Output register

    qc = QuantumCircuit(qx, qy, name="Deutsch")

    # Apply initial superposition
    qc.h(qx["all"])  # Hadamard on all input qubits
    qc.x(qy[0])  # Flip output qubit to |1⟩
    qc.h(qy[0])  # Hadamard on output qubit to create |−⟩ state

    # Apply the oracle
    oracle_qudits = [qc.qudit_dict[qudit_id] for qudit_id in qx["all"] + [qy[0]]]
    qc.append_operator(oracle, oracle_qudits)

    # Apply final Hadamard layer to input qubits
    qc.h(qx["all"])

    return qc


def Deustch_Jozsa_Algorithm(
    n_input_qubits: int, oracle, repr: str = "MPS", eps: float = 0
):
    """
    Executes the Deutsch-Jozsa algorithm to determine if a function is constant or balanced.

    Args:
        n_input_qubits: Number of input qubits
        oracle: Oracle gate implementing the function to be tested
        repr: Representation for contraction ('MPS' or other)
        eps: Error tolerance for MPS compression

    Returns:
        None (prints result)
    """
    # Check amplitude for output qubit in |0⟩ state
    qc = Deustch_Jozsa_circuit(n_input_qubits, oracle)
    amp0 = qc.check_state(int("0" * (n_input_qubits + 1), 2), repr, eps)

    # Check amplitude for output qubit in |1⟩ state
    qc = Deustch_Jozsa_circuit(n_input_qubits, oracle)
    amp1 = qc.check_state(int("1" + "0" * n_input_qubits, 2), repr, eps)

    # Calculate total amplitude of the all-zeros state in input register
    total_amplitude = np.sqrt(amp0**2 + amp1**2)

    # Determine if function is balanced or constant
    # If amplitude is ~0, the function is balanced; otherwise it's constant
    if np.round(total_amplitude, 6) == 0:
        print("Balanced function")
    else:
        print("Constant function")

    print(f"Amplitude of |0...0⟩ state: {total_amplitude}")
