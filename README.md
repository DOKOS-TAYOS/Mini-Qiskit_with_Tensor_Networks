# Mini-Qiskit with Tensor Networks

A miniature implementation of Qiskit's quantum circuit framework using tensor networks. This project demonstrates how quantum circuits can be represented and simulated efficiently using tensor network methods, providing both educational insight and practical quantum circuit simulation capabilities.

## Overview

This project implements a Qiskit-like quantum circuit interface from scratch using only tensor networks. It leverages the TensorNetwork library for efficient tensor contractions and provides support for both full statevector and Matrix Product State (MPS) representations.

### Key Features

- **Tensor Network Backend**: All quantum operations are represented and contracted using tensor networks
- **Qiskit-Compatible API**: Familiar interface for users of Qiskit
- **Multiple Representations**: Support for both statevector and MPS representations
- **MPS Compression**: Optional compression with configurable error tolerance for efficient simulation of larger circuits
- **Quantum Registers**: Support for named quantum registers similar to Qiskit
- **Visualization**: Circuit visualization using matplotlib
- **Quantum Algorithms**: Includes implementation of Deutsch-Jozsa algorithm


(See `requirements.txt` for complete dependency list)

## Quick Start

### Basic Circuit Creation

```python
from qiskit_from_scratch import QuantumCircuit

# Create a 3-qubit circuit
qc = QuantumCircuit(3, name='MyCircuit')

# Add gates
qc.h(0)           # Hadamard on qubit 0
qc.cx(0, 1)       # CNOT with control=0, target=1
qc.rz(np.pi/4, 2) # Z-rotation on qubit 2

# Get the statevector
statevector = qc.statevector()
```

### Using Quantum Registers

```python
from qiskit_from_scratch import QuantumCircuit, QuantumRegister

# Create quantum registers
qx = QuantumRegister(2, name='qx')
qy = QuantumRegister(1, name='qy')

# Create circuit with registers
qc = QuantumCircuit(qx, qy, name='RegisterCircuit')

# Apply gates using register notation
qc.h(qx['all'])   # Hadamard on all qubits in qx
qc.cx(qx[0], qy[0])  # CNOT between registers
```

### MPS Representation with Compression

```python
# Create and run circuit with MPS representation
qc = QuantumCircuit(10, name='LargeCircuit')
qc.h(list(range(10)))

# Contract using MPS with compression (eps = error tolerance)
mps = qc.contract(scheme='time', representation='MPS', eps=0.01)
```

### Computing Expectation Values

```python
# Create state preparation circuit
qc = QuantumCircuit(3, name='State')
qc.h(0)
qc.cx(0, 1)

# Create operator circuit
operator = QuantumCircuit(3, name='Observable')
operator.z(1)  # Z operator on qubit 1

# Convert operator to gate and compute expectation value
operator_gate = operator.to_gate()
expectation = qc.expected(operator_gate, representation='MPS', eps=0.1)
```

## Supported Gates

### Single-Qubit Gates
- **Pauli Gates**: `x()`, `y()`, `z()`
- **Hadamard**: `h()`
- **Rotation Gates**: `rx(angle)`, `ry(angle)`, `rz(angle)`

### Two-Qubit Gates
- **Controlled-X (CNOT)**: `cx(control, target)`
- **Controlled-Z**: `cz(control, target)`

### Multi-Qubit Gates
- **Toffoli (CCX)**: `ccx(control1, control2, target)`

### Custom Gates
- `append_gate_single()`: Add custom single-qubit gates
- `append_gate_control()`: Add custom controlled gates
- `append_gate_multi_control()`: Add custom multi-controlled gates
- `append_operator()`: Add arbitrary tensor network operators

## Architecture

### Core Classes

#### `TNCircuit`
Parent class that manages the tensor network representation of quantum circuits.

**Key Methods:**
- `append_operator(operator, qudits)`: Add operators to the circuit
- `contract(scheme, representation, eps)`: Contract the tensor network
  - `scheme`: 'time' (spatial not implemented)
  - `representation`: 'statevector' or 'MPS'
  - `eps`: MPS compression error tolerance

**Attributes:**
- `tensors`: 2D list of tensor nodes organized by [time][qudit]
- `n_qudits`: Number of qudits
- `dimension_qudits`: Dimension of each qudit (2 for qubits)
- `depth`: Circuit depth

#### `QuantumCircuit`
Inherits from `TNCircuit` and provides a Qiskit-like interface.

**Additional Methods:**
- Single and multi-qubit gate methods (h, x, y, z, rx, ry, rz, cx, cz, ccx)
- `statevector()`: Get the full statevector
- `expected(operator)`: Compute expectation value
- `check_state(state)`: Get amplitude of a specific basis state
- `to_gate()`: Convert circuit to a gate for use in other circuits
- `draw()`: Visualize the circuit

**Magic Methods:**
- `len(qc)`: Returns circuit depth
- `qc[i]`: Get all tensors at qudit i
- `print(qc)`: Display circuit diagram

#### `QuantumRegister`
Manages named collections of qudits.

**Features:**
- Named qudit access: `reg[0]`, `reg['all']`, `reg[(0,3)]`
- Integration with `QuantumCircuit`

## Implemented Algorithms

### Deutsch-Jozsa Algorithm

Complete implementation of the Deutsch-Jozsa algorithm for determining whether a function is constant or balanced.

```python
# Create a balanced oracle
oracle = QuantumCircuit(3, name='Oracle')
oracle.cx(1, 2)
oracle_gate = oracle.to_gate()

# Run Deutsch-Jozsa algorithm
Deustch_Jozsa_Algorithm(2, oracle_gate, repr='MPS', eps=0.5)
# Output: Balanced function
```

## Project Structure

```
Mini-Qiskit_with_Tensor_Networks/
│
├── qiskit_from_scratch.ipynb    # Main implementation and examples
├── auxiliary_functions.py        # Helper functions for tensor operations
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## How It Works

### Tensor Network Representation

Each qubit is represented as a tensor with multiple indices:
- **out**: Physical qubit dimension (connects forward in time)
- **in**: Connects to previous time step
- **up/down**: Connects to adjacent qubits (for multi-qubit gates)

Gates are represented as tensors that contract with the state tensors. The circuit evolves by systematically contracting these tensors according to the chosen scheme (time-based) and representation (statevector or MPS).

### Contraction Schemes

**Time-based Contraction**: 
- Contracts tensors layer by layer through time
- Maintains quantum state at each time step
- Supports both exact (statevector) and approximate (compressed MPS) representations

### MPS Compression

For large circuits, MPS representation with compression provides significant memory savings:
- Uses Tensor Train (TT) decomposition via tntorch
- Configurable error tolerance (eps parameter)
- Enables simulation of circuits with more qubits than full statevector methods

## Limitations

- Spatial contraction scheme not implemented
- Density operator representation not implemented
- PEPS (Projected Entangled Pair States) representation not implemented
- No measurement operations (classical registers)
- No noise models
- Limited gate set compared to full Qiskit

## Possible Enhancements

Potential areas for extension:
- Additional quantum gates (SWAP, phase gates, etc.)
- Measurement operations
- More quantum algorithms (Grover's, QFT, VQE)
- Density matrix representation
- Spatial contraction schemes
- Noise models and error mitigation
- Performance optimizations

## License

This project is for educational purposes, demonstrating the connection between quantum circuits and tensor networks.

## Acknowledgments

This implementation is inspired by Qiskit and uses the following libraries:
- TensorNetwork library for tensor operations
- tntorch for MPS compression
- PyTorch for tensor backend
- NumPy for numerical operations
- Matplotlib for visualization

## Author

Developed as an educational project to understand quantum circuit simulation using tensor network methods.

