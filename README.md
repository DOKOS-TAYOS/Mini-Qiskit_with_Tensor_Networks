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

Simply install the package and follow the examples in the [examples](./examples) folder.

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

