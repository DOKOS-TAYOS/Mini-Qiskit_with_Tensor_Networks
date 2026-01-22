"""
Class that creates the QuantumRegister. This will be basically a dictionary where we will have pairs qudit: qudit name, with the qudit names composed of the register name(qudit number in the register). We have added some methods to make it more useful than a dictionary.

Magic methods:
- call: returns the entire dictionary
- [items]: returns the keys corresponding to the input items.
    - If items is an integer, returns only one key.
    - If it is a tuple of two integers, returns all keys corresponding to the qudits in the range between both (first included, second excluded).
    - If it is a list, returns the keys corresponding to each qudit in that list.
    - If it is 'all', returns a list with all keys.
- len: returns the number of qudits in the register.
"""

import numpy as np
import tensornetwork as tn
import matplotlib.pyplot as plt

from .base import TNCircuit


class QuantumRegister:
    """
    A quantum register that manages a collection of qudits with named identifiers.

    The register creates a mapping between qudit indices and their string identifiers
    in the format 'name(index)'.

    Attributes:
        name (str): The name of the register
        n_qudits (int): The number of qudits in the register
        register_dict (dict): Internal mapping from indices to qudit names
    """

    def __init__(self, n_qudits: int, name: str):
        """
        Initialize a quantum register.

        Args:
            n_qudits (int): Number of qudits in the register (must be positive)
            name (str): Name identifier for the register (must be non-empty)

        Raises:
            ValueError: If n_qudits is not positive or name is empty
        """
        if n_qudits <= 0:
            raise ValueError(f"Number of qudits must be positive, got {n_qudits}")
        if not name or not isinstance(name, str):
            raise ValueError("Register name must be a non-empty string")

        self.name = name
        self.n_qudits = n_qudits
        # Create qudit identifiers in format 'name(index)'
        self.register_dict = {qudit: f"{name}({qudit})" for qudit in range(n_qudits)}

    def __call__(self) -> dict:
        """Return the complete qudit dictionary."""
        return self.register_dict

    def __getitem__(
        self, items: int | tuple[int, int] | list[int] | str
    ) -> str | list[str]:
        """
        Access qudit names by index, range, list, or 'all'.

        Args:
            items: Can be:
                - int: Returns single qudit name
                - tuple(int, int): Returns list of names in range [start, end)
                - list[int]: Returns list of names for specified indices
                - 'all': Returns list of all qudit names

        Returns:
            str or list[str]: Qudit name(s) corresponding to the input

        Raises:
            KeyError: If index is out of range
            ValueError: If tuple range is invalid
        """
        if isinstance(items, list):
            # List of specific qudit indices
            try:
                return [self.register_dict[idx] for idx in items]
            except KeyError as e:
                raise KeyError(f"Qudit index {e} out of range [0, {self.n_qudits})")
        elif items == "all":
            # All qudits in order
            return [self.register_dict[idx] for idx in range(self.n_qudits)]
        elif isinstance(items, tuple):
            # Range of qudits [start, end)
            if len(items) != 2:
                raise ValueError(
                    f"Tuple must have exactly 2 elements, got {len(items)}"
                )
            start, end = items
            if not (0 <= start < end <= self.n_qudits):
                raise ValueError(
                    f"Invalid range ({start}, {end}) for register with {self.n_qudits} qudits"
                )
            return [self.register_dict[idx] for idx in range(start, end)]
        else:
            # Single qudit index
            try:
                return self.register_dict[items]
            except KeyError:
                raise KeyError(f"Qudit index {items} out of range [0, {self.n_qudits})")

    def __len__(self) -> int:
        """Return the number of qudits in the register."""
        return self.n_qudits


class QuantumCircuit(TNCircuit):
    """
    Quantum circuit implementation using tensor networks, compatible with Qiskit-style API.

    Inherits from TNCircuit and provides a Qiskit-like interface for building quantum circuits
    using quantum registers or direct qubit counts.

    Args:
        *regs: Either an integer specifying the number of qubits, or one or more QuantumRegister objects
        name: Name of the circuit (default: 'Circuit')
    """

    def __init__(self, *regs, name: str = "Circuit"):
        self.name = name

        if isinstance(regs[0], int):
            # Direct qubit count specification
            n_qudits = regs[0]
            # Identity mapping: qubit index maps to itself
            self.qudit_dict = {i: i for i in range(n_qudits)}
        else:
            # Quantum register(s) specification
            n_qudits = 0
            self.qudit_dict = {}
            qudit_index = 0

            for reg in regs:
                n_qudits += len(reg)
                # Map each register key to its corresponding qudit index
                for key in reg():
                    self.qudit_dict[reg[key]] = qudit_index
                    qudit_index += 1

        # Initialize the parent TNCircuit with 2-dimensional qudits (qubits)
        super().__init__(n_qudits, qudits_dimension=2)

    # -------------------------------------------------------------------------
    # Magic methods for convenient circuit operations
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        """Returns the depth of the circuit."""
        return self.depth

    def __getitem__(self, key: int | str) -> list:
        """
        Returns all tensors applied to the specified qudit.

        Args:
            key: Qudit identifier (index or register key)

        Returns:
            List of tensors applied to the qudit across all layers
        """
        return [self.tensors[layer][key] for layer in range(self.depth)]

    # -------------------------------------------------------------------------
    # Quantum gate operations
    # -------------------------------------------------------------------------

    def append_gate_single(
        self, tensor: np.ndarray, name: str, positions: list
    ) -> None:
        """
        Appends a single-qudit gate to the circuit at specified positions.

        Args:
            tensor: The gate's tensor representation (2x2 matrix for qubits)
            name: Name/label for the gate
            positions: List of qudit positions where the gate should be applied
        """
        # Create tensor network nodes for each gate instance
        layer = [
            [
                tn.Node(tensor, name=name, axis_names=["in", "out"])
                for _ in range(len(positions))
            ]
        ]

        # Add the gates to the circuit at the specified positions
        self.append_operator(layer, positions)

    def x(self, qudits):
        """Apply Pauli-X gate to specified qudits."""
        if isinstance(qudits, (int, str)):
            qudits = [qudits]

        # Translate keys to qudit positions
        positions = [self.qudit_dict[key] for key in qudits]

        # Pauli-X gate tensor
        tensor = np.array([[0, 1], [1, 0]], dtype=complex)
        name = "X"

        self.append_gate_single(tensor, name, positions)

    def h(self, qudits):
        """Apply Hadamard gate to specified qudits."""
        if isinstance(qudits, (int, str)):
            qudits = [qudits]

        # Translate keys to qudit positions
        positions = [self.qudit_dict[key] for key in qudits]

        # Hadamard gate tensor
        tensor = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        name = "H"

        self.append_gate_single(tensor, name, positions)

    def z(self, qudits):
        """Apply Pauli-Z gate to specified qudits."""
        if isinstance(qudits, (int, str)):
            qudits = [qudits]

        # Translate keys to qudit positions
        positions = [self.qudit_dict[key] for key in qudits]

        # Pauli-Z gate tensor
        tensor = np.array([[1, 0], [0, -1]], dtype=complex)
        name = "Z"

        self.append_gate_single(tensor, name, positions)

    def y(self, qudits):
        """Apply Pauli-Y gate to specified qudits."""
        if isinstance(qudits, (int, str)):
            qudits = [qudits]

        # Translate keys to qudit positions
        positions = [self.qudit_dict[key] for key in qudits]

        # Pauli-Y gate tensor
        tensor = np.array([[0, -1j], [1j, 0]], dtype=complex)
        name = "Y"

        self.append_gate_single(tensor, name, positions)

    def rx(self, theta, qudits):
        """Apply rotation around X-axis to specified qudits."""
        if isinstance(qudits, (int, str)):
            qudits = [qudits]

        # Translate keys to qudit positions
        positions = [self.qudit_dict[key] for key in qudits]

        # RX gate tensor
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        tensor = np.array(
            [[cos_half, -1j * sin_half], [-1j * sin_half, cos_half]], dtype=complex
        )
        name = f"RX({theta:.3f})"

        self.append_gate_single(tensor, name, positions)

    def ry(self, theta, qudits):
        """Apply rotation around Y-axis to specified qudits."""
        if isinstance(qudits, (int, str)):
            qudits = [qudits]

        # Translate keys to qudit positions
        positions = [self.qudit_dict[key] for key in qudits]

        # RY gate tensor
        cos_half = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        tensor = np.array([[cos_half, -sin_half], [sin_half, cos_half]], dtype=complex)
        name = f"RY({theta:.3f})"

        self.append_gate_single(tensor, name, positions)

    def rz(self, theta, qudits):
        """Apply rotation around Z-axis to specified qudits."""
        if isinstance(qudits, (int, str)):
            qudits = [qudits]

        # Translate keys to qudit positions
        positions = [self.qudit_dict[key] for key in qudits]

        # RZ gate tensor
        tensor = np.array(
            [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
        )
        name = f"RZ({theta:.3f})"

        self.append_gate_single(tensor, name, positions)

    # ...........................................................................
    def append_gate_control(self, tensor, name, positions_control, positions_target):
        """
        Add a controlled gate with given tensor and name at specified positions.

        Args:
            tensor: Gate tensor to apply (controlled operation)
            name: Name of the gate
            positions_control: List of control qudit positions
            positions_target: List of target qudit positions
        """
        # Number of gates to apply
        n_gates = len(positions_control)
        min_qudit = min(min(positions_control), min(positions_target))
        max_qudit = max(max(positions_control), max(positions_target))
        total_qudits = max_qudit - min_qudit + 1

        # New list of positions to include all qudits in range
        positions = list(np.arange(min_qudit, max_qudit + 1))

        # List to store layers - could optimize by placing multiple gates in same layer
        total_layers = [[None for _ in range(total_qudits)] for _ in range(n_gates)]

        # Create control tensors
        ctrl_tensor_down = np.zeros((2, 2, 2), dtype=complex)
        ctrl_tensor_up = np.zeros((2, 2, 2), dtype=complex)
        pass_tensor = np.zeros((2, 2, 2, 2), dtype=complex)

        # Set up control and pass tensors
        for i in range(2):  # For each signal value
            ctrl_tensor_down[i, i, i] = 1
            ctrl_tensor_up[i, i, i] = 1
            for j in range(2):  # For each input value
                pass_tensor[j, j, i, i] = 1

        # Build each gate layer
        for gate in range(n_gates):
            difference = positions_control[gate] - positions_target[gate]

            if difference > 0:  # Control is below target
                total_layers[gate][positions_target[gate] - min_qudit] = tn.Node(
                    tensor, name=name, axis_names=["in", "out", "down"]
                )
                total_layers[gate][positions_control[gate] - min_qudit] = tn.Node(
                    ctrl_tensor_up, name="Ctrl_Up", axis_names=["in", "out", "up"]
                )
                # Add pass tensors for intermediate qudits
                for intermediate in range(1, difference):
                    total_layers[gate][
                        positions_target[gate] + intermediate - min_qudit
                    ] = tn.Node(
                        pass_tensor, name="Pass", axis_names=["in", "out", "up", "down"]
                    )

            elif difference < 0:  # Control is above target
                total_layers[gate][positions_target[gate] - min_qudit] = tn.Node(
                    tensor, name=name, axis_names=["in", "out", "up"]
                )
                total_layers[gate][positions_control[gate] - min_qudit] = tn.Node(
                    ctrl_tensor_down, name="Ctrl_Down", axis_names=["in", "out", "down"]
                )
                # Add pass tensors for intermediate qudits
                for intermediate in range(1, -difference):
                    total_layers[gate][
                        positions_target[gate] - intermediate - min_qudit
                    ] = tn.Node(
                        pass_tensor, name="Pass", axis_names=["in", "out", "up", "down"]
                    )

        # Add to network with their positions
        self.append_operator(total_layers, positions)

    def cx(self, qudit_control, qudit_target):
        """Apply controlled-X (CNOT) gate(s).

        Args:
            qudit_control: Control qudit(s) - can be int, str, or list
            qudit_target: Target qudit(s) - can be int, str, or list
        """
        # Ensure inputs are lists
        if isinstance(qudit_control, (int, str)):
            qudit_control = [qudit_control]
            qudit_target = [qudit_target]

        # Translate keys to qudit positions
        positions_control = [self.qudit_dict[key] for key in qudit_control]
        positions_target = [self.qudit_dict[key] for key in qudit_target]

        # Create controlled-X tensor
        x_gate = np.array([[0, 1], [1, 0]], dtype=complex)
        c_tensor = np.zeros((2, 2, 2), dtype=complex)

        c_tensor[:, :, 0] = np.eye(2, dtype=complex)  # Identity when control is |0⟩
        c_tensor[:, :, 1] = x_gate  # Apply X when control is |1⟩

        name = "C-X"

        # Add the controlled gate to the network
        self.append_gate_control(c_tensor, name, positions_control, positions_target)

    def cz(self, qudit_control, qudit_target):
        """Apply controlled-Z gate(s).

        Args:
            qudit_control: Control qudit(s) - can be int, str, or list
            qudit_target: Target qudit(s) - can be int, str, or list
        """
        # Ensure inputs are lists
        if isinstance(qudit_control, (int, str)):
            qudit_control = [qudit_control]
            qudit_target = [qudit_target]

        # Translate keys to qudit positions
        positions_control = [self.qudit_dict[key] for key in qudit_control]
        positions_target = [self.qudit_dict[key] for key in qudit_target]

        # Create controlled-Z tensor
        z_gate = np.array([[1, 0], [0, -1]], dtype=complex)
        c_tensor = np.zeros((2, 2, 2), dtype=complex)

        c_tensor[:, :, 0] = np.eye(2, dtype=complex)  # Identity when control is |0⟩
        c_tensor[:, :, 1] = z_gate  # Apply Z when control is |1⟩

        name = "C-Z"

        # Add the controlled gate to the network
        self.append_gate_control(c_tensor, name, positions_control, positions_target)

    # ...........................................................................
    def append_gate_multi_control(
        self, tensor, name, positions_control, positions_target
    ):
        """Add a multi-controlled gate to the quantum circuit.

        This method adds a gate controlled by multiple qudits. The control signal
        propagates through intermediate qudits using auxiliary tensors.

        Args:
            tensor: The gate tensor to apply when all controls are |1⟩
            name: Name identifier for the gate
            positions_control: List of control qudit positions
            positions_target: Target qudit position (single qudit only)
        """
        # Number of controls and range of affected qudits
        n_controls = len(positions_control)
        min_qudit = min(min(positions_control), positions_target)
        max_qudit = max(max(positions_control), positions_target)
        total_qudits = max_qudit - min_qudit + 1

        # Create list of all positions in the range
        positions = list(np.arange(min_qudit, max_qudit + 1))

        # Define control signal propagation tensors
        # ctrl_tensor: Initiates control signal (identity on input, outputs signal)
        ctrl_tensor_down = np.zeros((2, 2, 2), dtype=complex)
        ctrl_tensor_up = np.zeros((2, 2, 2), dtype=complex)

        # cctrl_tensor: Propagates control signal through intermediate qudits
        cctrl_tensor_up = np.zeros((2, 2, 2, 2), dtype=complex)
        cctrl_tensor_down = np.zeros((2, 2, 2, 2), dtype=complex)

        # pass_tensor: Allows signal to pass through uninvolved qudits
        pass_tensor = np.zeros((2, 2, 2, 2), dtype=complex)

        # Initialize control tensors
        for i in range(2):
            ctrl_tensor_down[i, i, i] = 1  # Output signal equals input state
            ctrl_tensor_up[i, i, i] = 1
            for j in range(2):
                pass_tensor[j, j, i, i] = 1  # Pass signal through unchanged

        # Initialize conditional control tensors
        for j in range(2):
            # Signal 0: pass through identity
            cctrl_tensor_up[j, j, 0, 0] = 1
            cctrl_tensor_down[j, j, 0, 0] = 1
            # Signal 1: AND with input state
            cctrl_tensor_up[j, j, 1, j] = 1
            cctrl_tensor_down[j, j, j, 1] = 1

        # Initialize layer with pass-through tensors
        total_layers = [
            [
                tn.Node(
                    pass_tensor, name="Pass", axis_names=["in", "out", "up", "down"]
                )
                for _ in range(total_qudits)
            ]
        ]

        # Case 1: All controls below target
        if min_qudit == positions_target:
            # Create controlled tensor (signal from below)
            c_tensor = np.zeros((2, 2, 2), dtype=complex)
            c_tensor[:, :, 0] = np.eye(2, dtype=complex)  # Identity when signal is 0
            c_tensor[:, :, 1] = tensor  # Apply gate when signal is 1

            total_layers[0][positions_target - min_qudit] = tn.Node(
                c_tensor, name=name, axis_names=["in", "out", "down"]
            )

            for ctrl in range(n_controls):
                idx = positions_control[ctrl] - min_qudit
                if positions_control[ctrl] == max_qudit:  # Bottommost control
                    total_layers[0][idx] = tn.Node(
                        ctrl_tensor_up, name="Ctrl_Up", axis_names=["in", "out", "up"]
                    )
                else:  # Intermediate control
                    total_layers[0][idx] = tn.Node(
                        cctrl_tensor_up,
                        name="cCtrl_Up",
                        axis_names=["in", "out", "up", "down"],
                    )

        # Case 2: All controls above target
        elif max_qudit == positions_target:
            # Create controlled tensor (signal from above)
            c_tensor = np.zeros((2, 2, 2), dtype=complex)
            c_tensor[:, :, 0] = np.eye(2, dtype=complex)  # Identity when signal is 0
            c_tensor[:, :, 1] = tensor  # Apply gate when signal is 1

            total_layers[0][positions_target - min_qudit] = tn.Node(
                c_tensor, name=name, axis_names=["in", "out", "up"]
            )

            for ctrl in range(n_controls):
                idx = positions_control[ctrl] - min_qudit
                if positions_control[ctrl] == min_qudit:  # Topmost control
                    total_layers[0][idx] = tn.Node(
                        ctrl_tensor_down,
                        name="Ctrl_Down",
                        axis_names=["in", "out", "down"],
                    )
                else:  # Intermediate control
                    total_layers[0][idx] = tn.Node(
                        cctrl_tensor_down,
                        name="cCtrl_Down",
                        axis_names=["in", "out", "up", "down"],
                    )

        # Case 3: Target in the middle (controls on both sides)
        else:
            # Create controlled tensor (signals from both sides)
            c_tensor = np.zeros((2, 2, 2, 2), dtype=complex)
            c_tensor[:, :, 0, 0] = np.eye(
                2, dtype=complex
            )  # Identity if either signal is 0
            c_tensor[:, :, 1, 0] = np.eye(2, dtype=complex)
            c_tensor[:, :, 0, 1] = np.eye(2, dtype=complex)
            c_tensor[:, :, 1, 1] = tensor  # Apply gate only when both signals are 1

            total_layers[0][positions_target - min_qudit] = tn.Node(
                c_tensor, name=name, axis_names=["in", "out", "up", "down"]
            )

            for ctrl in range(n_controls):
                idx = positions_control[ctrl] - min_qudit

                if positions_control[ctrl] == min_qudit:  # Topmost control
                    total_layers[0][idx] = tn.Node(
                        ctrl_tensor_down,
                        name="Ctrl_Down",
                        axis_names=["in", "out", "down"],
                    )
                elif positions_control[ctrl] == max_qudit:  # Bottommost control
                    total_layers[0][idx] = tn.Node(
                        ctrl_tensor_up, name="Ctrl_Up", axis_names=["in", "out", "up"]
                    )
                else:  # Intermediate control
                    if positions_control[ctrl] < positions_target:  # Above target
                        total_layers[0][idx] = tn.Node(
                            cctrl_tensor_down,
                            name="cCtrl_Down",
                            axis_names=["in", "out", "up", "down"],
                        )
                    else:  # Below target
                        total_layers[0][idx] = tn.Node(
                            cctrl_tensor_up,
                            name="cCtrl_Up",
                            axis_names=["in", "out", "up", "down"],
                        )

        # Add the layer to the circuit
        self.append_operator(total_layers, positions)

    def ccx(self, qudit_control1, qudit_control2, qudit_target):
        """
        Apply a Toffoli (CCX) gate to the circuit.

        Args:
            qudit_control1: First control qudit identifier
            qudit_control2: Second control qudit identifier
            qudit_target: Target qudit identifier
        """
        # Translate qudit identifiers to positions
        positions_control1 = self.qudit_dict[qudit_control1]
        positions_control2 = self.qudit_dict[qudit_control2]
        positions_target = self.qudit_dict[qudit_target]

        # Pauli-X gate tensor (NOT gate)
        x_tensor = np.array([[0, 1], [1, 0]], dtype=complex)

        # Apply multi-controlled gate
        self.append_gate_multi_control(
            x_tensor, "CC-X", [positions_control1, positions_control2], positions_target
        )

    # ---------------------------------------------------------------------------
    # Calculators
    def state_vector(self) -> np.ndarray:
        """
        Returns the state vector without storing it.

        Returns:
            np.ndarray: The state vector as a complex numpy array
        """
        return self.contract(scheme="time", representation="statevector", eps=0)

    def expected(
        self, operator, representation: str = "statevector", eps: float = 0
    ) -> complex:
        """
        Calculate the expected value of an operator.

        Args:
            operator: The operator to measure
            representation: Contraction representation ('statevector' or 'MPS')
            eps: Compression tolerance for MPS representation

        Returns:
            float: The expected value of the operator
        """
        # Create the conjugate transpose of the current state
        copy_tensor_list = [
            [None for _ in range(self.n_qudits)] for __ in range(self.depth)
        ]

        for i_layer, layer in enumerate(self.tensors[::-1]):  # Reverse order
            for i_qudit, tensor in enumerate(layer):
                if tensor is not None:
                    # Copy tensor with conjugate transpose
                    original_axis_names = tensor.axis_names
                    conj_tensor = np.conj(tensor.tensor)

                    if "out" in original_axis_names and "in" in original_axis_names:
                        # Swap 'in' and 'out' axes for transpose
                        conj_tensor = np.moveaxis(conj_tensor, 0, 1)
                        copy_tensor_list[i_layer][i_qudit] = tn.Node(
                            conj_tensor,
                            name=tensor.name,
                            axis_names=original_axis_names,
                        )
                    elif "in" in original_axis_names:  # No 'out' axis
                        axis_names = ["out"] + original_axis_names[1:]
                        copy_tensor_list[i_layer][i_qudit] = tn.Node(
                            conj_tensor, name=tensor.name, axis_names=axis_names
                        )
                    elif "out" in original_axis_names:  # No 'in' axis
                        axis_names = ["in"] + original_axis_names[1:]
                        copy_tensor_list[i_layer][i_qudit] = tn.Node(
                            conj_tensor, name=tensor.name, axis_names=axis_names
                        )
                    else:  # Neither 'in' nor 'out'
                        copy_tensor_list[i_layer][i_qudit] = tn.Node(
                            conj_tensor,
                            name=tensor.name,
                            axis_names=original_axis_names,
                        )

        # Add the operator to the tensor network
        self.append_operator(operator, list(np.arange(self.n_qudits)))

        # Add the conjugate copy
        self.append_operator(copy_tensor_list, list(np.arange(self.n_qudits)))

        # Contract the entire network
        result = self.contract(scheme="time", representation=representation, eps=eps)

        # For MPS representation, contract the MPS without output indices
        if representation == "MPS":
            vector = result[0].tensor.flatten()

            for qudit in range(1, self.n_qudits - 1):
                current_tensor = result[qudit].tensor

                if eps == 0:
                    # No compression - no 'out' axis
                    if "down" in result[0].axis_names:  # Vector
                        if "down" in result[qudit].axis_names:  # Matrix
                            vector = vector @ np.reshape(
                                current_tensor,
                                (current_tensor.shape[0], current_tensor.shape[1]),
                            )
                        else:  # Vector
                            vector = vector @ current_tensor.flatten()
                    else:  # Scalar
                        vector = vector * current_tensor.flatten()
                else:
                    # With compression - has 'out' axis
                    if "down" in result[0].axis_names:  # Vector
                        if "down" in result[qudit].axis_names:  # Matrix
                            vector = vector @ np.reshape(
                                current_tensor,
                                (current_tensor.shape[1], current_tensor.shape[2]),
                            )
                        else:  # Vector
                            vector = vector @ current_tensor.flatten()
                    else:  # Scalar
                        vector = vector * current_tensor.flatten()

            # Contract with the last qudit
            if self.n_qudits > 1:
                result = vector @ result[-1].tensor.flatten()
            else:
                result = vector

            return result
        else:
            return result.tensor.flatten()[0]

    def check_state(
        self, state: int, representation: str = "statevector", eps: float = 0
    ) -> complex:
        """
        Check the amplitude of a specific computational basis state.

        Args:
            state: Integer representing the computational basis state
            representation: 'statevector' or 'MPS'
            eps: Compression tolerance (0 for no compression)

        Returns:
            Complex amplitude of the specified state
        """
        # Convert state to binary representation
        bin_state = bin(state)[2:]
        bin_state = ("0" * (self.n_qudits - len(bin_state)) + bin_state)[::-1]

        # Create bra vectors for each qubit
        layer = [
            [
                tn.Node(np.array([1, 0], dtype=complex), name="<0|", axis_names=["in"])
                if bin_state[i] == "0"
                else tn.Node(
                    np.array([0, 1], dtype=complex), name="<1|", axis_names=["in"]
                )
                for i in range(self.n_qudits)
            ]
        ]

        self.append_operator(layer, list(np.arange(self.n_qudits)))

        result = self.contract(scheme="time", representation=representation, eps=eps)

        # For MPS representation, contract the MPS without output indices
        if representation == "MPS":
            # Flatten first tensor to vector
            vector = result[0].tensor.flatten()

            for qudit in range(1, self.n_qudits - 1):
                current_tensor = result[qudit].tensor

                if eps == 0:
                    # No compression - no 'out' axis
                    if "down" in result[0].axis_names:  # Vector
                        if "down" in result[qudit].axis_names:  # Matrix
                            vector = vector @ np.reshape(
                                current_tensor,
                                (current_tensor.shape[0], current_tensor.shape[1]),
                            )
                        else:  # Vector
                            vector = vector @ current_tensor.flatten()
                    else:  # Scalar
                        vector = vector * current_tensor.flatten()
                else:
                    # With compression - has 'out' axis
                    if "down" in result[0].axis_names:  # Vector
                        if "down" in result[qudit].axis_names:  # Matrix
                            vector = vector @ np.reshape(
                                current_tensor,
                                (current_tensor.shape[1], current_tensor.shape[2]),
                            )
                        else:  # Vector
                            vector = vector @ current_tensor.flatten()
                    else:  # Scalar
                        vector = vector * current_tensor.flatten()

            # Contract with the last qudit
            if self.n_qudits > 1:
                result = vector @ result[-1].tensor.flatten()
            else:
                result = vector

            return result
        else:
            return result.tensor.flatten()[0]

    # ---------------------------------------------------------------------------
    def to_gate(self) -> list:
        """
        Returns the circuit layers as a gate.

        Returns:
            list: Circuit layers excluding the initial state layer
        """
        return self.tensors[1:]

    def __repr__(self):
        """
        Visualizes the quantum circuit using matplotlib.

        Creates a graphical representation of the circuit showing:
        - Horizontal lines for each qudit
        - Boxes for gates/operators with their names
        - Vertical connections for multi-qudit gates

        Returns:
            str: Empty string (visualization is shown via matplotlib)
        """
        # Create figure with size proportional to circuit dimensions
        plt.figure(figsize=(self.depth * 0.5 + 2, self.n_qudits * 0.5 + 2))

        # Remove ticks and axis
        plt.xticks([])
        plt.yticks([])
        plt.axis("off")

        # Draw timeline for each qudit
        for qudit in range(self.n_qudits):
            plt.plot([0, self.depth - 1 + 1], [qudit, qudit], "w-")

        # Draw gates and operators
        for layer in range(self.depth):
            for qudit in range(self.n_qudits):
                tensor = self.tensors[layer][qudit]

                if tensor is not None:
                    y_pos = self.n_qudits - 1 - qudit

                    # Draw connections for multi-qudit gates
                    if "up" in tensor.axis_names:
                        plt.plot([layer, layer], [y_pos, y_pos + 0.7], "w-")
                    elif "down" in tensor.axis_names:
                        plt.plot([layer, layer], [y_pos, y_pos - 0.7], "w-")

                    # Check if gate is Pass
                    gate_name = tensor.name
                    if gate_name == "Pass":
                        # Draw small circle for Pass gate
                        if "down" in tensor.axis_names:
                            plt.plot([layer, layer], [y_pos, y_pos - 0.7], "w-")
                        plt.plot(layer, y_pos, "s", markersize=10, color="white")
                        plt.text(
                            layer,
                            y_pos,
                            gate_name,
                            fontsize=4,
                            verticalalignment="center",
                            horizontalalignment="center",
                            color="black",
                        )
                    else:
                        # Determine marker shape based on gate name
                        if "Ctrl_" in gate_name:
                            marker = "o"  # Circle for control gates
                            fontsize = 6
                        else:
                            marker = "s"  # Square for other gates
                            fontsize = 8

                        # Draw gate box or circle
                        plt.plot(layer, y_pos, marker, markersize=30)

                        # Add gate name
                        if len(gate_name) > 8:
                            # Split into two lines for long names
                            mid = len(gate_name) // 2
                            gate_text = f"{gate_name[:mid]}\n{gate_name[mid:]}"
                        else:
                            gate_text = gate_name

                        plt.text(
                            layer,
                            y_pos,
                            gate_text,
                            fontsize=fontsize,
                            verticalalignment="center",
                            horizontalalignment="center",
                            color="black",
                        )

        plt.show()
        return ""
