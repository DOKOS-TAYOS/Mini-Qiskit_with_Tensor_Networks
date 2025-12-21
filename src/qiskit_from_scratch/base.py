"""
This is the parent class that will store the tensors, perform general contractions and compressions.

To initialize the object, we must pass the following arguments:

- n_qudits: number of 'qudits' in the system.
- dimension_qudits: dimension of the qudits. For qubits, set to 2.

The methods it has are the following:
- append_operator: receives an operator as a list of lists of tensors (to include operators with more than one layer)
and the series of qubits to which it should be applied. The method adds these tensor layers at the first available time
for all involved qudits, creating a new time if necessary. Modifies self.tensors.
Input:

    - operator: list of lists of tensors to add to the circuit.
    - qudits:   list of qudits for each element of the operator.

- contract: receives a scheme from among `'time', 'spatial'` (although spatial is not implemented for the reasons
mentioned in the document) and a representation style from among `'statevector', 'MPS', 'density operator', 'PEPS'`
(although for simplicity only statevector and MPS are implemented). This method contracts the tensor network along
the line received by the scheme, and maintains the tensor shape in the indicated representation. For example, if we
select time and MPS, at each step we will have the MPS form for that time step, contracting each 'qudit' only with
its corresponding operations. Additionally, we can add an eps that will tell us the maximum error allowed in the
MPS compression of tntorch (only available for MPS with real numbers). If eps=0, then we do not approximate. Input:
    - scheme: contraction scheme.
    - representation: type of state representation.
    - eps: maximum error for compression.

Its attributes are:
- tensors: list of lists of tensors in Tensornetwork Node format. There are two indices to find a tensor.
The first tells us the time step of that tensor and the second tells us the qudit where it is located. That is,
`tensors[time][qudit]=Node`. If there is no operation at that (time, qudit) pair, there will be a None at that position.
The axes will always respect this order:
`'in','out','up','down'`, these being the indices that connect towards 'before', 'after', 'above' and 'below' in the quantum circuit.
- n_qudits: number of qudits.
- dimension_qudits: dimension of the qudits.
- depth: depth of the circuit.
- current_MPS: stores the MPS of the tensor network, in case it is to be used later for extended methods
(not implemented as they fall outside the scope of the task).
"""

import numpy as np
import tensornetwork as tn

from .utils import process_axes, TT_State_Compression


class TNCircuit:
    """Parent class for storing tensors and performing general contractions and compressions."""

    def __init__(self, n_qudits: int, qudits_dimension: int = 2):
        """Initialize a tensor network circuit.

        Args:
            n_qudits: Number of qudits in the system.
            qudits_dimension: Dimension of the qudits (default: 2 for qubits).
        """
        # Create initial state tensor |0⟩
        initial_tensor = np.zeros(qudits_dimension, dtype=complex)
        initial_tensor[0] = 1

        # Initialize first layer with all qudits in |0⟩ state
        self.tensors = [
            [
                tn.Node(initial_tensor, name=f"|0>_{qudit}", axis_names=["out"])
                for qudit in range(n_qudits)
            ]
        ]

        # Store circuit parameters
        self.n_qudits = n_qudits
        self.dimension_qudits = qudits_dimension

        # Initialize circuit depth
        self.depth = 1

    # ---------------------------------------------------------------------------
    def append_operator(self, operator: list, qudits: list):
        """Add an operator to the circuit at the earliest available time.

        Receives an operator as a list of lists of tensors (to include operators
        with more than one layer) and the series of qubits to which it should be applied.
        The method adds these tensor layers at the first available time for all
        involved qudits, creating new time steps if necessary.

        Args:
            operator: List of lists of tensors to add to the circuit. Each sublist
                     represents a layer, and each element in a sublist is a tensor
                     for the corresponding qudit.
            qudits: List of qudit indices for each element of the operator.

        Modifies:
            self.tensors: Updates the tensor network with the new operator.
            self.depth: May increase if new time steps are needed.
        """
        # Number of layers and qudits involved
        n_layers = len(operator)
        n_qudits_operator = len(qudits)

        # Validate input
        assert n_qudits_operator <= self.n_qudits, (
            f"The number of qudits in the gate ({n_qudits_operator}) must not exceed "
            f"the number of qudits in the circuit ({self.n_qudits})"
        )

        # Find the first available time position
        # Start by assuming we need a new time step
        t_free = self.depth

        # Search backwards through existing time steps
        for t in range(self.depth - 1, -1, -1):
            # Check if this time step is free for all required qudits
            is_free = all(self.tensors[t][qudit] is None for qudit in qudits)

            if is_free:
                # This time step is available
                t_free = t
            else:
                # Found a conflict; can't go further back due to non-commutativity
                break

        # Calculate how many new time steps we need
        time_steps_needed = (t_free + n_layers) - self.depth

        if time_steps_needed > 0:
            # Add new time steps
            new_layers = [
                [None for _ in range(self.n_qudits)] for _ in range(time_steps_needed)
            ]
            self.tensors.extend(new_layers)
            self.depth += time_steps_needed

        # Add the operator tensors to the circuit
        for layer_idx in range(n_layers):
            for qudit_idx in range(n_qudits_operator):
                target_time = t_free + layer_idx
                target_qudit = qudits[qudit_idx]
                self.tensors[target_time][target_qudit] = operator[layer_idx][qudit_idx]

    # ---------------------------------------------------------------------------
    def contract(
        self, scheme: str = "time", representation: str = "statevector", eps: float = 0
    ):
        """Contract the tensor network according to the specified scheme and representation.

        Args:
            scheme: Contraction scheme ('time' or 'spatial'). Only 'time' is implemented.
            representation: State representation ('statevector' or 'MPS').
                           Other options like 'density operator' and 'PEPS' are not implemented.
            eps: Maximum error tolerance for MPS compression (only for real-valued MPS).
                If eps=0, no approximation is applied.

        Returns:
            The contracted tensor network in the specified representation.

        Note:
            - Spatial contraction is not implemented due to complexity.
            - For 'time' + 'statevector': contracts all tensors into a single statevector.
            - For 'time' + 'MPS': maintains MPS form at each time step with optional compression.
        """
        if scheme == "time":  # Contract along the time dimension
            if representation == "statevector":  # Full statevector representation
                # Track the most recent time step for each qudit
                last_layer = np.zeros(self.n_qudits, dtype=int)

                # Connect tensors across time steps
                for t in range(1, self.depth):
                    # Connect each qudit to its previous time step
                    for qudit in range(self.n_qudits - 1):
                        if self.tensors[t][qudit] is not None:
                            # Connect input edge to output of previous time step
                            (
                                self.tensors[t][qudit]["in"]
                                ^ self.tensors[last_layer[qudit]][qudit]["out"]
                            )
                            last_layer[qudit] = t

                            # Connect spatial edges (down/up) for multi-qudit gates
                            if "down" in self.tensors[t][qudit].axis_names:
                                (
                                    self.tensors[t][qudit]["down"]
                                    ^ self.tensors[t][qudit + 1]["up"]
                                )

                    # Handle the last qudit separately (no 'down' edge)
                    if self.tensors[t][-1] is not None:
                        (
                            self.tensors[t][-1]["in"]
                            ^ self.tensors[last_layer[-1]][-1]["out"]
                        )
                        last_layer[-1] = t

                # Contract the initial time step (t=0)
                current_tensor = self.tensors[0][0]
                for qudit in range(1, self.n_qudits):
                    # Use outer product for unconnected tensors in the same layer
                    current_tensor = tn.contract_between(
                        current_tensor,
                        self.tensors[0][qudit],
                        name=f"state^{qudit}_0",
                        allow_outer_product=True,
                    )

                # Contract subsequent time steps
                # Contracting top-to-bottom ensures output indices stack in correct order
                for t in range(1, self.depth):
                    for qudit in range(self.n_qudits):
                        if self.tensors[t][qudit] is not None:
                            current_tensor = tn.contract_between(
                                current_tensor,
                                self.tensors[t][qudit],
                                name=f"state^{qudit}_{t}",
                            )

                return current_tensor

            # ...................................................................
            elif representation == "MPS":  # Use the MPS representation
                # Initialize MPS from the first layer
                current_MPS = self.tensors[0].copy()  # First layer

                # Reshape initial tensors to have proper MPS bond dimensions
                # First qubit: shape (d, 1) with axes ['out', 'down']
                tensor = current_MPS[0].tensor
                current_MPS[0] = tn.Node(
                    np.reshape(tensor, (tensor.shape[0], 1)),
                    name=current_MPS[0].name,
                    axis_names=["out", "down"],
                )

                # Middle qubits: shape (d, 1, 1) with axes ['out', 'up', 'down']
                for qudit in range(1, self.n_qudits - 1):
                    tensor = current_MPS[qudit].tensor
                    current_MPS[qudit] = tn.Node(
                        np.reshape(tensor, (tensor.shape[0], 1, 1)),
                        name=current_MPS[qudit].name,
                        axis_names=["out", "up", "down"],
                    )

                # Last qubit: shape (d, 1) with axes ['out', 'up']
                tensor = current_MPS[-1].tensor
                current_MPS[-1] = tn.Node(
                    np.reshape(tensor, (tensor.shape[0], 1)),
                    name=current_MPS[-1].name,
                    axis_names=["out", "up"],
                )

                # Contract with subsequent layers
                for t in range(1, self.depth):
                    for qudit in range(self.n_qudits):
                        # Skip if no tensor at this position
                        if self.tensors[t][qudit] is None:
                            continue

                        # Connect input edge to output of current MPS
                        self.tensors[t][qudit]["in"] ^ current_MPS[qudit]["out"]

                        # Determine axis names after contraction
                        has_out = "out" in self.tensors[t][qudit].axis_names
                        gate_axes = [
                            _ + "1"
                            for _ in self.tensors[t][qudit].axis_names[2:]
                            if _ in ["up", "down"]
                        ]
                        mps_axes = [_ + "2" for _ in current_MPS[qudit].axis_names[1:]]

                        if has_out:
                            axis_names = ["out"] + gate_axes + mps_axes
                        else:
                            axis_names = gate_axes + mps_axes

                        # Contract gate with current MPS tensor
                        current_MPS[qudit] = tn.contract_between(
                            self.tensors[t][qudit],
                            current_MPS[qudit],
                            name=f"state^{qudit}_{t}",
                            axis_names=axis_names,
                        )

                        # At each step we need to group the indices that connect in time correctly.
                        # If there are up1, down1, up2, down2 we need to join them into single up, down indices.
                        tensor_node = current_MPS[qudit].tensor
                        shape = tensor_node.shape

                        # Determine which axes we have (excluding 'out' if present)
                        has_out = "out" in axis_names
                        bond_axes = axis_names[1:] if has_out else axis_names

                        # Process the tensor
                        if bond_axes:  # Only process if there are bond axes
                            tensor_node, final_axis_names = process_axes(
                                tensor_node, bond_axes, has_out
                            )
                        else:
                            final_axis_names = ["out"] if has_out else []

                        # Create new node with processed tensor
                        current_MPS[qudit] = tn.Node(
                            tensor_node,
                            name=f"MPS^{qudit}_{t}",
                            axis_names=final_axis_names,
                        )

                    # If we have eps greater than 0, compress the state
                    if eps > 0:
                        # Transform MPS to canonical form [up, out, down] for compression
                        transformed_MPS = [None] * self.n_qudits
                        for qudit in range(self.n_qudits):
                            tensor_node = current_MPS[qudit].tensor
                            shape = tensor_node.shape
                            axes = current_MPS[qudit].axis_names

                            # Map current axes to canonical form [up, out, down]
                            if axes == ["out", "up", "down"]:
                                # Reorder from [out, up, down] to [up, out, down]
                                transformed_MPS[qudit] = current_MPS[
                                    qudit
                                ].reorder_axes([1, 0, 2])

                            elif axes == ["out", "down"]:
                                # Add missing up dimension
                                tensor_node = np.reshape(
                                    tensor_node, (1, shape[0], shape[1])
                                )
                                transformed_MPS[qudit] = tn.Node(
                                    tensor_node,
                                    name=f"MPS^{qudit}_{t}",
                                    axis_names=["up", "out", "down"],
                                )

                            elif axes == ["out", "up"]:
                                # Add missing down dimension
                                tensor_node = np.moveaxis(tensor_node, 0, 1)
                                shape = tensor_node.shape
                                tensor_node = np.reshape(
                                    tensor_node, (shape[0], shape[1], 1)
                                )
                                transformed_MPS[qudit] = tn.Node(
                                    tensor_node,
                                    name=f"MPS^{qudit}_{t}",
                                    axis_names=["up", "out", "down"],
                                )

                            elif axes == ["up", "down"]:
                                # Add missing out dimension
                                tensor_node = np.reshape(
                                    tensor_node, (shape[0], 1, shape[1])
                                )
                                transformed_MPS[qudit] = tn.Node(
                                    tensor_node,
                                    name=f"MPS^{qudit}_{t}",
                                    axis_names=["up", "out", "down"],
                                )

                            elif axes == ["out"]:
                                # Add missing up and down dimensions
                                tensor_node = np.reshape(tensor_node, (1, shape[0], 1))
                                transformed_MPS[qudit] = tn.Node(
                                    tensor_node,
                                    name=f"MPS^{qudit}_{t}",
                                    axis_names=["up", "out", "down"],
                                )

                            elif axes == ["up"]:
                                # Add missing out and down dimensions
                                tensor_node = np.reshape(tensor_node, (shape[0], 1, 1))
                                transformed_MPS[qudit] = tn.Node(
                                    tensor_node,
                                    name=f"MPS^{qudit}_{t}",
                                    axis_names=["up", "out", "down"],
                                )

                            elif axes == ["down"]:
                                # Add missing up and out dimensions
                                tensor_node = np.reshape(tensor_node, (1, 1, shape[0]))
                                transformed_MPS[qudit] = tn.Node(
                                    tensor_node,
                                    name=f"MPS^{qudit}_{t}",
                                    axis_names=["up", "out", "down"],
                                )

                            elif axes == []:
                                # Add all missing dimensions
                                tensor_node = np.reshape(tensor_node, (1, 1, 1))
                                transformed_MPS[qudit] = tn.Node(
                                    tensor_node,
                                    name=f"MPS^{qudit}_{t}",
                                    axis_names=["up", "out", "down"],
                                )

                        # Compress the MPS using TT decomposition
                        compressed_MPS = TT_State_Compression(transformed_MPS, eps)

                        # Transform back to original form
                        # First qudit: remove up dimension
                        tensor_node = compressed_MPS[0].tensor
                        shape = tensor_node.shape
                        tensor_node = np.reshape(tensor_node, (shape[1], shape[2]))
                        current_MPS[0] = tn.Node(
                            tensor_node, name=f"MPS^{0}_{t}", axis_names=["out", "down"]
                        )

                        # Intermediate qudits: reorder from [up, out, down] to [out, up, down]
                        for qudit in range(1, self.n_qudits - 1):
                            current_MPS[qudit] = compressed_MPS[qudit].reorder_axes(
                                [1, 0, 2]
                            )

                        # Last qudit: remove down dimension
                        tensor_node = compressed_MPS[-1].tensor
                        tensor_node = np.moveaxis(tensor_node, 0, 1)
                        shape = tensor_node.shape
                        tensor_node = np.reshape(tensor_node, (shape[0], shape[1]))
                        current_MPS[-1] = tn.Node(
                            tensor_node,
                            name=f"MPS^{self.n_qudits - 1}_{t}",
                            axis_names=["out", "up"],
                        )

                # Return the MPS list
                # Note: If final tensors lack output indices, structure may differ
                self.current_MPS = current_MPS
                return current_MPS

            # ...................................................................
            elif representation == "density operator" or representation == "PEPS":
                # Density operator and PEPS representations are not yet implemented
                raise NotImplementedError(
                    f"Representation '{representation}' is not yet supported. "
                    "Currently only 'statevector' and 'MPS' representations are available."
                )

        # ...................................................................
        elif scheme == "spatial":
            # Spatial scheme is not yet implemented
            raise NotImplementedError(
                f"Scheme '{scheme}' is not yet supported. "
                "Currently only 'temporal' scheme is available."
            )
