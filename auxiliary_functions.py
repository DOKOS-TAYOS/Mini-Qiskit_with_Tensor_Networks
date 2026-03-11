"""
Auxiliary functions for a mini-Qiskit implementation built on TensorNetwork.

This module provides low-level tensor operations used by the quantum circuit simulator:
- Tensor conversion and manipulation (ensure_torch_tensor, tensor_to_numpy_flat, etc.)
- Circuit grid management (copy_node_grid, append_operator_to_grid)
- Adjoint and basis-state construction (adjoint_node, build_adjoint_grid, build_basis_bra_layer)
- MPS/TT canonicalization and compression (process_axes, canonicalize_mps_node, TT_State_Compression)
- Control-gate tensor construction (build_control_tensors, build_controlled_target_tensor)
- Circuit visualization helpers (gate_style for draw_circuit)

All tensors use PyTorch as the backend for GPU support and complex arithmetic.
"""

import tensornetwork as tn
import torch
import tntorch


# Canonical axis order for MPS/TT cores: left bond, physical index, right bond.
# This order is required by tntorch for TT compression and keeps contraction paths consistent.
CANONICAL_MPS_AXES = ("up", "out", "down")


def ensure_torch_tensor(tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert input data to a PyTorch tensor on the requested device and dtype.

    Handles numpy arrays, lists, or existing tensors. Used to guarantee all
    circuit tensors live on the same device (CPU/GPU) and use consistent dtypes.
    """
    return torch.as_tensor(tensor, dtype=dtype, device=device)


def tensor_to_numpy_flat(tensor: torch.Tensor):
    """
    Return a flattened numpy copy of a torch tensor.

    Detaches from the computation graph, moves to CPU, and flattens to 1D.
    Useful for exporting state vectors or amplitudes for analysis.
    """
    return tensor.detach().cpu().numpy().reshape(-1)


def tensor_to_python_scalar(tensor: torch.Tensor):
    """
    Convert a scalar torch tensor to a native Python scalar (complex or float).

    Used when extracting single amplitudes or expectation values for display.
    """
    return tensor.detach().cpu().item()


def copy_node_grid(node_grid: list[list[tn.Node | None]]) -> list[list[tn.Node | None]]:
    """
    Deep-copy a layered grid of TensorNetwork nodes while preserving internal edges.

    The circuit is stored as layers[time][qudit] = Node. This function replicates
    all nodes so that contractions (e.g. for expectation values) do not mutate
    the original circuit. Uses tn.replicate_nodes to preserve edge connectivity.
    """
    flat_nodes = [node for layer in node_grid for node in layer if node is not None]
    if not flat_nodes:
        return [[None for _ in layer] for layer in node_grid]

    copied_nodes = tn.replicate_nodes(flat_nodes)
    copied_iter = iter(copied_nodes)
    return [
        [next(copied_iter) if node is not None else None for node in layer]
        for layer in node_grid
    ]


def append_operator_to_grid(
    layers: list[list[tn.Node | None]],
    operator: list[list[tn.Node | None]],
    qudits: list[int],
    *,
    n_qudits: int,
) -> list[list[tn.Node | None]]:
    """
    Append an operator to a layered grid using the same scheduling rule as the circuit.

    Finds the earliest time slot where all target qudits are free (no gate already
    placed), then places each operator layer there. Extends the grid with new
    time steps if needed. Used when adding gates, bra layers, or adjoint circuits.
    """
    if not operator:
        raise ValueError("Operator must contain at least one layer.")
    if len(qudits) == 0:
        raise ValueError("At least one qudit position is required.")
    if len(qudits) > n_qudits:
        raise ValueError(
            f"The number of qudits in the gate ({len(qudits)}) must not exceed "
            f"the number of qudits in the circuit ({n_qudits})."
        )
    if any(not 0 <= qudit < n_qudits for qudit in qudits):
        raise ValueError(f"Qudit positions must be in [0, {n_qudits}).")

    for layer in operator:
        if len(layer) != len(qudits):
            raise ValueError(
                "Each operator layer must contain exactly one tensor per target qudit."
            )

    current_depth = len(layers)
    t_free = current_depth
    for time_idx in range(current_depth - 1, -1, -1):
        if all(layers[time_idx][qudit] is None for qudit in qudits):
            t_free = time_idx
        else:
            break

    time_steps_needed = (t_free + len(operator)) - current_depth
    if time_steps_needed > 0:
        layers.extend([[None for _ in range(n_qudits)] for _ in range(time_steps_needed)])

    for layer_idx, operator_layer in enumerate(operator):
        for qudit_idx, node in enumerate(operator_layer):
            layers[t_free + layer_idx][qudits[qudit_idx]] = node

    return layers


def adjoint_node(node: tn.Node) -> tn.Node:
    """
    Return the conjugate transpose (adjoint) of a circuit node.

    Takes the complex conjugate and swaps "in" and "out" axes so that
    the resulting node represents the Hermitian conjugate of the gate.
    Used when computing expectation values <psi|O|psi>: we append the
    adjoint of the circuit to obtain the bra <psi|, then contract.
    """
    tensor = torch.conj(node.tensor)
    axis_names = list(node.axis_names)

    if "in" in axis_names and "out" in axis_names:
        in_axis = axis_names.index("in")
        out_axis = axis_names.index("out")
        permutation = list(range(tensor.ndim))
        permutation[in_axis], permutation[out_axis] = permutation[out_axis], permutation[in_axis]
        tensor = tensor.permute(permutation)
        axis_names[in_axis], axis_names[out_axis] = axis_names[out_axis], axis_names[in_axis]
    else:
        axis_names = [
            "out" if axis_name == "in" else "in" if axis_name == "out" else axis_name
            for axis_name in axis_names
        ]

    return tn.Node(tensor, name=node.name, axis_names=axis_names, backend="pytorch")


def build_adjoint_grid(node_grid: list[list[tn.Node | None]]) -> list[list[tn.Node | None]]:
    """
    Return the reversed adjoint of a layered circuit grid.

    Applies adjoint_node to each node and reverses the layer order. This
    represents the Hermitian conjugate of the full circuit, used for
    expectation-value contractions.
    """
    return [
        [adjoint_node(node) if node is not None else None for node in layer]
        for layer in reversed(node_grid)
    ]


def build_basis_bra_layer(
    bitstring: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[list[tn.Node]]:
    """
    Create a single layer of computational-basis bra vectors.

    For each bit in the bitstring, creates <0| or <1| as a 1D tensor with
    axis "in". Used to project the final state onto a specific basis state
    (e.g. for check_state or amplitude extraction).
    """
    zero = torch.tensor([1, 0], dtype=dtype, device=device)
    one = torch.tensor([0, 1], dtype=dtype, device=device)
    return [[
        tn.Node(
            zero.clone() if bit == "0" else one.clone(),
            name=f"<{bit}|",
            axis_names=["in"],
            backend="pytorch",
        )
        for bit in bitstring
    ]]

def process_axes(
    tensor: torch.Tensor,
    bond_axes: list[str],
    has_out: bool,
) -> tuple[torch.Tensor, list[str]]:
    """
    Group temporary MPS bond axes into canonical up/down bonds.

    After contracting a gate with an MPS core, we get axes like up1, up2, down1, down2.
    This function permutes and reshapes them into a single "up" and "down" bond
    so the result stays in canonical MPS form. Used during MPS time evolution.
    """
    axis_map = {name: index for index, name in enumerate(bond_axes)}
    out_offset = 1 if has_out else 0

    # Determine which bond axes are present
    has_up1 = "up1" in axis_map
    has_up2 = "up2" in axis_map
    has_down1 = "down1" in axis_map
    has_down2 = "down2" in axis_map

    # Build target permutation order: out (if present), then up axes, then down axes
    target_order = []
    if has_out:
        target_order.append(0)
    
    up_axes = []
    if has_up1:
        up_axes.append(axis_map["up1"] + out_offset)
    if has_up2:
        up_axes.append(axis_map["up2"] + out_offset)
    target_order.extend(up_axes)
    
    down_axes = []
    if has_down1:
        down_axes.append(axis_map["down1"] + out_offset)
    if has_down2:
        down_axes.append(axis_map["down2"] + out_offset)
    target_order.extend(down_axes)

    # Permute if necessary
    if target_order != list(range(tensor.ndim)):
        tensor = tensor.permute(target_order)

    # Build new shape by merging paired bond dimensions
    new_shape = []
    if has_out:
        new_shape.append(tensor.shape[0])

    # Merge up dimensions
    if has_up1 and has_up2:
        new_shape.append(tensor.shape[out_offset] * tensor.shape[out_offset + 1])
    elif has_up1 or has_up2:
        new_shape.append(tensor.shape[out_offset])

    # Merge down dimensions
    down_start_idx = out_offset + len(up_axes)
    if has_down1 and has_down2:
        new_shape.append(tensor.shape[down_start_idx] * tensor.shape[down_start_idx + 1])
    elif has_down1 or has_down2:
        new_shape.append(tensor.shape[down_start_idx])

    tensor = tensor.reshape(tuple(new_shape))

    # Build final axis names
    final_axes = []
    if has_out:
        final_axes.append("out")
    if has_up1 or has_up2:
        final_axes.append("up")
    if has_down1 or has_down2:
        final_axes.append("down")

    return tensor, final_axes


def canonicalize_mps_node(node: tn.Node, *, name: str | None = None) -> tn.Node:
    """
    Convert a node with some subset of up/out/down axes into canonical TT-core order.

    Permutes axes to (up, out, down) and pads missing axes with dimension 1.
    Ensures compatibility with tntorch and consistent MPS contraction.
    """
    axis_names = list(node.axis_names)
    invalid_axes = [axis for axis in axis_names if axis not in CANONICAL_MPS_AXES]
    if invalid_axes:
        raise ValueError(f"Unexpected MPS axes: {invalid_axes}")

    present_axes = [axis for axis in CANONICAL_MPS_AXES if axis in axis_names]
    tensor = node.tensor
    permutation = [axis_names.index(axis) for axis in present_axes]
    if permutation != list(range(len(present_axes))):
        tensor = tensor.permute(permutation)

    shape_by_axis = {
        axis_name: tensor.shape[index] for index, axis_name in enumerate(present_axes)
    }
    canonical_shape = tuple(shape_by_axis.get(axis_name, 1) for axis_name in CANONICAL_MPS_AXES)
    tensor = tensor.reshape(canonical_shape)

    return tn.Node(
        tensor,
        name=node.name if name is None else name,
        axis_names=list(CANONICAL_MPS_AXES),
        backend="pytorch",
    )


def canonicalize_mps_grid(nodes: list[tn.Node]) -> list[tn.Node]:
    """
    Canonicalize all nodes in an MPS chain.

    Applies canonicalize_mps_node to each core so the full MPS has
    consistent axis ordering for contraction and TT compression.
    """
    return [canonicalize_mps_node(node) for node in nodes]


def TT_State_Compression(
    tn_state: list[tn.Node],
    eps: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[tn.Node]:
    """
    Compress a canonical MPS into TT format with the requested tolerance.

    Uses tntorch.round_tt to perform SVD-based truncation. Larger eps yields
    smaller bond dimensions and faster contractions at the cost of accuracy.
    Returns a new list of MPS nodes with reduced bond dimensions.
    """
    canonical_nodes = canonicalize_mps_grid(tn_state)
    tensor_list = [
        ensure_torch_tensor(node.tensor, dtype=dtype, device=device)
        for node in canonical_nodes
    ]
    tt_compressed = tntorch.Tensor(tensor_list)
    tt_compressed.round_tt(eps=eps)

    compressed_nodes = []
    for index, core in enumerate(tt_compressed.cores):
        compressed_nodes.append(
            tn.Node(
                core.to(dtype=dtype, device=device),
                name=canonical_nodes[index].name,
                axis_names=list(CANONICAL_MPS_AXES),
                backend="pytorch",
            )
        )
    return compressed_nodes


def mps_nodes_to_dense_tensor(nodes: list[tn.Node]) -> torch.Tensor:
    """
    Contract a canonical MPS chain into a dense tensor over the physical indices.

    Sequentially contracts adjacent cores along the bond dimensions, producing
    the full state vector. Used when the final result must be a dense array.
    """
    canonical_nodes = canonicalize_mps_grid(nodes)
    dense_tensor = canonical_nodes[0].tensor
    for node in canonical_nodes[1:]:
        dense_tensor = torch.tensordot(dense_tensor, node.tensor, dims=([-1], [0]))
    dense_tensor = dense_tensor.squeeze(0).squeeze(-1)
    if dense_tensor.ndim > 1:
        dense_tensor = dense_tensor.permute(tuple(reversed(range(dense_tensor.ndim))))
    return dense_tensor


def mps_nodes_to_scalar(nodes: list[tn.Node]):
    """
    Reduce a scalar-valued MPS chain to a Python scalar.

    Contracts the MPS to a dense tensor and extracts the single element.
    Used for expectation values and amplitude checks when using MPS representation.
    """
    scalar_tensor = mps_nodes_to_dense_tensor(nodes).reshape(-1)[0]
    return tensor_to_python_scalar(scalar_tensor)


def build_control_tensors(*, device: torch.device, dtype: torch.dtype):
    """
    Create control-signal propagation tensors for single-control gates.

    Returns ctrl_tensor_down, ctrl_tensor_up, and pass_tensor. These 3-index
    tensors propagate a control signal (0 or 1) along the circuit: down/up
    carry the signal between qubits, pass forwards it through intermediate wires.
    Used to build CNOT, CZ, and other single-control gates.
    """
    ctrl_tensor_down = torch.zeros((2, 2, 2), dtype=dtype, device=device)
    ctrl_tensor_up = torch.zeros((2, 2, 2), dtype=dtype, device=device)
    pass_tensor = torch.zeros((2, 2, 2, 2), dtype=dtype, device=device)

    for signal in range(2):
        ctrl_tensor_down[signal, signal, signal] = 1
        ctrl_tensor_up[signal, signal, signal] = 1
        for state in range(2):
            pass_tensor[state, state, signal, signal] = 1

    return ctrl_tensor_down, ctrl_tensor_up, pass_tensor


def build_multi_control_tensors(*, device: torch.device, dtype: torch.dtype):
    """
    Create control-signal propagation tensors for multi-control gates.

    Extends build_control_tensors with cctrl_up and cctrl_down: 4-index tensors
    that combine two control signals (e.g. for Toffoli). Used when the target
    qubit lies between two controls and needs to receive both signals.
    """
    ctrl_tensor_down, ctrl_tensor_up, pass_tensor = build_control_tensors(
        device=device, dtype=dtype
    )
    cctrl_tensor_up = torch.zeros((2, 2, 2, 2), dtype=dtype, device=device)
    cctrl_tensor_down = torch.zeros((2, 2, 2, 2), dtype=dtype, device=device)

    for state in range(2):
        cctrl_tensor_up[state, state, 0, 0] = 1
        cctrl_tensor_down[state, state, 0, 0] = 1
        cctrl_tensor_up[state, state, 1, state] = 1
        cctrl_tensor_down[state, state, state, 1] = 1

    return {
        "ctrl_down": ctrl_tensor_down,
        "ctrl_up": ctrl_tensor_up,
        "pass": pass_tensor,
        "cctrl_up": cctrl_tensor_up,
        "cctrl_down": cctrl_tensor_down,
    }


def build_controlled_target_tensor(
    base_tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_signals: int,
) -> torch.Tensor:
    """
    Wrap a one-qubit gate tensor with one or two control-signal axes.

    n_signals=1: applies base_tensor only when control=1, identity when control=0.
    n_signals=2: applies base_tensor only when both controls=1. Used to build
    the target part of CNOT, Toffoli, and other controlled gates.
    """
    identity = torch.eye(2, dtype=dtype, device=device)
    base_tensor = ensure_torch_tensor(base_tensor, device=device, dtype=dtype)

    if n_signals == 1:
        controlled_tensor = torch.zeros((2, 2, 2), dtype=dtype, device=device)
        controlled_tensor[:, :, 0] = identity
        controlled_tensor[:, :, 1] = base_tensor
        return controlled_tensor

    if n_signals == 2:
        controlled_tensor = torch.zeros((2, 2, 2, 2), dtype=dtype, device=device)
        controlled_tensor[:, :, 0, 0] = identity
        controlled_tensor[:, :, 1, 0] = identity
        controlled_tensor[:, :, 0, 1] = identity
        controlled_tensor[:, :, 1, 1] = base_tensor
        return controlled_tensor

    raise ValueError("Supported control-signal counts are 1 or 2.")


def gate_style(gate_name: str) -> tuple[str, str, str]:
    """
    Return (facecolor, edgecolor, label) for a gate based on its name.

    Used by draw_circuit for circuit visualization. Maps gate names to
    colors and display labels for consistent styling.
    """
    if gate_name == "Pass":
        return "#2a2a2a", "#444", ""
    if gate_name.startswith("|0>") or gate_name.startswith("MPS"):
        return "#64748b", "#94a3b8", "|0⟩"
    if "Ctrl_" in gate_name or "cCtrl_" in gate_name:
        return "#4a9eff", "#6bb3ff", "●"
    if gate_name in ("X", "Y", "Z"):
        return "#ff6b6b", "#ff8e8e", gate_name
    if gate_name == "H":
        return "#69db7c", "#8ee08e", gate_name
    if gate_name in ("CNOT", "CX"):
        return "#ffd43b", "#ffe066", "⊕"
    if gate_name == "CZ":
        return "#ffd43b", "#ffe066", "Z"
    if "CCX" in gate_name or "Toffoli" in gate_name:
        return "#ffd43b", "#ffe066", "⊕"
    if gate_name.startswith("RX") or gate_name.startswith("RY") or gate_name.startswith("RZ"):
        return "#b197fc", "#d0bfff", gate_name
    return "#94a3b8", "#cbd5e1", gate_name
