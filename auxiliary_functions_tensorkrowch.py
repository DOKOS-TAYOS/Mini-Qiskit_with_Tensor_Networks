"""
Auxiliary functions for the TensorKrowch version of the mini-Qiskit notebook.

This module mirrors the public helper surface used by the notebook while
isolating the TensorKrowch-specific graph and contraction details from the
TensorNetwork implementation.
"""

from __future__ import annotations

import copy

import tensorkrowch as tk
import torch


CANONICAL_MPS_AXES = ("up", "out", "down")


def ensure_torch_tensor(tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(tensor, dtype=dtype, device=device)


def tensor_to_numpy_flat(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy().reshape(-1)


def tensor_to_python_scalar(tensor: torch.Tensor):
    return tensor.detach().cpu().item()


def _make_node(
    tensor,
    *,
    name: str,
    axis_names: list[str] | tuple[str, ...],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tk.Node:
    if dtype is None:
        if isinstance(tensor, torch.Tensor):
            dtype = tensor.dtype
        else:
            dtype = torch.complex128
    if device is None:
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device("cpu")

    tensor = ensure_torch_tensor(tensor, device=device, dtype=dtype)
    return tk.Node(
        shape=tensor.shape,
        axes_names=list(axis_names),
        name=name,
        tensor=tensor,
    )


def set_axis_names(node: tk.Node, axis_names: list[str] | tuple[str, ...]) -> tk.Node:
    if len(node.axes) != len(axis_names):
        raise ValueError(
            f"Axis-name count mismatch: node has {len(node.axes)} axes but "
            f"{len(axis_names)} names were provided."
        )
    for axis, axis_name in zip(node.axes, axis_names):
        axis.name = axis_name
    return node


def copy_node_grid(node_grid: list[list[tk.Node | None]]) -> list[list[tk.Node | None]]:
    return copy.deepcopy(node_grid)


def append_operator_to_grid(
    layers: list[list[tk.Node | None]],
    operator: list[list[tk.Node | None]],
    qudits: list[int],
    *,
    n_qudits: int,
) -> list[list[tk.Node | None]]:
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


def connect_edges(edge1, edge2):
    return tk.connect(edge1, edge2)


def contract_between(
    node1: tk.Node,
    node2: tk.Node,
    *,
    name: str | None = None,
    allow_outer_product: bool = False,
    axis_names: list[str] | tuple[str, ...] | None = None,
) -> tk.Node:
    shared_edges = tk.get_shared_edges(node1, node2)
    if shared_edges:
        result = tk.contract_between(node1, node2)
    else:
        if not allow_outer_product:
            raise ValueError(
                f"No shared edges between nodes {node1.name!r} and {node2.name!r}."
            )
        if node1.network is not node2.network:
            node2.move_to_network(node1.network)
        result = tk.tprod(node1, node2)

    if name is not None:
        result.name = name
    if axis_names is not None:
        set_axis_names(result, axis_names)
    return result


def adjoint_node(node: tk.Node) -> tk.Node:
    tensor = torch.conj(node.tensor)
    axis_names = list(node.axes_names)

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

    return _make_node(
        tensor,
        name=node.name,
        axis_names=axis_names,
        device=tensor.device,
        dtype=tensor.dtype,
    )


def build_adjoint_grid(node_grid: list[list[tk.Node | None]]) -> list[list[tk.Node | None]]:
    return [
        [adjoint_node(node) if node is not None else None for node in layer]
        for layer in reversed(node_grid)
    ]


def build_basis_bra_layer(
    bitstring: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[list[tk.Node]]:
    zero = torch.tensor([1, 0], dtype=dtype, device=device)
    one = torch.tensor([0, 1], dtype=dtype, device=device)
    return [[
        _make_node(
            zero.clone() if bit == "0" else one.clone(),
            name=f"<{bit}|",
            axis_names=["in"],
            device=device,
            dtype=dtype,
        )
        for bit in bitstring
    ]]


def connect_circuit_layers(
    layers: list[list[tk.Node | None]],
    *,
    n_qudits: int,
) -> list:
    last_layer = [0] * n_qudits

    for time_idx in range(1, len(layers)):
        for qudit, node in enumerate(layers[time_idx]):
            if node is None:
                continue

            previous_node = layers[last_layer[qudit]][qudit]
            if previous_node is None or "out" not in previous_node.axes_names:
                raise ValueError(
                    f"Qudit {qudit} has no output axis available for contraction."
                )
            if "in" not in node.axes_names:
                raise ValueError(
                    f"Node {node.name!r} at layer {time_idx}, qudit {qudit} is missing an input axis."
                )

            connect_edges(node["in"], previous_node["out"])
            last_layer[qudit] = time_idx

            if "down" in node.axes_names:
                if qudit + 1 >= n_qudits:
                    raise ValueError(
                        "Invalid multi-qudit gate layout: missing lower neighbour."
                    )

                lower_node = layers[time_idx][qudit + 1]
                if lower_node is None or "up" not in lower_node.axes_names:
                    raise ValueError(
                        "Invalid multi-qudit gate layout: missing lower neighbour."
                    )

                connect_edges(node["down"], lower_node["up"])

    output_edges = []
    for qudit in range(n_qudits):
        final_node = layers[last_layer[qudit]][qudit]
        if final_node is not None and "out" in final_node.axes_names:
            output_edges.append(final_node["out"])
    return output_edges


def contract_grid_row(
    row_nodes: list[tk.Node | None],
    *,
    name: str,
) -> tk.Node:
    nodes = [node for node in row_nodes if node is not None]
    if not nodes:
        raise ValueError("Cannot contract an empty circuit row.")

    current = nodes[0]
    for index, node in enumerate(nodes[1:], start=1):
        current = contract_between(current, node, name=f"{name}^{index}")
    return current


def reorder_dense_output_axes(node: tk.Node, output_edges) -> tk.Node:
    if not output_edges:
        return node

    current_indices = []
    for edge in reversed(output_edges):
        try:
            current_indices.append(node.edges.index(edge))
        except ValueError as exc:
            raise ValueError("Could not locate a final output edge on the contracted node.") from exc

    remaining_indices = [
        index for index in range(len(node.edges)) if index not in current_indices
    ]
    target_order = current_indices + remaining_indices
    if target_order == list(range(len(node.edges))):
        return node

    permuted = tk.permute(node, target_order)
    permuted.name = node.name
    return permuted


def process_axes(
    tensor: torch.Tensor,
    bond_axes: list[str],
    has_out: bool,
) -> tuple[torch.Tensor, list[str]]:
    axis_map = {name: index for index, name in enumerate(bond_axes)}
    out_offset = 1 if has_out else 0

    has_up1 = "up1" in axis_map
    has_up2 = "up2" in axis_map
    has_down1 = "down1" in axis_map
    has_down2 = "down2" in axis_map

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

    if target_order != list(range(tensor.ndim)):
        tensor = tensor.permute(target_order)

    new_shape = []
    if has_out:
        new_shape.append(tensor.shape[0])

    if has_up1 and has_up2:
        new_shape.append(tensor.shape[out_offset] * tensor.shape[out_offset + 1])
    elif has_up1 or has_up2:
        new_shape.append(tensor.shape[out_offset])

    down_start_idx = out_offset + len(up_axes)
    if has_down1 and has_down2:
        new_shape.append(tensor.shape[down_start_idx] * tensor.shape[down_start_idx + 1])
    elif has_down1 or has_down2:
        new_shape.append(tensor.shape[down_start_idx])

    tensor = tensor.reshape(tuple(new_shape))

    final_axes = []
    if has_out:
        final_axes.append("out")
    if has_up1 or has_up2:
        final_axes.append("up")
    if has_down1 or has_down2:
        final_axes.append("down")

    return tensor, final_axes


def canonicalize_mps_node(node: tk.Node, *, name: str | None = None) -> tk.Node:
    axis_names = list(node.axes_names)
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

    return _make_node(
        tensor,
        name=node.name if name is None else name,
        axis_names=list(CANONICAL_MPS_AXES),
        device=tensor.device,
        dtype=tensor.dtype,
    )


def canonicalize_mps_grid(nodes: list[tk.Node]) -> list[tk.Node]:
    return [canonicalize_mps_node(node) for node in nodes]


def _truncate_rank(singular_values: torch.Tensor, eps: float) -> int:
    if singular_values.numel() == 0:
        return 1
    if eps <= 0:
        return singular_values.numel()
    keep = int((singular_values > eps).sum().item())
    return max(1, keep)


def dense_tensor_to_mps_nodes(
    dense_tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    eps: float = 0,
) -> list[tk.Node]:
    dense_tensor = ensure_torch_tensor(dense_tensor, device=device, dtype=dtype)
    n_sites = dense_tensor.ndim

    if n_sites == 0:
        dense_tensor = dense_tensor.reshape(1)
        n_sites = 1
    if n_sites == 1:
        return [
            _make_node(
                dense_tensor.reshape(1, dense_tensor.shape[0], 1),
                name="MPS^0",
                axis_names=list(CANONICAL_MPS_AXES),
                device=device,
                dtype=dtype,
            )
        ]

    working = dense_tensor
    dims = list(working.shape)
    left_rank = 1
    nodes = []

    for site, phys_dim in enumerate(dims[:-1]):
        working = working.reshape(left_rank * phys_dim, -1)
        u, singular_values, vh = torch.linalg.svd(working, full_matrices=False)
        rank = _truncate_rank(singular_values, eps)

        u = u[:, :rank]
        singular_values = singular_values[:rank]
        vh = vh[:rank, :]

        core = u.reshape(left_rank, phys_dim, rank)
        nodes.append(
            _make_node(
                core,
                name=f"MPS^{site}",
                axis_names=list(CANONICAL_MPS_AXES),
                device=device,
                dtype=dtype,
            )
        )

        working = torch.diag(singular_values.to(dtype=vh.dtype)) @ vh
        left_rank = rank

    final_core = working.reshape(left_rank, dims[-1], 1)
    nodes.append(
        _make_node(
            final_core,
            name=f"MPS^{len(dims) - 1}",
            axis_names=list(CANONICAL_MPS_AXES),
            device=device,
            dtype=dtype,
        )
    )
    return nodes


def TT_State_Compression(
    tn_state: list[tk.Node],
    eps: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[tk.Node]:
    canonical_nodes = canonicalize_mps_grid(tn_state)
    dense_tensor = mps_nodes_to_dense_tensor(canonical_nodes)
    if dense_tensor.ndim > 1:
        dense_tensor = dense_tensor.permute(tuple(reversed(range(dense_tensor.ndim))))
    return dense_tensor_to_mps_nodes(dense_tensor, device=device, dtype=dtype, eps=eps)


def mps_nodes_to_dense_tensor(nodes: list[tk.Node]) -> torch.Tensor:
    canonical_nodes = canonicalize_mps_grid(nodes)
    dense_tensor = canonical_nodes[0].tensor
    for node in canonical_nodes[1:]:
        dense_tensor = torch.tensordot(dense_tensor, node.tensor, dims=([-1], [0]))
    dense_tensor = dense_tensor.squeeze(0).squeeze(-1)
    if dense_tensor.ndim > 1:
        dense_tensor = dense_tensor.permute(tuple(reversed(range(dense_tensor.ndim))))
    return dense_tensor


def mps_nodes_to_scalar(nodes: list[tk.Node]):
    scalar_tensor = mps_nodes_to_dense_tensor(nodes).reshape(-1)[0]
    return tensor_to_python_scalar(scalar_tensor)


def build_control_tensors(*, device: torch.device, dtype: torch.dtype):
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
    if gate_name == "Pass":
        return "#2a2a2a", "#444", ""
    if gate_name.startswith("|0>") or gate_name.startswith("MPS"):
        return "#64748b", "#94a3b8", "|0>"
    if "Ctrl_" in gate_name or "cCtrl_" in gate_name:
        return "#4a9eff", "#6bb3ff", "o"
    if gate_name in ("X", "Y", "Z"):
        return "#ff6b6b", "#ff8e8e", gate_name
    if gate_name == "H":
        return "#69db7c", "#8ee08e", gate_name
    if gate_name in ("CNOT", "CX", "C-X", "CC-X"):
        return "#ffd43b", "#ffe066", "+"
    if gate_name in ("CZ", "C-Z"):
        return "#ffd43b", "#ffe066", "Z"
    if gate_name.startswith(("RX", "RY", "RZ")):
        return "#b197fc", "#d0bfff", gate_name
    return "#94a3b8", "#cbd5e1", gate_name
