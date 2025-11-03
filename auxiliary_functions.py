import numpy as np
import torch
import tntorch
import tensornetwork as tn

def TT_State_Compression(TN_State: list, eps: float) -> list:
    """
    Compresses a tensor network state into Tensor Train (TT) format with specified precision.
    
    Args:
        TN_State: List of tensornetwork Node objects representing the state
        eps: Relative error tolerance for compression
        
    Returns:
        List of tensornetwork Node objects in compressed TT format
    """
    # Convert to tntorch Tensor format
    tensor_list = [torch.tensor(node.tensor, dtype=torch.complex128) for node in TN_State]
    TT_Compressed = tntorch.Tensor(tensor_list)
    
    # Compress using TT-rounding
    TT_Compressed.round_tt(eps=eps)
    
    # Convert back to tensornetwork format
    TT_cores = []
    for i, core in enumerate(TT_Compressed.cores):
        TT_cores.append(
            tn.Node(
                core.numpy(),
                name=TN_State[i].name,
                axis_names=TN_State[i].axis_names
            )
        )
    
    return TT_cores


def process_axes(tensor, bond_axes, has_out):
    """Process tensor axes by grouping up1/up2 and down1/down2."""
    shape = tensor.shape
    out_offset = 1 if has_out else 0
    
    # Map axis names to their positions
    axis_map = {name: i for i, name in enumerate(bond_axes)}
    
    # Determine which axes we have
    has_up1 = 'up1' in axis_map
    has_up2 = 'up2' in axis_map
    has_down1 = 'down1' in axis_map
    has_down2 = 'down2' in axis_map
    
    # Build target axis order: [out?], up axes, down axes
    target_order = []
    if has_out:
        target_order.append(0)
    
    # Add up axes in order
    if has_up1:
        target_order.append(axis_map['up1'] + out_offset)
    if has_up2:
        target_order.append(axis_map['up2'] + out_offset)
    
    # Add down axes in order
    if has_down1:
        target_order.append(axis_map['down1'] + out_offset)
    if has_down2:
        target_order.append(axis_map['down2'] + out_offset)
    
    # Reorder axes if needed
    current_order = list(range(len(tensor.shape)))
    if target_order != current_order:
        tensor = np.transpose(tensor, target_order)
        shape = tensor.shape
    
    # Reshape to group up and down dimensions
    new_shape = []
    if has_out:
        new_shape.append(shape[0])
    
    # Group up dimensions
    if has_up1 or has_up2:
        up_idx = out_offset
        if has_up1 and has_up2:
            new_shape.append(shape[up_idx] * shape[up_idx + 1])
        else:
            new_shape.append(shape[up_idx])
    
    # Group down dimensions
    if has_down1 or has_down2:
        down_idx = out_offset + (1 if has_up1 else 0) + (1 if has_up2 else 0)
        if has_down1 and has_down2:
            new_shape.append(shape[down_idx] * shape[down_idx + 1])
        else:
            new_shape.append(shape[down_idx])
    
    tensor = np.reshape(tensor, new_shape)
    
    # Determine final axis names
    final_axes = []
    if has_out:
        final_axes.append('out')
    if has_up1 or has_up2:
        final_axes.append('up')
    if has_down1 or has_down2:
        final_axes.append('down')
    
    return tensor, final_axes