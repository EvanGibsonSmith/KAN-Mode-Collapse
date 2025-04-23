def count_kan_group_flops(module, inputs, outputs):
    """
    Estimate the FLOPs (actually MACs × 2) for the KAT_Group layer.
    fvcore will divide by 2 later if you're only interested in MACs.
    """
    input_tensor = inputs[0]  # Get the input tensor
    shape = input_tensor.shape

    if input_tensor.dim() == 3:
        batch_size, length, channels = shape
        num_elements = batch_size * length * channels
    elif input_tensor.dim() == 2:
        batch_size, channels = shape
        num_elements = batch_size * channels
    else:
        raise ValueError("Invalid input shape for KAT_Group")

    # Estimate FLOPs: 1 multiply + 1 add per degree term
    numerator_deg = module.order[0]
    denominator_deg = module.order[1]
    total_deg = numerator_deg + denominator_deg

    # Each element involves total_deg MACs => 2 * MACs for FLOPs
    flops = 2 * num_elements * total_deg
    print("Custom hook triggered!")
    return dict(flops=flops)

# Then, set your custom handler for KAT_Group
def kan_group_op_handle(module, inputs, outputs):
    """
    Custom FLOP counting for KAT_Group.
    This should call the custom FLOP counting function you registered.
    """
    return count_kan_group_flops(module, inputs, outputs)
