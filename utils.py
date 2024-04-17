import io
import torch

def split(data: io.BytesIO, size: int) -> list[bytes]:
    """
    Split a BytesIO object into chunks of size `size`.
    """
    data.seek(0)
    chunks = []
    while True:
        chunk = data.read(size)
        if not chunk:
            break
        chunks.append(chunk)
    return chunks

def fmt(s: str, *fmts: str) -> str:
    """
    Colorize a string.
    """
    format_defs = {
        'green': '\033[92m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'gray': '\033[90m',
        'italic': '\033[3m',
        'bold': '\033[1m',
    }
    return ''.join(format_defs[fmt] for fmt in fmts) + s + '\033[0m'

def compare_models(state_dict1, state_dict2, eps=1e-6):
    total_changed = 0
    total_params = 0

    for key in state_dict1:
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]
        # Flatten the tensors to compare element-wise
        flat_tensor1 = tensor1.view(-1)
        flat_tensor2 = tensor2.view(-1)
        total_params += flat_tensor1.numel()  # Count total elements
        # Find mismatches
        mismatch_indices = torch.where(~torch.isclose(flat_tensor1, flat_tensor2, atol=eps))[0]
        total_changed += len(mismatch_indices)
        
    print(fmt(f"{total_changed}/{total_params} parameters changed ({int(total_changed/total_params*100)}%)", 'gray', 'italic'))