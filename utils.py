import torch
import math

def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape).to(mat_a.device)
    
    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)
    
    return res


def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)


def check_available_ram(device="cpu"):
    """
    Returns available RAM on target device
    args:
        device:     str or torch.device
    """
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, torch.device):
        device = device
    else:
        raise RuntimeError("`device` must be str or torch.device")
    
    if device.type == "cpu":
        return psutil.virtual_memory().available
    else:
        total = torch.cuda.get_device_properties(device).total_memory
        used = torch.cuda.memory_allocated(device)
        return total - used

def will_it_fit(size, device="cpu", safe_mode=True):
    """
    Returns True if an array of given byte size fits in target device.

    if self.safe_mode = False, this function simply compares the given byte size with the remaining RAM on target device. This option is faster, 
        but it doesn't take memory fragmentation into account. So it will still be possible to run out of memory.

    if self.safe_mode = True, it will try to allocate a tensor with the given size. if allocation fails, return False. 
        This option is recommended when the other option fails because of OOM.
    
    args:
        size:       int
        device:     str or torch.device
        safe_mode:  bool
    returns:
        result:     bool
    """
    if safe_mode:
        try:
            torch.empty(size, device=device, dtype=torch.uint8)
        except:
            return False
        return True
    else:
        return check_available_ram(device) >= size


def find_optimal_splits(n, get_required_memory, device="cpu", safe_mode=True):
    """
    Find an optimal number of split for `n`, such that `get_required_memory(math.ceil(n / n_split))` fits in target device's RAM.
    get_required_memory should be a fucntion that receives `math.ceil(n/n_split)` and returns the required memory in bytes.

    args:
        n:                      int
        get_required_memory:    function
        device:                 str or torch.device
        safe_mode:              bool

    returns:
        n_splits:               int

    """
    splits = 1
    sub_n = n
    break_next = False
    while True:
        if break_next:
            break
        if splits > n:
            splits = n
            break_next = True
        sub_n = math.ceil(n / splits)
        required_memory = get_required_memory(sub_n)
        if will_it_fit(required_memory, device):
            break
        else:
            splits *= 2
            continue
    return splits