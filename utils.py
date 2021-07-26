import torch

def cal_mutmal_x_cov(n_components, mat_a, mat_b):
    """
    cal x mutmal covriance without use mutmal to reduce memory
    the mat_a is (x-mu) mat and the mat_b is convariance mat
    the bmm or matmul function in torch is high memory consumption so use this instead
    mat_a:torch.Tensor (n,k,1,d)
    mat_b:torch.Tensor (1,k,d,d)
    """
    res = torch.zeros(mat_a.shape).double().to(mat_a.device)
    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)
    return res


def cal_mutmal_x_x(mat_a, mat_b):
    """
    cal x mutmal x without use mutmal to reduce memory
    the bmm or matmul function in torch is high memory consumption,
    so turn matrix multiplication into vector multiplication
    mat_a:torch.Tensor (n,k,1,d)
    mat_b:torch.Tensor (n,k,d,1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)