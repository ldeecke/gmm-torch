# Benchmark
GPU: Tesla T4 (16GM DRAM)

- covariance_type = "full"
- init_params = "random"
- n_ter = 20
- delta = 0

| setup | original | k-loop | optimized (single) | optimized (double) |
| --- | --- | --- | --- | --- |
| n_features=16, n_components=16, n_data=100,000 | 6.9s | 6.9s | 0.5s | 3.44s |
| n_features=16, n_components=16, n_data=1,000,000 | OOM | 68.8s | 3.7s | 34.0s |
| n_features=64, n_components=64, n_data=100,000 | OOM | 161s | 3.57s | 13.9s |
| n_features=64, n_components=64, n_data=1,000,000 | OOM | OOM | 44.4s | 527s |
| n_features=256, n_components=256, n_data=100,000 | OOM | OOM | NAN | 686s |
| n_features=256, n_components=16, n_data=1,000,000 | OOM | OOM | 60s | 454s |

### Notes:
- OOM: Out Of Memory
- NAN: Covar contains NaN
- k-loop: almost the same as original `GaussianMixture`, except 
```python
var = torch.sum((x - mu).unsqueeze(-1).matmul((x - mu).unsqueeze(-2)) * resp.unsqueeze(-1), dim=0,
  keepdim=True) / torch.sum(resp, dim=0, keepdim=True).unsqueeze(-1) + eps
```
in `_m_step` is replaced with 
```python
var = torch.empty(1, self.n_components, self.n_features, self.n_features, device=x.device, dtype=resp.dtype)
eps = (torch.eye(self.n_features) * self.eps).to(x.device)
for i in range(self.n_components):
  sub_mu = mu[:, i, :]
  sub_resp = resp[:, i, :]
  sub_x_mu = (x - sub_mu).squeeze(1)
  outer = torch.matmul(sub_x_mu[:, :, None], sub_x_mu[:, None, :])
  outer_sum = torch.sum(outer * sub_resp[:, :, None], dim=0, keepdim=True)
  sub_var = outer_sum / resp_sum[i] + eps
  var[:, i, :, :] = sub_var
```
