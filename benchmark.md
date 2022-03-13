# Benchmark
GPU: Tesla T4 (16GM DRAM)

- covariance_type = "full"
- init_params = "random"
- n_ter = 20
- delta = 0

| setup | original | k-loop | optimized (single) | optimized (double) |
| --- | --- | --- | --- | --- |
| n_features=16, n_components=16, n_data=100,000 | 6.9 | 6.9s | 0.5s | 3.44s |
| n_features=16, n_components=16, n_data=1,000,000 | OOM | 68.8s | 3.7s | 34.0s |
| n_features=64, n_components=64, n_data=100,000 | OOM | 161s | 3.57s | 13.9s |
| n_features=64, n_components=64, n_data=1,000,000 | OOM | OOM | 44.4s | 527s |
| n_features=256, n_components=256, n_data=100,000 | OOM | OOM | NAN | 686s |
| n_features=256, n_components=16, n_data=1,000,000 | OOM | OOM | 60s | 454s |

- OOM: Out Of Memory
- NAN: Covar contains NaN
