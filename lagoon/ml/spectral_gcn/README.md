# Spectral GCN
NOT IMPLEMENTED YET

[Original paper](https://arxiv.org/abs/1609.02907)

## Original Algorithm
```
# Given: Data matrix X_0 of shape N x F_0, N = number of nodes, F_0 = number of features, A = processed adjacency matrix.
for k in 1..K:
    X_k = act(A * X_k-1 * W_k)
Return X_K
```
Considerations:
- Adjacency matrix for the whole graph can become very large. To mitigate:
    - Throw away some nodes with low degree
    - Use [FastGCN](https://arxiv.org/abs/1801.10247)
- Changing the graph will need computing a new adjacency matrix.

