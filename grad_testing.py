import torch
import torch.nn as nn

torch.manual_seed(42)

B = 10
T = 100
N = 2

neural_network = nn.Linear(1, 2, False).float()

# t_col = torch.stack([1 / x * torch.arange(T, dtype=float, requires_grad=True) for x in range(1, B+1)]).float()
t_col = torch.linspace(0,200,1000).reshape(10,100).requires_grad_()
# shuffle t_col batches to mimic network batching
t_col = t_col[torch.randperm(10),:]
zp_col_hat = torch.zeros([B, T, N])
# zp_col_hat[:, :, 0] = torch.sin(t_col)
# zp_col_hat[:, :, 1] = torch.cos(t_col)
zp_col_hat[:, :, 0] = torch.sin(t_col) + torch.cos(2*t_col)
zp_col_hat[:, :, 1] = torch.cos(t_col) - 2*torch.sin(2*t_col)
# zp_col_hat = neural_network(t_col.unsqueeze(-1))

dxdt_ = torch.zeros_like(zp_col_hat)
for i in range(zp_col_hat.shape[2]):
    dxdt_[:, :, i] = \
    torch.autograd.grad(zp_col_hat[:, :, i], t_col, torch.ones_like(zp_col_hat[:, :, i]), create_graph=True)[0]
# for i in range(zp_col_hat.shape[2]):
#     for b in range(zp_col_hat.shape[0]):
#         dxdt_[b, :, i] = \
#         torch.autograd.grad(zp_col_hat[b, :, i], t_col[b,:], torch.ones_like(zp_col_hat[b, :, i]), create_graph=True)[0]
print(dxdt_.size())
print(dxdt_)

import matplotlib.pyplot as plt

gradients = dxdt_.detach().cpu().numpy()
original = zp_col_hat.detach().cpu().numpy()

plt.figure(1)
plt.subplot(311)
plt.plot(gradients[0, :, 0], label='gradient (should be cos(t))')
plt.plot(original[0, :, 0], label='original (should be sin(t))')
plt.legend()
plt.subplot(312)
plt.plot(gradients[1, :, 1], label='gradient (should be -sin(t))')
plt.plot(original[1, :, 1], label='original (should be cos(t))')
plt.legend()
plt.subplot(313)
plt.plot(gradients[2, :, 0], label='gradient (should be cos(t))')
plt.plot(original[2, :, 0], label='original (should be sin(t))')
plt.legend()
plt.show()