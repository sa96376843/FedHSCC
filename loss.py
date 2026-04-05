# *************************************************************************
# Copyright 2023 ByteDance and/or its affiliates
#
# Copyright 2023 FedDecorr Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# *************************************************************************

# =========================================================================
# MODIFICATIONS by Jianhang Feng on 2026-04-05
# - Added class FedHSCCLoss (Hypergraph Structural Consistency Constraint loss)
# =========================================================================

import torch
import torch.nn as nn


class FedDecorrLoss(nn.Module):

    def __init__(self):
        super(FedDecorrLoss, self).__init__()
        self.eps = 1e-8

    def _off_diagonal(self, mat):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = mat.shape
        assert n == m
        return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x):
        N, C = x.shape
        if N == 1:
            return 0.0

        x = x - x.mean(dim=0, keepdim=True)
        x = x / torch.sqrt(self.eps + x.var(dim=0, keepdim=True))

        corr_mat = torch.matmul(x.t(), x)

        loss = (self._off_diagonal(corr_mat).pow(2)).mean()
        loss = loss / N

        return loss


class FedHSCCLoss(nn.Module):
    def __init__(self, beta=5, gamma=10):
        super(FedHSCCLoss, self).__init__()
        self.eps = 1e-8
        self.beta = beta
        self.gamma = gamma

    def forward(self, x):
        N, C = x.shape
        if N <= 1 or C <= 1:
            return torch.tensor(0.0, device=x.device)

        # Standardize and compute correlation matrix
        x_centered = x - x.mean(dim=0, keepdim=True)
        x_norm = x_centered / torch.sqrt(self.eps + x.var(dim=0, keepdim=True))
        corr_mat = torch.mm(x_norm.T, x_norm) / (N - 1)

        # Get off-diagonal elements
        mask_offdiag = ~torch.eye(C, dtype=torch.bool, device=x.device)
        corr_offdiag = corr_mat[mask_offdiag]

        # Compute 25% quantile of absolute off-diagonal values as tau
        # Thus the top 25% most correlated will be considered strong
        tau = torch.quantile(torch.abs(corr_offdiag), 0.75)

        # Hierarchical penalty computation
        mask_strong = torch.abs(corr_offdiag) > tau

        loss_strong = self.gamma * torch.sum(corr_offdiag[mask_strong] ** 2)
        loss_weak = self.beta * torch.sum(torch.abs(corr_offdiag[~mask_strong]))

        return (loss_strong + loss_weak) / (C * (C - 1)) / 2