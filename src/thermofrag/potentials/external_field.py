"""Chemical-potential head mu_theta(y) with Laplace last layer.

Implements the property external field - mu(y) . phi(m, x) of eq (2).
Calibrated by the thermodynamic-identity loss L_mu, eq (7).

The Laplace last layer gives an analytic posterior over the last linear
weights conditional on a held MAP estimate of the rest of the network
(MacKay 1992 / Daxberger 2021). We use the diagonal Hessian approximation
(Generalized Gauss-Newton on an L2 regression log-likelihood), which for
mean_head = Linear(hidden -> n_properties) with shared-feature trunk gives
a per-weight posterior variance of::

    Var(W_ji) ~= 1 / (prior_prec + sum_n feats_n[i]^2)

identical across output rows j. The predictive variance at a new y is then
``sum_i trunk(y)[i]^2 * Var(W_ji)``, implemented below in ``predictive_variance``.
"""
from __future__ import annotations

from typing import Callable, Iterable

import torch
import torch.nn as nn


class ChemicalPotentialHead(nn.Module):
    def __init__(self, n_properties: int, hidden: int = 256):
        super().__init__()
        self.n_properties = n_properties
        self.hidden = hidden
        self.trunk = nn.Sequential(
            nn.Linear(n_properties, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden, n_properties)
        # Diagonal posterior variance of mean_head weights, flattened to [n_properties*hidden].
        # Under GGN on MSE the per-row posterior is identical, so we broadcast at query time.
        # Initialize to a broad prior so predictive_variance is non-degenerate before fitting.
        self.register_buffer("laplace_diag", torch.ones(n_properties * hidden))
        self.register_buffer("_laplace_fitted", torch.zeros((), dtype=torch.bool))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Return mu(y), shape [B, n_properties]."""
        return self.mean_head(self.trunk(y))

    @torch.no_grad()
    def fit_laplace(
        self,
        y_iter: Iterable[torch.Tensor],
        prior_prec: float = 1.0,
    ) -> None:
        """Fit a diagonal Laplace posterior over ``mean_head`` weights.

        ``y_iter`` yields batches of property-target tensors ``y`` of shape
        [B, n_properties]. The GGN diagonal of ``mean_head`` under MSE equals
        ``sum_n trunk(y_n)**2`` (feature outer products collapse to diag).

        After this call, ``predictive_variance`` returns calibrated epistemic
        variance on ``mu(y)``. Re-run whenever ``mean_head`` is retrained.
        """
        device = self.laplace_diag.device
        fisher = torch.zeros(self.hidden, device=device)
        any_batch = False
        for y in y_iter:
            y = y.to(device, non_blocking=True)
            feats = self.trunk(y)  # [B, hidden]
            fisher += (feats * feats).sum(dim=0)
            any_batch = True
        if not any_batch:
            raise ValueError("fit_laplace got an empty iterator")
        inv_prec = 1.0 / (prior_prec + fisher)  # [hidden]
        # Broadcast across n_properties output rows.
        post_var = inv_prec.unsqueeze(0).expand(self.n_properties, self.hidden).reshape(-1)
        self.laplace_diag.copy_(post_var)
        self._laplace_fitted.fill_(True)

    def predictive_variance(self, y: torch.Tensor) -> torch.Tensor:
        """Diagonal predictive variance of mu(y), shape [B, n_properties].

        Implements ``Var[mu_j(y)] = sum_i feats(y)[i]**2 * Var(W_ji)``.
        Works whether or not ``fit_laplace`` has been called — before fitting
        ``laplace_diag`` is ones (prior only), giving a broad default.
        """
        feats = self.trunk(y)  # [B, hidden]
        post_var = self.laplace_diag.view(self.n_properties, self.hidden)  # [P, H]
        # [B, 1, H] * [1, P, H] -> sum over H -> [B, P]
        return (feats.unsqueeze(1).pow(2) * post_var.unsqueeze(0)).sum(dim=-1)

    # Convenience wrapper used by evaluation code; callable-style.
    def log_norm_from_features(self, phi: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return mu(y) . phi(m) per row, shape [B]. Matches H^external term."""
        return (self.mean_head(self.trunk(y)) * phi).sum(-1)


def make_laplace_y_iter(phi_values: torch.Tensor, batch_size: int = 256):
    """Helper: chunk a (N, n_properties) tensor into an iterator of batches.

    Typical use::
        phi = torch.stack([compute_phi(m) for m in data])
        head.fit_laplace(make_laplace_y_iter(phi))
    """
    N = phi_values.shape[0]
    for start in range(0, N, batch_size):
        yield phi_values[start : start + batch_size]
