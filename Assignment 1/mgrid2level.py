"""Two-level multigrid demo (Python translation of mgrid2level.m).

This script solves a 1D Poisson-type problem on a fine grid, performs weighted
Jacobi smoothing, applies a single coarse-grid correction, then post-smooths.

It reproduces the original MATLAB script's plotting and pause behavior.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve


def build_1d_laplacian_dirichlet(m: int, h: float) -> sparse.spmatrix:
    """Return the standard 1D second-difference matrix with Dirichlet BCs."""
    data = np.zeros((3, m), dtype=float)
    data[0, 1:] = 1.0 / h**2
    data[1, :] = -2.0 / h**2
    data[2, :-1] = 1.0 / h**2
    return sparse.spdiags(data, [-1, 0, 1], m, m, format="csr")


def weighted_jacobi_step(A: sparse.spmatrix, F: np.ndarray, U: np.ndarray, omega: float) -> np.ndarray:
    """One weighted Jacobi iteration for A U = F."""
    diagA = A.diagonal()
    r = F - A @ U
    return U + omega * (r / diagA)


def main() -> None:
    a = 0.5

    psi = lambda x: 20 * np.pi * x**3
    psidot = lambda x: 3 * 20 * np.pi * x**2
    psiddot = lambda x: 2 * 3 * 20 * np.pi * x

    f = lambda x: -20 + a * psiddot(x) * np.cos(psi(x)) - a * psidot(x) ** 2 * np.sin(psi(x))
    u = lambda x: 1 + 12 * x - 10 * x**2 + a * np.sin(psi(x))

    m = 255
    h = 1.0 / (m + 1)

    A = build_1d_laplacian_dirichlet(m, h)

    X = np.linspace(h, 1.0 - h, m)
    F = f(X)
    F = F.astype(float, copy=True)
    F[0] -= u(0.0) / h**2
    F[-1] -= u(1.0) / h**2

    Uhat = u(X)
    Ehat = spsolve(A, F) - Uhat

    omega = 2.0 / 3.0

    plt.ion()
    fig, (ax_u, ax_e) = plt.subplots(1, 2, figsize=(12, 4))
    fig.set_facecolor("white")

    U2 = 1 + 2 * X

    for i in range(1, 11):
        U2 = weighted_jacobi_step(A, F, U2, omega)
        E2 = U2 - Uhat

        ax_u.cla()
        ax_u.plot(X, Uhat, "b-", label="Uhat")
        ax_u.plot(X, U2, "gx", label="U2")
        ax_u.set_xlabel("x")
        ax_u.set_ylabel("U")
        ax_u.set_title(f"Iter={i:4d}")
        ax_u.tick_params(labelsize=16)

        ax_e.cla()
        ax_e.plot(X, Ehat, "b-", label="Ehat")
        ax_e.plot(X, E2, "gx", label="E2")
        ax_e.set_xlabel("x")
        ax_e.set_ylabel("E")
        ax_e.set_title(f"Iter={i:4d}")
        ax_e.tick_params(labelsize=16)

        plt.pause(1)

    input("Paused after pre-smoothing. Press Enter to continue...")

    r = F - A @ U2

    m_coarse = (m - 1) // 2
    h_coarse = 1.0 / (m_coarse + 1)

    r_coarse = r[1::2]
    assert len(r_coarse) == m_coarse

    A_coarse = build_1d_laplacian_dirichlet(m_coarse, h_coarse)

    e_coarse = spsolve(A_coarse, -r_coarse)

    e = np.zeros_like(r)
    e[1::2] = e_coarse

    # Linear interpolation back to fine grid at the remaining points.
    for idx in range(0, m, 2):
        e_left = e[idx - 1] if idx - 1 >= 0 else 0.0
        e_right = e[idx + 1] if idx + 1 < m else 0.0
        e[idx] = 0.5 * (e_left + e_right)

    U2 = U2 - e
    E2 = U2 - Uhat

    ax_u.cla()
    ax_u.plot(X, Uhat, "b-")
    ax_u.plot(X, U2, "gx")
    ax_u.set_xlabel("x")
    ax_u.set_ylabel("U")
    ax_u.set_title("After coarse grid projection")
    ax_u.tick_params(labelsize=16)

    ax_e.cla()
    ax_e.plot(X, Ehat, "b-")
    ax_e.plot(X, E2, "gx")
    ax_e.set_xlabel("x")
    ax_e.set_ylabel("E")
    ax_e.set_title("After coarse grid projection")
    ax_e.tick_params(labelsize=16)

    plt.pause(0.1)
    input("Paused after coarse grid projection. Press Enter to continue...")

    for i in range(1, 11):
        U2 = weighted_jacobi_step(A, F, U2, omega)
        E2 = U2 - Uhat

        ax_u.cla()
        ax_u.plot(X, Uhat, "b-")
        ax_u.plot(X, U2, "gx")
        ax_u.set_xlabel("x")
        ax_u.set_ylabel("U")
        ax_u.set_title(f"Iter={i:4d}")
        ax_u.tick_params(labelsize=16)

        ax_e.cla()
        ax_e.plot(X, Ehat, "b-")
        ax_e.plot(X, E2, "gx")
        ax_e.set_xlabel("x")
        ax_e.set_ylabel("E")
        ax_e.set_title(f"Iter={i:4d}")
        ax_e.tick_params(labelsize=16)

        plt.pause(1)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
