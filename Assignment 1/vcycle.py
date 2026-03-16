"""Multigrid V-cycle for 2D Poisson (Python translation of VCycle.m).

Solves the 2D Poisson problem on (0,1)^2 with Dirichlet boundary conditions:

    \\Delta u = f

using a geometric multigrid V-cycle with weighted Jacobi smoothing.

The script reproduces the MATLAB driver loop: repeated V-cycles, plotting the
current solution after each outer iteration.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def _vec_to_grid(v: np.ndarray, m: int) -> np.ndarray:
    return v.reshape((m, m), order="F")


def _grid_to_vec(g: np.ndarray) -> np.ndarray:
    return g.reshape((-1,), order="F")


def form_rhs(m: int, f_func, u_func) -> np.ndarray:
    """Form RHS for the 5-point Laplacian with Dirichlet boundary conditions.

    Discretization uses the Laplacian stencil:
        (u_{i-1,j} + u_{i+1,j} + u_{i,j-1} + u_{i,j+1} - 4 u_{i,j}) / h^2 = f

    Boundary values are moved to the RHS.

    Returns a vector of length m*m in MATLAB-compatible (column-major) ordering.
    """
    h = 1.0 / (m + 1)
    F = np.zeros((m, m), dtype=float)

    xs = np.linspace(h, 1.0 - h, m)
    ys = np.linspace(h, 1.0 - h, m)

    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            rhs = f_func(x, y)

            if i == 0:
                rhs -= u_func(0.0, y) / h**2
            if i == m - 1:
                rhs -= u_func(1.0, y) / h**2
            if j == 0:
                rhs -= u_func(x, 0.0) / h**2
            if j == m - 1:
                rhs -= u_func(x, 1.0) / h**2

            F[i, j] = rhs

    return _grid_to_vec(F)


def apply_laplacian(U: np.ndarray, m: int) -> np.ndarray:
    """Apply the 5-point Laplacian operator to the interior grid unknowns."""
    h = 1.0 / (m + 1)
    Ug = _vec_to_grid(U, m)

    left = np.zeros_like(Ug)
    right = np.zeros_like(Ug)
    down = np.zeros_like(Ug)
    up = np.zeros_like(Ug)

    left[1:, :] = Ug[:-1, :]
    right[:-1, :] = Ug[1:, :]
    down[:, 1:] = Ug[:, :-1]
    up[:, :-1] = Ug[:, 1:]

    AUg = (left + right + down + up - 4.0 * Ug) / h**2
    return _grid_to_vec(AUg)


def smooth_weighted_jacobi(U: np.ndarray, omega: float, nsmooth: int, m: int, F: np.ndarray) -> np.ndarray:
    """Weighted Jacobi smoothing for A U = F, where A is the 5-point Laplacian."""
    h = 1.0 / (m + 1)
    Ug = _vec_to_grid(U, m)
    Fg = _vec_to_grid(F, m)

    for _ in range(nsmooth):
        left = np.zeros_like(Ug)
        right = np.zeros_like(Ug)
        down = np.zeros_like(Ug)
        up = np.zeros_like(Ug)

        left[1:, :] = Ug[:-1, :]
        right[:-1, :] = Ug[1:, :]
        down[:, 1:] = Ug[:, :-1]
        up[:, :-1] = Ug[:, 1:]

        jacobi_update = (left + right + down + up - (h**2) * Fg) / 4.0
        Ug = (1.0 - omega) * Ug + omega * jacobi_update

    return _grid_to_vec(Ug)


def restrict_injection(r_fine: np.ndarray, m: int) -> np.ndarray:
    """Inject fine-grid residual to the coarse grid (m = 2*mc + 1)."""
    rg = _vec_to_grid(r_fine, m)
    r_coarse = rg[1::2, 1::2]
    return _grid_to_vec(r_coarse)


def prolong_bilinear(e_coarse: np.ndarray, mc: int) -> np.ndarray:
    """Bilinear prolongation from coarse grid (mc x mc) to fine grid (m x m).

    Fine size is m = 2*mc + 1.
    """
    ec = _vec_to_grid(e_coarse, mc)
    m = 2 * mc + 1
    ef = np.zeros((m, m), dtype=float)

    # Inject coarse points onto fine grid at odd indices.
    ef[1::2, 1::2] = ec

    # Interpolate in x-direction (even i, odd j)
    ef[0::2, 1::2] = 0.5 * (
        np.pad(ec, ((1, 0), (0, 0)), mode="constant")[:-1, :] +
        np.pad(ec, ((0, 1), (0, 0)), mode="constant")[1:, :]
    )

    # Interpolate in y-direction (odd i, even j)
    ef[1::2, 0::2] = 0.5 * (
        np.pad(ec, ((0, 0), (1, 0)), mode="constant")[:, :-1] +
        np.pad(ec, ((0, 0), (0, 1)), mode="constant")[:, 1:]
    )

    # Interpolate centers (even i, even j): average of 4 surrounding coarse points
    ec_pad = np.pad(ec, ((1, 1), (1, 1)), mode="constant")
    ef[0::2, 0::2] = 0.25 * (
        ec_pad[0:mc + 1, 0:mc + 1] +
        ec_pad[1:mc + 2, 0:mc + 1] +
        ec_pad[0:mc + 1, 1:mc + 2] +
        ec_pad[1:mc + 2, 1:mc + 2]
    )

    return _grid_to_vec(ef)


def vcycle(U: np.ndarray, omega: float, nsmooth: int, m: int, F: np.ndarray) -> np.ndarray:
    """One multigrid V-cycle for A U = F on an m x m interior grid."""
    l2m = np.log2(m + 1)
    if l2m != round(l2m):
        raise ValueError("m+1 must be a power of 2")
    if U.shape != (m * m,):
        raise ValueError("U must have length m*m")

    h = 1.0 / (m + 1)

    if m == 1:
        # (-4 u) / h^2 = F  =>  u = -(h^2/4) * F
        return np.array([-(h**2) * F[0] / 4.0], dtype=float)

    # 1) Pre-smooth
    Unew = smooth_weighted_jacobi(U, omega, nsmooth, m, F)

    # 2) Residual
    r = F - apply_laplacian(Unew, m)

    # 3) Restrict residual
    mc = (m - 1) // 2
    r_coarse = restrict_injection(r, m)

    # 4) Coarse-grid correction (approximately solve A e = r)
    e_coarse = vcycle(np.zeros(mc * mc), omega, nsmooth, mc, r_coarse)

    # 5) Prolongate the error
    e_fine = prolong_bilinear(e_coarse, mc)

    # 6) Update solution
    Unew = Unew + e_fine

    # 7) Post-smooth
    Unew = smooth_weighted_jacobi(Unew, omega, nsmooth, m, F)

    return Unew


def plotU(m: int, U: np.ndarray) -> None:
    h = 1.0 / (m + 1)
    x = np.linspace(h, 1.0 - h, m)
    y = np.linspace(h, 1.0 - h, m)
    X, Y = np.meshgrid(x, y, indexing="ij")

    Ug = _vec_to_grid(U, m)

    fig = plt.gcf()
    fig.clf()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Ug, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_title("Computed solution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("U")


def main() -> None:
    # Exact solution and RHS (chosen so that Laplacian(u) = f)
    u = lambda x, y: np.exp(np.pi * x) * np.sin(np.pi * y) + 0.5 * (x * y) ** 2
    f = lambda x, y: x**2 + y**2

    m = 2**6 - 1
    U = np.zeros(m * m)

    F = form_rhs(m, f, u)

    omega = 2.0 / 3.0
    epsilon = 1.0e-10

    plt.ion()
    plt.figure(figsize=(8, 6))

    for i in range(1, 101):
        R = F - apply_laplacian(U, m)
        rel_resid = np.linalg.norm(R) / np.linalg.norm(F)
        print(f"*** Outer iteration: {i:3d}, rel. resid.: {rel_resid:e}")

        if rel_resid < epsilon:
            break

        U = vcycle(U, omega, 3, m, F)
        plotU(m, U)
        plt.pause(0.5)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
