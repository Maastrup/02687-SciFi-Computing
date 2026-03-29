import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from finite_diff_functions import l2_norm_grid, smooth, coarsen, scatter

# MoMS solution to the Poisson problem in 2D
def u_func(x,y):
    return np.sin(4*np.pi*(x+y)) + np.cos(4*np.pi*x*y)

def f_func(x,y):
    term1 = -32*np.pi**2*np.sin(4*np.pi*(x+y))
    term2 = -16*np.pi**2*(x**2 + y**2)*np.cos(4*np.pi*x*y)
    return term1 + term2

# LHS of a discretized Poisson problem
def Amult(U, m):
    h = 1.0 / (m + 1)
    AU = np.zeros_like(U)

    for j in range(m):
        for i in range(m):
            k = i + j*m   # 2D -> 1D index
            
            center = U[k]

            left = U[k-1] if i > 0 else 0.0
            right = U[k+1] if i < m - 1 else 0.0
            down = U[k-m] if j > 0 else 0.0
            up = U[k+m] if j < m - 1 else 0.0

            # Discrete Laplacian (Δ_h): (neighbors - 4*center)/h^2
            AU[k] = (left + right + down + up - 4 * center) / h**2

    return AU

def form_rhs(m, f_func, u_func):
    '''
        Form the discretized RHS of the INTERIOR points of a Poisson problem in 2D with rectangular domain
        
        # INPUTS
            m: number of INTERIOR points in the discretized domain
            f_func: RHS as a analytical function of (x,y) of the BVP
            u_func: DC boundary as analytical function of (x,y)

        # OUTPUT
            b: discretized RHS of the interior points in the discretized domain
    '''

    h = 1.0/(m+1)
    b = np.zeros(m**2)

    for j in range(m):
        for i in range(m):

            k = j*m + i
            x = (i+1)*h
            y = (j+1)*h

            b[k] = f_func(x,y)

            # left boundary
            if i == 0:
                b[k] -= u_func(0,y)/h**2

            # right boundary
            if i == m-1:
                b[k] -= u_func(1,y)/h**2

            # bottom boundary
            if j == 0:
                b[k] -= u_func(x,0)/h**2

            # top boundary
            if j == m-1:
                b[k] -= u_func(x,1)/h**2

    return b


## VCYCLE
def vcycle(A, R, P, u, f, l):
    '''
        # Parameters
            A: A matrix free operator computing discretized Poisson matrix-vector product A(u) = -A*u
            R: Function | (u, m) -> u with len(u) = m//2 Restriction operator coarsening the grid vectors to length m/2 x m/2
            P: Function | (u, m) -> u with len(u) = m**2 Prolongation operator interpolating grid vectors from length m/2 x m/2 -> m x m
            u: current guess at solution to Poisson problem
            f: RHS of discretized Poisson problem
            l: initial number of nodes along x- and y-axis
        # Output
            u: 
    '''
    if l == 1:
        # l = 1 ==> A^-1 = [-h**2/4] ==> u = u[0] = -h**2 / 4 * f[0]
        # l = 1 ==> h = 1/2
        #       ==> u = u[0] = -1/8 * f[0]
        u[0] = -1 / 16 * f[0]
    else:
        for _ in range(3): # nu_pre in algorithm
            u = smooth(u, 0.65306, 2**l - 1, f)
        
        # Residual for Δ_h u = f
        r = f - A(u, 2**l - 1)
        r = R(r, 2**l - 1)
        e = np.zeros_like(r)
        e = vcycle(A, R, P, e, r, l - 1)
        e = P(e, 2**l - 1)
        u += e
        
        for _ in range(3):
            u = smooth(u, 0.65306, 2**l - 1, f)
    
    return u

# multi grid loop
l = 5 # levels
m = 2**l - 1
print(f"m = {m}")
x = np.linspace(0, 1, m)
y = np.linspace(0,1, m)
X,Y = np.meshgrid(x, y, indexing='ij')
U0 = (1 + 2*X + 2*Y)
for i in range(m):
    for j in range(m):
        assert U0[i,j] - U0.flatten()[i + j*m] < 1e-15

U0 = U0.flatten()
F = form_rhs(m, f_func, u_func)
assert F.shape == U0.shape

kmax = 10

k = 0
rk = F - Amult(U0, m)  # residual r = f - Δ_h u
Uk = np.copy(U0)
EPS = 1e-4
for k in range(kmax):
    norm = l2_norm_grid(rk, 1/(m+1)) # h = 1/m+1
    print(f"||r||_2: {norm}")
    if norm < EPS: break

    Uk = vcycle(Amult, coarsen, scatter, Uk, F, l)
    rk = F - Amult(Uk, m)
    
