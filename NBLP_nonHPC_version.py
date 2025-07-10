# Ariel Norambuena 1
# Vicente Chomali-Castro 2
# 1 Centro Multidisciplinario de Física, Universidad Mayor,
# 2 Department of Physics, University of Illinois Urbana-Champaign


##################################################################################

from joblib import Parallel, delayed
import numpy as np
from qutip import *
from scipy.optimize import differential_evolution
import os
import time

#Print options; keep enabled in case of data printing to avoid data loss/truncation.
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

up = basis(2, 0)
down = basis(2, 1)
ex = basis(2, 0)
ey = basis(2, 1)

def Initial_States(params):
    global up, down, ex, ey

    theta1, phi1, theta2, phi2, theta3, phi3, theta4, phi4 = params

    Psi_spin1 = np.sin(theta1/2) * up + np.cos(theta1/2) * np.exp(1j * phi1) * down
    Psi_orb1  = np.sin(theta2/2) * ex + np.cos(theta2/2) * np.exp(1j * phi2) * ey
    Psi_spin2 = np.sin(theta3/2) * up + np.cos(theta3/2) * np.exp(1j * phi3) * down
    Psi_orb2  = np.sin(theta4/2) * ex + np.cos(theta4/2) * np.exp(1j * phi4) * ey

    psi1_0 = tensor(Psi_orb1, Psi_spin1)
    psi2_0 = tensor(Psi_orb2, Psi_spin2)

    Psi1_0 = tensor(basis(N, nQ), psi1_0)
    Psi2_0 = tensor(basis(N, nQ), psi2_0)

    return Psi1_0, Psi2_0

def Hamiltonian_collapse_operators():
    global up, down, ex, ey, Lp, Jp
    a = tensor(destroy(N), tensor(qeye(2), qeye(2)))
    Hph = wc * a.dag() * a
    Lm = Lp.dag()
    Hint = g1 * (a + a.dag()) * (Lp + Lm) - 1j * g2 * (a + a.dag()) * (Lm - Lp)
 
    # Dissipative part
    c_ops = []
    
    rate = kappa * (1 + n_th_a)
    if rate > 0.0:
        c_ops.append(np.sqrt(rate) * a)
    
    rate = kappa * n_th_a
    if rate > 0.0:
        c_ops.append(np.sqrt(rate) * a.dag())
    
    rate = gamma_abs
    if rate > 0.0:
        c_ops.append(np.sqrt(rate) * Jp)
    
    Jm = Jp.dag()
    rate = gamma_em
    if rate > 0.0:
        c_ops.append(np.sqrt(rate) * Jm)

    return Hph, Hint, c_ops

def tracedist(A, B):
    X = A - B
    X = X.dag() * X

    vals = X.eigenenergies(sparse=True)
	
    return float(np.real(0.5 * np.sum(np.sqrt(np.abs(vals)))))

def TraceDistance(result1, result2):
    rho_t1 = [ptrace(rho1, 1) for rho1	 in result1.states]
    rho_t2 = [ptrace(rho2, 1) for rho2 in result2.states]

    D_t = []

    for rho1, rho2 in zip(rho_t1, rho_t2):
        dist = tracedist(rho1, rho2)
        D_t.append(dist)

    return np.array(D_t)

def Dynamical_NM(D_t, tlist, rnd):

    dt = tlist[1] - tlist[0]
    dTraceDist = np.diff(D_t) / dt
    dTraceDist = np.append(dTraceDist, 0)

    DynNM = 0.5 * np.trapz(np.abs(dTraceDist) + dTraceDist, dx=dt)
    DynNM = np.round(DynNM,rnd)

    return DynNM

# Options for mesolve, QuTiP's master equation solver.
opts = Options(method='diag')

def run_simulation(params, H, tlist, rnd):
    Psi01, Psi02 = Initial_States(params)

    result1 = mesolve(H, Psi01, tlist, c_ops, [], options=opts)
    result2 = mesolve(H, Psi02, tlist, c_ops, [], options=opts)

    D_t = TraceDistance(result1, result2)
    NM = Dynamical_NM(D_t, tlist, rnd)
    print(NM)

    return -NM

def get_TraceDistance_NM(params, H, tlist, rnd):
    Psi01, Psi02 = Initial_States(params)

    result1 = mesolve(H, Psi01, tlist, c_ops, [], options=opts)
    result2 = mesolve(H, Psi02, tlist, c_ops, [], options=opts)

    D_t = TraceDistance(result1, result2)
    NM = Dynamical_NM(D_t, tlist, rnd)

    return D_t, NM

def Hamiltonian_SiV(Bx, By, Bz):
    HSiV = (-lSO * tensor(Lz, Sz) +
            gamma_x * tensor(sigma_z, Id) +
            gamma_y * tensor(sigma_x, Id) +
            f * gammaL * Bz * tensor(Lz, Id) +
            gammas * (Bx * tensor(Id, Sx) + By * tensor(Id, Sy) + Bz * tensor(Id, Sz)))
    HSiV = tensor(qeye(N), HSiV)
    return HSiV

# This is the tolerance for the optimization. A lower tolerance yields more reliable results.
tol = 0.001
# These are bounds of the vectors in the space to be optimized over to obtain the BLP.
bounds = [(0, np.pi), (0, 2*np.pi), (0, np.pi), (0, 2*np.pi), (0, np.pi), (0, 2*np.pi), (0, np.pi), (0, 2*np.pi)]

def compute_NM_for_combination(comb):
    # Get the specific combination for the current script number
    Bx, Bz = comb
    print(f"Calculating (Bx,Bz)=({Bx},{Bz}).")

    # Update SiV Hamiltonian
    HSiV = Hamiltonian_SiV(Bx, By, Bz) 
    H = HSiV + Hph + Hint

    # Low resolution optimization for optimal initial conditions for a certain combination of Bx and Bz
    tf = 30 / np.abs(g)
    tlist = np.linspace(0, tf, 10000) # in linspace(0, tf, N), execution time scales as N.
    
    rnd = 6 # Round to six decimal places for optimization

    try:
        opt = differential_evolution(run_simulation,
                                     x0=[1.565e+00, 3.324e+00, 3.069e+00, 3.773e-03,
                                         1.565e+00, 5.859e+00, 4.958e-02, 3.170e+00],
                                     bounds=bounds, args=(H, tlist, rnd), tol=tol, maxiter=400, workers=1)
        # The x0 values are arbitrary, so we decided to use the optimal parameters for the NBLP at (Bx,Bz)=(0,0).
        # Execution time scales as sqrt(maxiter).
        # Leave workers as 1, otherwise it will conflict with the parallel processing that is
        # later set up with JobLib.

        # Calculate degree of non-Markovianity with the optimal initial conditions just found
        tf = 100 / np.abs(g)
        tlist = np.linspace(0, tf, 100000)
        rnd = 6 # Round to six decimal places for final calculation with optimized parameters

        D_t, NM = get_TraceDistance_NM(opt.x, H, tlist, rnd)
    except:
        # If for some reason the calculation fails or one instantiation of a parallel process
        # crashes, the NBLP for the corresponding (Bx,Bz) combination will be set to zero.
        NM = 0

    # Save each result to a separate file.
    # Results are saved as they are calculated. That way, some data can always be recovered
    # in case of a code crash. It also allows the study of the data as it is generated; one
    # doesn't have to wait for the code to finish calculating all the data to see the data.
    script_dir = os.path.dirname(__file__)
    index = combs.index(comb)
    filename = os.path.join(script_dir, f"{index + 1}.txt")
    with open(filename, 'w') as file:
        file.write(f"{NM}")

    return NM


##################################################################################    
# ––– SETUP OF PHYSICAL SYSTEM –––

# Physical parameters of the system

hbar = 1.054571817 * 1e-34
kB = 1.380649 * 1e-23

f = 0.1

gammas = 2.8 * 1e+6
gammaL = gammas / 2
gamma_x = 2 * np.pi * 1 * 1e+9
gamma_y = 2 * np.pi * 1 * 1e+9

lSO = 2 * np.pi * 45 * 1e+9

wa = np.sqrt(lSO**2 + gamma_x**2 + gamma_y**2)

g1 = 1e-3 * wa
g2 = g1
g = g1 + 1j * g2

Q = 1e+5
T = 100
Delta = np.sqrt(lSO**2 + gamma_x**2 + gamma_y**2)
Nsiv = 1 / (np.exp((hbar * Delta) / (kB * T)) - 1)

gammaSiV = 2 * np.pi * 1.78 * 1e+6
gamma_abs = gammaSiV * Nsiv
gamma_em = gammaSiV * (Nsiv + 1)

N = 5 # Execution time scales as N^3.

n_th_a = 0
nQ = 1
wc = 0.003 * wa
kappa = wc / Q

By = 0
# The magnetic field is set to zero in the y-axis, as the heatmap will plot
# the NBLP against Bx and Bz.

# Operators of the system

sigma_x = ex * ey.dag() + ey * ex.dag()
sigma_y = -1j * ex * ey.dag() + 1j * ey * ex.dag()
sigma_z = ex * ex.dag() - ey * ey.dag()

Lz = sigma_y
Id = qeye(2)
sx = up * down.dag() + down * up.dag()
sy = -1j * up * down.dag() + 1j * down * up.dag()
sz = up * up.dag() - down * down.dag()
Sx = sx / 2
Sy = sy / 2
Sz = sz / 2
ep = (ex + 1j * ey) / np.sqrt(2)
em = (ex - 1j * ey) / np.sqrt(2)
e1 = tensor(em, down)
e2 = tensor(ep, up)
e3 = tensor(ep, down)
e4 = tensor(em, up)
Lp = e3 * e1.dag() + e2 * e4.dag()
Jp = e1 * e3.dag() + e2 * e4.dag()
Lp = tensor(qeye(N), Lp)
Jp = tensor(qeye(N), Jp)


##################################################################################    
# ––– SETUP OF HEATMAP RESOLUTION AND CALCULATION –––

NB = 71
# This is the resolution of one quadrant. One quadrant will have NB^2 data points.
# In the end, the heat map will be generated by reflecting this quadrant's data across
# the x and y axes, giving (NB + NB - 1)^2.
#      NB must be odd, 
# since only that way the lines
# x = 0 (Bz = 0) and y = 0 (Bx = 0) are included in the heatmap, and this are considered
# to be important lines. Also, our reflection of the data does not reflect the lines x = 0
# and y = 0, and therefore, again, NB must be odd.
# Execution time scales as NB^2.

#Figure 4(a) uses ranges: 0, 2.0 * 1e+2
#Figure 4(b) uses ranges: 0, 2.0 * 1e+1
Bxi, Bxf = 0, 2.0 * 1e+6
Bzi, Bzf = 0, 2.0 * 1e+6
Bx_values = np.linspace(Bxi, Bxf, NB)
Bz_values = np.linspace(Bzi, Bzf, NB)

combs = [(Bx, Bz) for Bx in Bx_values for Bz in Bz_values]

Hph, Hint, c_ops = Hamiltonian_collapse_operators()

start_index = 0
end_index = NB**2
combs_to_process = combs[start_index:end_index]

# Start parallel processing
start = time.time()
# njobs must be set to -1 to ensure that all CPU cores of the computer are used in the
# parallel computation to ensure a faster execution of the code.
# Execution time scales as n_jobs^{-1}.
results = Parallel(n_jobs=-1)(delayed(compute_NM_for_combination)(comb) for comb in combs_to_process)
execution_time = time.time() - start
print(f"Execution time: {execution_time} seconds")

# Print final output
print(f"Finished processing combinations {start_index} to {end_index - 1}.")
