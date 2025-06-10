import numpy as np 
from numpy.typing import NDArray, ArrayLike
from qutip import *

"""
Here we define the colapse operators taking into account the crossed terms. It is important to highlight here that the exponential 
'np.exp(1.j*np.abs(ϕ[n]-ϕ[m]))' contributes to both coherent and incoherent coupligs as detailed in Alejandro-Porras' work.
"""


def CoherentCoupling(ϕ: np.ndarray, ɣ_0: float):  # A matrix with elements Δ_nm
    output = np.ones([len(ϕ), len(ϕ)])
    for n in range(0, len(ϕ)):
        for m in range(0, len(ϕ)):
            output[n][m] = ɣ_0 * np.sin(np.abs(ϕ[n] - ϕ[m]))
    return output


def CollectiveDecay(ϕ: np.ndarray, ɣ_0: float):  # A matrix with elements Γ_nm
    output = np.ones([len(ϕ), len(ϕ)])
    for n in range(0, len(ϕ)):
        for m in range(0, len(ϕ)):
            output[n][m] = ɣ_0 * np.cos(np.abs(ϕ[n] - ϕ[m]))
    return output


def Sminus(
    N_atoms: int,
):  # To save space, we define Sminus and Splus can be obtained by transpose conjugation
    output = []  # Arising
    for i in range(0, N_atoms):
        if N_atoms == 1:
            output.append(sigmam())
        if N_atoms > 1:
            qeyeN = [qeye(2) for j in range(0, N_atoms - 1)]
            qeyeN.insert(i, sigmam())
            output.append(tensor(qeyeN))
    return output


def H_int(N_atoms: int, ϕ: np.ndarray, ɣ_0: float):
    Δ = CoherentCoupling(ϕ, ɣ_0)
    output = 0
    S_minus = Sminus(N_atoms)
    if N_atoms == 1:
        output += 0 * qeye(2)
    if N_atoms > 1:
        for n in range(0, N_atoms):
            for m in range(0, N_atoms):
                if m != n:
                    output += Δ[n][m] * S_minus[n] * S_minus[m].dag()
    return output


def colapse_operators(N_atoms: int, ϕ: np.ndarray, ɣ_0: float):
    Γ = CollectiveDecay(ϕ, ɣ_0)
    output = []
    S_minus = Sminus(N_atoms)
    for n in range(0, N_atoms):
        for m in range(0, N_atoms):
            output.append(
                0.5
                * (Γ[n][m])
                * (
                    2.0 * spre(S_minus[n]) * spost(S_minus[m].dag())
                    - 1.0 * spre(S_minus[m].dag() * S_minus[n])
                    - 1.0 * spost(S_minus[m].dag() * S_minus[n])
                )
            )
    return output


def ME_solution(
    N_atoms: int, ϕ: np.ndarray, ɣ_0: float, t_max: float, rho_0, observ="populations"
):  # observables has to be a list of quantum objects
    if observ == "populations":
        σ = []
        for n in range(0, N_atoms):
            I = [qeye(2) for n0 in range(0, N_atoms)]
            I[n] = sigmam()
            σ.append(tensor(I))
        O = [σ[n].dag() * σ[n] for n in range(0, N_atoms)]
        tlist = np.linspace(0, t_max, 500)
        c_ops = colapse_operators(N_atoms, ϕ, ɣ_0)
        simulation = mesolve(H_int(N_atoms, ϕ, ɣ_0), rho_0, tlist, c_ops, e_ops=O)
    output = simulation.expect
    simulation = 0  # to save memory in case we do not need to do postprocessing using the master equation
    return [tlist, output]