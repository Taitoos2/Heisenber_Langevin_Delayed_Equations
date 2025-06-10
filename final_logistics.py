import numpy as np 
import matplotlib.pyplot as plt 
from scipy.sparse import kron, eye, csr_matrix
from dataclasses import dataclass, field
from typing import Callable, Optional


#------------ploting format-----------------------------------------

Colors_array = np.asarray([
    ["#08589e", "#4eb3d3"],  
    ["#91003f", "#c994c7"],  
    ["#006600", "#99FF99"],  
    ["#cc4c02", "#fe9929"],  
    ["black", "#E0E0E0"],    
    ["#990000", "#FFCCCC"],  
])

markerlist = ["o", "s", "D", "p", "8", "v", "p", "v", "^", "8"]  

def format_plot(n: int, mark: int = None):
    
    color_idx = n % len(Colors_array)
    marker_idx = n % len(markerlist)
    
    params = {
        'color': Colors_array[color_idx, 1],
        'markeredgecolor': Colors_array[color_idx, 0],
        'markeredgewidth': 1.5,
        'marker': markerlist[marker_idx],  # Añade el marcador correspondiente
    }
    
    if mark is not None:
        params['markevery'] = mark
    
    return params
# --------------------------- base functions used in class --------------------------------------------------- 

def sm_operator_s(i: int, n: int) -> np.ndarray:
	''' generator of sigma^{-}_{i} operators in the initial state (standard pauli matrices times the identity)'''
	return csr_matrix(mkron(np.eye(2**i), [[0, 0], [1, 0]], np.eye(2 ** (n - i - 1))))

def sm_operator(i: int, n: int) -> np.ndarray:
    ''' generator of sigma^{-}_{i} operators in the initial state (standard pauli matrices times the identity)'''
    return mkron(np.eye(2**i), [[0, 0], [1, 0]], np.eye(2 ** (n - i - 1))).astype(np.complex128)

def vector_basis(s: str) -> np.ndarray:
	if s == "1":
		return np.array([1, 0])
	if s == "0":
		return np.array([0, 1])
	if s == "+":
		return np.array([1, 1]) / np.sqrt(2)
	if s == "-":
		return np.array([1, -1]) / np.sqrt(2)
	
def mkron(A, *args):
	if len(args) == 0:
		return A
	return np.kron(A, mkron(args[0], *args[1:]))


def vector_state(chain: str):
	vectors = [vector_basis(s) for s in chain]
	state = vectors[0]
	for vector in vectors[1:]:
		state = np.kron(state, vector)
	return state

def sparse_prod(arr1,arr2):
	''' given to arrays of sparse matrices, returns the array result of matrix-multiplication '''
	return np.asarray([arr1[i]@arr2[i] for i,_ in enumerate(arr1)])

def show(arr):
	return [ np.abs(matriz.todense()) for matriz in arr]

def RK4(t,y,f,dt):
	''' Runge-Kutta integration '''
	k1=f(t,y)
	k2 =f(t+dt/2,y + dt*k1/2)
	k3 = f(t+dt/2,y+dt*k2/2)
	k4 = f(t+dt,y+dt*k3)
	return  y + (k1+2*k2+2*k3+k4)*(dt/6) 

#------------------ Two-level system integrator ----------------------------------------------------- 
@dataclass
class Spin_DDE_integrator:
	''' This class represents an approximation for the de-excitation operators (sigma^-) for each of the two-level 
	emitters placed in a chain inside an infinite-length bosonic waveguide that undergo a non-Markovian evolution. 
	These operators ignore the EM noise and therefore they belong to a 2**N dimensional hilbert space, with N being the 
	number of emitters in the chain. 
	  '''
	pr_check: bool = False 
	gamma: float = 1.0          # effective coupling between the emitter and the EM field.
	phi: float = 2 * np.pi      # phase acquired by the photon when travelling
	tau: float = 2              # time separation between the ends of the waveguide
	N_steps : int = 50         	# Number of timesteps between t = 0 and t = tau 
	initial_state : str = '111'       # initial state of the emitters 
	s_list: list = field(default_factory=list)          # list that stores values of sigma
	a_out_list: list = field(default_factory=list)      # list that stores values of a_out(t)
	t_list: list = field(default_factory=list)          # list that stores values of t
	
	def __post_init__(self):
		self.N = len(self.initial_state)  # Number of emitters in the waveguide 
		self.D = 2 ** self.N 
		self.dt = self.tau / self.N_steps
		if self.pr_check:
			self.dt = np.round(self.tau / self.N_steps,3)  # fixed decimals, for heatmap . In future I'll fix 
		s_initial = np.asarray([sm_operator_s(i,self.N).astype(np.complex128) for i in range(self.N)])
		self.s_list.append(s_initial)
		self.a_out_list.append(-0.5*np.sqrt(self.gamma)*s_initial)
		self.t_list.append(0.0)
	
	def rho_0(self):
		''' initial value of the density matrix '''
		vector= vector_state(self.initial_state)
		return csr_matrix(np.outer(vector,np.conj(vector)))
		
	def sz(self):
		sm = self.s_list[-1]
		sp = np.asarray([np.conjugate(matriz.T) for matriz in sm])
		return 2*sparse_prod(sp,sm) - np.asarray([csr_matrix(np.eye(self.D)) for i in range(self.N)])
	
	def a_in(self,t : float):
		''' this is the electromagnetic current that flows into each of the emitters. 
		needed for the calculation of the derivative and the output current.  '''
		n = int(t/self.tau)
		idx =  int(t/self.dt)
		current = np.asarray([csr_matrix((self.D,self.D)).astype(np.complex128) for i in range(self.N)])
		if n > 0 : 
			for i in range(self.N):
				for k in range(max(0,i - n),min(self.N ,i+n+1)):
					if i != k:
						current[i] += self.s_list[idx - abs(int(i-k))*self.N_steps][k]*np.exp(1j*self.phi*np.abs(i-k))
		return 0.5*np.sqrt(self.gamma)*current 
	 
	def derivative(self,t,y):
		dsdt  = 0.5*self.gamma*sparse_prod(self.sz(),y) 
		dsdt += np.sqrt(self.gamma) * sparse_prod(self.sz(),self.a_in(t))
		return dsdt
	
	def evolve(self,t_max):
		while self.t_list[-1]<=t_max:
			y_new=RK4(self.t_list[-1],self.s_list[-1],self.derivative,self.dt)
			self.t_list.append(self.t_list[-1]+self.dt)
			self.s_list.append(y_new)
			self.a_out_list.append(self.a_in(self.t_list[-1])+0.5*np.sqrt(self.gamma)*y_new)

	def excited_state(self):
		''' pop[n] is the population of the excited state of the n-th emitter, and pop[0] is t '''
		pop = [self.t_list]
		for n in range(self.N):
			sm = np.asarray(self.s_list)[:,n]
			sp = np.asarray([self.rho_0()@matriz.getH() for matriz in sm])
			pop.append( [matriz.diagonal().sum() for matriz in sparse_prod(sp,sm)] )
		return pop 	
	
	def output_current(self):
		''' note: this is only the output current in one direction. the calculus well deserves to be looked. '''
		I_out = [self.t_list]
		for n in range(self.N):
			a = np.asarray(self.a_out_list)[:,n]
			a_dag = np.asarray([self.rho_0()@matriz.getH() for matriz in a])
			I_out.append( [matriz.diagonal().sum() for matriz in sparse_prod(a_dag,a)] )
		return I_out
	
# ------------ Boson integrator --------------------------

@dataclass
class Boson_DDE_integrator:
	
	pr_check: bool = False 
	gamma: float = 1.0          # effective coupling between the emitter and the EM field.
	phi: float = 2 * np.pi      # phase acquired by the photon when travelling
	tau: float = 2              # time separation between the ends of the waveguide
	N_steps : int = 50         	# Number of timesteps between t = 0 and t = tau 
	initial_state : str = '111'       # initial state of the emitters 
	J_list: list = field(default_factory=list)          # list that stores values of sigma
	t_list: list = field(default_factory=list)          # list that stores values of t

	def __post_init__(self):
		self.N = len(self.initial_state)  # Number of emitters in the waveguide 
		self.dt = self.tau / self.N_steps	
		if self.pr_check:
			self.dt = np.round(self.tau / self.N_steps,4		)  # fixed decimals, for heatmap . In future I'll fix 
		self.J_list.append(np.eye(self.N).astype(np.complex128))
		self.t_list.append(0.0)
	
	def rho_0(self):
		''' initial value of the density matrix '''
		vector= vector_state(self.initial_state)
		return np.outer(vector,np.conj(vector))
		

	def a_in(self,t : float):
		''' this is the electromagnetic current that flows into each of the emitters. 
		needed for the calculation of the derivative and the output current.  '''
		n_delay = int(t/self.tau)
		idx =  int(np.round(t/self.dt,4))
		current = np.zeros((self.N,self.N)).astype(np.complex128)
		if n_delay > 0 : 
			for l in range(self.N):
				for m  in range(self.N):
					for n in range(max(0,l - n_delay),min(self.N ,l+n_delay+1)):
						if n != l:
							current[l,m] +=self.J_list[idx - abs(int(n-l))*self.N_steps][n,m]*np.exp(1j*self.phi*np.abs(l-n))
		return current 
	 
	def derivative(self,t,y):
		dsdt  = -0.5*self.gamma*y 
		dsdt += -0.5*self.gamma * self.a_in(t)
		return dsdt
	
	def evolve(self,t_max):
		while self.t_list[-1]<=t_max:
			y_new=RK4(self.t_list[-1],self.J_list[-1],self.derivative,self.dt)
			self.J_list.append(y_new)
			self.t_list.append(self.t_list[-1]+self.dt)

	def excited_state(self):
		''' pop[n] is the population of the excited state of the n-th emitter, and pop[0] is t '''
		arr = np.asarray(self.J_list)
		pop = [np.asarray(self.t_list)]
		for l in range(self.N):
			pop_l = np.zeros(len(self.t_list)).astype(np.complex128)
			for m in range(self.N):
				for n in range(self.N):
					pop_l += np.conjugate(arr[:,l,m])*arr[:,l,n] * np.trace(sm_operator(m,self.N).T@sm_operator(n,self.N)@self.rho_0())
			pop.append(pop_l)
		return pop 
	def output_current(self):
		''' output current on one of the ends of the chain. does NOT work like in the TLS case. '''
		current= np.zeros(len(self.t_list)).astype(np.complex128)
		t = np.asarray(self.t_list)
		J_arr = np.asarray(self.J_list)
		for l in range(self.N):
			for j in range(self.N):
				for m in range(self.N):
					for k in range(self.N):
						coef = np.trace(sm_operator(k,self.N).T@sm_operator(j,self.N)@self.rho_0())
						current += 0.25*self.gamma*np.conjugate(J_arr[:,m,k])*J_arr[:,l,j]*np.heaviside(t-l*self.tau,1) * np.heaviside(t-m*self.tau,1)*coef
		return [np.asarray(self.t_list), current ]

# -------------- Analytical for bosons (in a future) ------- 

import numpy.polynomial as poly

def generate_sigma_minus(N, n):
    sigma = csr_matrix(np.array([[0, 0], [1, 0]]))
    id = eye(2, format="csr")
    left = kron(eye(2**n, format="csr"), sigma)
    right = eye(2 ** (N - n - 1), format="csr")
    return kron(left, right).tocsr()

def Solve_analytical_new(
    N: int,
    n: int,
    initial: str,
    gamma: float,
    phi: float,
    T: float,
    t_max: float,
    dt: float,
) -> list:
    """Solves the dynamics for the n-th emitter on a chain of N boson-like emitters, with parameters
    -----------------------------------------------------------
    initial:    initial state of the emitters in the chain, a string combining '1', '0' , '+' and '-'
    gamma:      coupling between the EM field and the emitters (assumed to be identical for every emitter)
    delta:      Energy gap between the levels of the emitter (assumed to be identical for every emitter)
    t_max:      final time of integration
    dt:         time step.
    """
    delta = phi / T

    M = int(np.floor(t_max) / T)
    t = np.arange(0, t_max, dt)
    vector = vector_state(initial)

    if len(initial) != N:
        print("the initial state is not compatible with the number of emitters")
        return 0

    record = [poly.Polynomial(0) for j in range(0, N)]
    record[n] += 1
    record = [record]

    def integrate_s_qubit(s: int) -> poly.Polynomial:
        """returns the integrated sum of polynomials in a time interval [mT, (m+1)T] for a specific function f_s"""
        nt = len(record)
        result = poly.Polynomial(0)
        for j in range(0, N):
            dT = int(np.abs(s - j))
            if dT <= nt and j != s:
                result += (
                    record[-1 * dT][j](poly.Polynomial([-dT * T, 1]))
                    * (-gamma / 2)
                    * np.exp(
                        (1j * delta + gamma / 2) * T * dT
                    )  # este 0 hay que quitarlo, es para ver las cosas
                )
        result = result.integ()
        return result - result((nt) * T)

    def time_step():
        """performs an integration for every emitter"""
        new_list = []
        for s in range(0, N):
            new_list.append(integrate_s_qubit(s))
        return new_list

    def poly_to_fn(polis):
        """given an array of polynomials [p1(t),p2(t), p3(t)...]
        return p1(t) θ(t) + p2(t)θ(t-T)+ p3(t) θ(t-2T)+..."""
        sum = np.zeros(len(t), dtype=complex)
        for n, poli in enumerate(polis):
            sum += poli(t) * np.heaviside(t - n * T, 1)
        return sum

    for m in range(0, M):
        record.append(time_step())

    record = np.array(record).T
    f_functions = [
        np.exp((-gamma / 2 + 1j * delta) * t) * poly_to_fn(poli) for poli in record
    ]

    probability = np.zeros(len(t), dtype=complex)

    for i in range(0, N):
        for j in range(0, N):
            sigma_i_dag = generate_sigma_minus(N, i).conjugate().transpose()
            sigma_j = generate_sigma_minus(N, j)
            probability += (
                (vector.conjugate() @ sigma_i_dag @ sigma_j @ vector)
                * np.conjugate(f_functions[i])
                * f_functions[j]
            )

    return t, probability, f_functions


def Boson_analyt(
    N: float,
    state: str,
    gamma: float,
    ϕ_0: float,
    T: float,
    t_max: float,
    dt: float,
):
    """solves the population of the excited level for each one of the N emitters
    returns [t, p1, p2 ... pn ]"""
    J_func = []
    t, pop, J_0 = Solve_analytical_new(N, 0, state, gamma, ϕ_0, T, t_max, dt)
    output = [t, pop]
    J_func.append(J_0)
    for j in range(1, N):
        _, dummie, J_i = Solve_analytical_new(N, j, state, gamma, ϕ_0, T, t_max, dt)
        output.append(dummie)
        J_func.append(J_i)
    return output, J_func
# -------------------------- code for calculation of output current in bosons (to be improved)-------------

def output_current(
    t: np.ndarray, J_func: list, gamma: float, T: float, initial: str
) -> np.ndarray:
    """first attempt at getting the output current."""
    N = len(J_func[0])  # Number of emitters
    vector = vector_state(initial)
    current = np.zeros(len(t), dtype=complex)
    for k in range(0, N):
        sigma_k_dag = generate_sigma_minus(N, k).conjugate().transpose()
        for l in range(0, N):
            sigma_l = generate_sigma_minus(N, l)
            coef = vector.conjugate() @ sigma_k_dag @ sigma_l @ vector
            for n in range(0, N):
                J_nl = np.zeros(len(t), dtype=complex)
                J_nl[t >= n * T] += J_func[n][l][0 : len(t[t >= n * T])]
                for m in range(0, N):
                    J_mk = np.zeros(len(t), dtype=complex)
                    J_mk[t >= m * T] += J_func[m][k][0 : len(t[t >= (m * T)])]
                    current += coef * np.conj(J_mk) * J_nl
    return gamma * current / 4


def change_J_format(J: np.ndarray) -> list:
    J_new = []
    for n in range(0, J.shape[-1]):
        J_n = []
        for m in range(0, J.shape[-1]):
            J_n.append(J[:, n, m])
        J_new.append(J_n)
    return J_new