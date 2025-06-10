from typing import Callable, Any
import scipy.sparse.linalg
import numpy as np
import scipy.sparse as sp
import itertools
from typing import Iterable
from numpy.typing import NDArray, ArrayLike
import time
import dataclasses



"""State: A sorted tuple of integers, denoted which modes are occupied with excitations"""
State = tuple[int, ...]

"""Basis: A dictionary mapping states to positions in a vector state"""
Basis = dict[State, int]

"""Representation for operators"""
Operator = sp.csr_matrix

State_2 = np.typing.NDArray  # had to change it because STATE was defined two different ways 
Vector = np.ndarray


def construct_basis(qubits: int, bosons: int, excitations: int) -> Basis:
    """
    Construct the basis for a given number of 'qubits' and 'bosons' with a fixed
    number of 'excitations'.

    Creates the basis of all possible occupied modes for the given number of
    `excitations`. Each state is represented by a sorted tuple of the modes that
    are occupied. Modes 0 up to (qubits-1) are hard-core boson modes and thus
    can only appear once. All other modes are ordinary bosons and may host 0 up
    to `excitations`.

    Args:
        qubits (int): Number of qubits or hard-core boson modes >= 0
        bosons (int): Number of bosonic modes >= 0
        excitations (int): Number of excitations >= 0

    Returns:
        basis: Map from configurations to an index in the Hilbert space basis
    """

    def make_bosonic_states(n_modes: int, excitations: int):
        return itertools.combinations_with_replacement(np.arange(n_modes), excitations)

    def select_hardcore_boson_states(qubits: int, states: Iterable) -> Iterable:
        return itertools.filterfalse(lambda x: unphysical_state(x, qubits), states)

    return {
        v: i
        for i, v in enumerate(
            select_hardcore_boson_states(
                qubits, make_bosonic_states(qubits + bosons, excitations)
            )
        )
    }


def unphysical_state(configuration: State, qubits: int) -> bool:
    """Given a sorted list of occupied modes, check whether a qubit mode
    appears more than once.

    Args:
        configuration (State): Sorted tuple with the occupied modes
        qubits (int): Number of hard-core boson modes >= 0

    Returns:
        bool: False if state is physical, True if it is not.
    """
    last = -1
    for mode in configuration:
        if mode >= qubits:
            return False
        if last == mode:
            return True
        last = mode


def number_operator(basis: Basis, mode: int) -> Operator:
    """Create the number operator for the given 'mode' in this 'basis'.

    Args:
        basis (Basis): basis of bosonic states
        mode (int): mode index (>= 0)

    Returns:
        sp.csr_matrix: diagonal matrix representing the occupation of 'mode'
    """
    L = len(basis)
    rows = np.arange(L)
    occupation = np.zeros((L,))
    for state, ndx in basis.items():
        occupation[ndx] = state.count(mode)
    return sp.csr_matrix((occupation, (rows, rows)), shape=(L, L))


def mode_occupations(basis: Basis, wavefunction: ArrayLike) -> NDArray[np.double]:
    """Compute the average of the modes occupations for all modes.

    Assume 'basis' is the bosonic basis for 'N' modes and that 'wavefunction'
    is a 1D vector for a state in this basis. Then 'mode_occupation' will
    return a vector of size 'N' with the average of the occupation number
    operators for each mode.

    If 'wavefunction' is an N-dimensional array, we assume that the first index
    is associated to the physical dimension of the basis and the same task
    is performed for all values of the 2 to N indices.

    Args:
        basis (Basis): basis of bosonic states
        wavefunction (ArrayLike): 1D wavefunction, or N-D collection of them
    Returns:
        occupations (NDArray): 1D vector of occupation numbers, or N-D collection
        of the computations for different wavefunctions.
    """
    wavefunction = np.asarray(wavefunction)
    num_modes = max(max(state) for state in basis) + 1
    probability = np.abs(wavefunction.reshape(len(basis), -1)) ** 2
    output = np.zeros((num_modes, probability.shape[1]))
    for state, ndx in basis.items():
        for mode in state:
            output[mode, :] += probability[ndx, :]
    return output.reshape(num_modes, *wavefunction.shape[1:])


def probability_amplitudes(basis: Basis, wavefunction: ArrayLike) -> NDArray[np.double]:
    """Compute the average of the modes occupations for all modes.

    Assume 'basis' is the bosonic basis for 'N' modes and that 'wavefunction'
    is a 1D vector for a state in this basis. Then 'probability_amplitudes' will
    return a vector of size 'N' with the probability amplitude of each of the modes.
    The occupation number would then be np.abs(probability_amplitudes)**2.

    Args:
        basis (Basis): basis of bosonic states
        wavefunction (ArrayLike): 1D wavefunction, or N-D collection of them
    Returns:
        probability_amplitudes (NDArray): 1D vector of probability amplitudes
        numbers, or N-D collection of the computations for different wavefunctions.
    """
    wavefunction = np.asarray(wavefunction)
    num_modes = max(max(state) for state in basis) + 1
    probability_amplitude = wavefunction.reshape(len(basis), -1)
    output = np.zeros((num_modes, probability_amplitude.shape[1]), dtype="complex128")
    for state, ndx in basis.items():
        for mode in state:
            output[mode, :] += probability_amplitude[ndx, :]
    return output.reshape(num_modes, *wavefunction.shape[1:])


def move_excitation_operator(
    origin_mode: int, destination_mode: int, basis: Basis
) -> sp.csr_matrix:
    """
    Create a sparse matrix representation of an operator that moves an
    excitation from mode 'origin' to mode 'destination'.

    Args:
        origin (int): Index of the origin mode
        destination (int): Index of the destination mode
        basis (dict): Collection of physical states (see: construct_basis)

    Returns:
        Operator (sp.csr_matrix): Matrix representation of the quantum operator
    """

    row = []
    column = []
    coefficient = []

    for state in basis:
        origin_occupation = state.count(origin_mode)
        if origin_occupation:
            ndx = state.index(origin_mode)
            transformed_state = tuple(
                sorted(state[:ndx] + state[ndx + 1 :] + (destination_mode,))
            )
            if transformed_state in basis:
                destination_occupation = transformed_state.count(destination_mode)
                row.append(basis[transformed_state])
                column.append(basis[state])
                coefficient.append(np.sqrt(origin_occupation * destination_occupation))

    return sp.csr_matrix((coefficient, (row, column)), shape=(len(basis), len(basis)))


def diagonals_with_energies(basis: Basis, frequencies: np.ndarray) -> sp.dia_matrix:
    """_summary_

    Args:
        basis (dict): _description_
        frequencies (np.ndarray): _description_

    Returns:
        sp.dia_matrix: _description_
    """
    energy = np.empty(len(basis))  # initialize the energy coresponding to each vector.

    for occupation, pos in basis.items():
        energy[pos] = np.sum(frequencies[list(occupation)])

    return sp.diags(energy)


def concatenate_basis(qubits: int, bosons: int, excitations: int) -> Basis:
    """
    Create a basis with a variable number of excitations, from 0 up to 'excitations',
    using the given number of 'qubits' and 'bosons' modes.

    Args:
        qubits (int): Number of qubits or hard-core boson modes >= 0
        bosons (int): Number of bosonic modes >= 0
        excitations (int): Number of excitations of the biggest subspace >= 0

    Returns:
        basis: Collection of all the states that constitute the basis properly sorted.
    """
    Basis = {}

    for excitations in range(excitations + 1):
        if excitations == 0:
            Basis[()] = (
                0  # For the subspace of 0 excitations we manually create the empty tuple corresponding to vacuum.
            )

        else:
            Basis_subspace = (
                {}
            )  # We initialize the variable Basis for a particular subspace

            index_0 = len(
                Basis
            )  # Very important. The indexation of the new subspace must begin where the previous left

            def make_bosonic_states(n_modes: int, excitations: int):
                return itertools.combinations_with_replacement(
                    np.arange(n_modes), excitations
                )

            def select_hardcore_boson_states(qubits: int, states: Iterable) -> Iterable:
                return itertools.filterfalse(
                    lambda x: unphysical_state(x, qubits), states
                )

            for i, v in enumerate(
                select_hardcore_boson_states(
                    qubits, make_bosonic_states(qubits + bosons, excitations)
                )
            ):
                Basis_subspace[v] = i + index_0

                Basis.update(
                    Basis_subspace
                )  # Update is a sort of append for dictionaries.

    return Basis


def erase(mode: int, basis: Basis) -> sp.csr_matrix:
    """Creates a sparse matrix representation of an operator that erases an excitation
    from 'mode'.

    This function creates the sparse matrix representation of a Fock anihilation
    operator in the given 'basis'.

    For this function to make sense, 'basis' must contain states from 0 up to a
    maximum number of excitations. Otherwise, when we remove 'mode' we will not find
    a good state to map it to.

    Args:
        mode (int): Index of the mode from which the excitation is going to be removed
        basis (Basis): Collection of physical states (see: construct_basis)

    Returns:
        Operator (sp.csr_matrix): Matrix representation of the quantum operator
    """
    row = []
    column = []
    coefficient = []
    for state in basis:
        # We run over all states in the basis. If the 'mode' is present
        # we construct a new state where we have removed _one_ occurrence of
        # this mode, thus eliminating a particle.
        count = state.count(mode)
        if count:
            # Since the modes are sorted in the state, we can assume the outcome
            # is sorted.
            mode_position = state.index(mode)
            transformed = tuple(state[:mode_position] + state[mode_position + 1 :])
            row.append(basis[transformed])
            column.append(basis[state])
            coefficient.append(np.sqrt(count))

    return sp.csr_matrix((coefficient, (row, column)), shape=(len(basis), len(basis)))

@dataclasses.dataclass
class Waveguide:
    """Class describing regularly spaced emitters in a finite waveguide.

    Parameters
    ----------
    Nqubits: int
        Number of emitters (larger than 1)
    positions: list[float]
    Delta: float
        Emitter frequency
    T: float
        Time for a photon to travel among neighboring emitters.
    excitations: int = 1
        Number of excitations for the simulation.
    Nmodes: int = 100
        Number of photonic modes that participate in the simulation.
    model: str = 'ring'
        It can be 'ring', 'obc', 'centered', depending on whether the
        waveguide is closed ('ring'), or whether it is opened and the
        qubits are either on one side ('obc'), or 'centered'
    """

    positions: list[float] | np.ndarray | None = None
    Nqubits: int | None = None
    Delta: float = 4.0
    Nmodes: int | None = None
    T: float = 400.0
    cutoff: float = 30.0
    model: str = "ring"
    excitations: int = 1
    v: float = 1.0
    g: float | None = None
    gamma: float | None = 0.1
    bandwidth: float = 20

    FSR: float = 0.0
    "Free spectral range, computed from delay"
    wk: np.ndarray = dataclasses.field(default_factory=lambda: np.asarray([1.0]))
    "Sequence of frequencies, computed from delay, Delta and Nmodes"
    phase: float = 0.0
    "Phase acquired by a photon travelling to a neighbor side, computed from delay and Delta"
    basis: Basis | None = None
    "Basis of states computed after the problem has been defined"

    def __post_init__(self):
        #
        # The number of qubits can be deduced from the vector of
        # positions, if provided. If this is absent, `Nqubits` must be givn
        # and the positions are equispaced according to `T`
        #
        if self.positions is not None:
            self.Nqubits = len(self.positions)
        elif self.Nqubits is not None:
            self.positions = self.T * np.arange(self.Nqubits)
        else:
            raise Exception(
                "Either a list of qubit positions or a value for Nqubits must be provided"
            )
        if self.Nqubits < 0:
            raise Exception("We need at least one emitter")
        if self.Nqubits > 10:
            raise Exception("Too many emitters for efficient simulation.")
        #
        # If the waveguide is finite, we may want to "center" the qubits to make
        # the model more symmetric
        #
        if self.model not in ["ring", "obc", "centered-obc"]:
            raise Exception(f"Invalid model variety {self.model}")
        x = np.asarray(self.positions)
        if self.model == "centered-obc":
            x = x - np.average(x)
        if "obc" in self.model:
            if np.any(x < 0) or np.any(x > self.T / 2):
                raise Exception("Qubit positions outside the waveguide")
        #
        # The effective coupling is deduced from the desired decay rate,
        # a unit that is frequently set in our simulations. The quality of
        # this estimation for the linear dispersion relation is verified below
        #
        if self.g is not None:
            self.gamma = np.pi * self.g * self.g * self.Delta * self.T / 4.0
        else:
            self.g = np.sqrt(4.0 * self.gamma / (np.pi * self.Delta * self.T))
            if self.model == "ring":
                self.g /= np.sqrt(2)
        #
        # The free spectral range is given by the time for a photon to run around
        # the waveguide.
        self.FSR = 2 * np.pi / self.T
        #
        # The number of modes can either be given, or estimated by how many times
        # the bandwidth of a typical photon must be contained in the simulation.
        # The quality of this estimate is demonstrated below for a single qubit
        if self.Nmodes is None:
            self.Nmodes = max(int(2 * self.bandwidth * self.gamma / self.FSR), 10)
        elif self.Nmodes < 10:
            raise Exception("Must provide a large enough number of photon modes.")
        #
        # Here we define the frequencies as equispaced values in a linear dispersion
        # relation with `v` typically 1 (assigned above). If the problem is in a
        # ring, we need two modes per frequency.
        index = int(self.Delta / self.FSR)
        nmin = max(0.0, index - self.Nmodes // 2)
        nmax = self.Nmodes // 2 + index
        self.wk = self.FSR * np.arange(nmin, nmax + 1)
        self.k = self.wk / self.v
        if self.model == "ring":
            self.wk = np.concatenate((self.wk[::-1], self.wk))
            self.k = np.concatenate((-self.k[::-1], self.k))
        self.Nmodes = len(self.wk)
        self.gk = self.g * np.sqrt(self.wk) * np.exp(-self.wk / self.cutoff)
        #
        t = time.process_time()
        self.basis = construct_basis(
            qubits=self.Nqubits, bosons=self.Nmodes, excitations=self.excitations
        )

    def number_state(self, mode: int) -> Operator:
        """Return the operator that gives the number of bosons for the given
        photon mode."""
        if mode not in range(self.Nmodes):
            raise Exception("Invalid photon mode index")
        return number_operator(self.basis, mode + self.Nqubits)

    def qubit_P1(self, qubit: int) -> Operator:
        """Return the projector onto an excited state of the given qubit."""
        if qubit not in range(self.Nqubits):
            raise Exception("Invalid qubit index")
        return number_operator(self.basis, qubit)

    def couplings(self, qubit: int) -> Vector:
        """Return the vector of couplings of a qubit to all photonic modes."""
        if qubit not in range(self.Nqubits):
            raise Exception("Invalid qubit index")
        if self.model == "ring":
            return np.exp(-1j * self.k * self.positions[qubit] * self.v) * self.gk
        else:
            return np.cos(self.k * self.positions[qubit] * self.v) * self.gk

    def frequencies(self) -> Vector:
        """Vector of frequencies for all photon states."""
        return self.wk

    def Hamiltonian(self) -> Operator:
        """Hamiltonian of all qubits interacting with all photons in the RWA limit."""
        H = self.Delta * sum(
            number_operator(self.basis, i) for i in range(self.Nqubits)
        ) + sum(
            wi * number_operator(self.basis, i + self.Nqubits)
            for i, wi in enumerate(self.wk)
        )
        Hint = sum(
            gki * move_excitation_operator(i, k + self.Nqubits, self.basis)
            for i in range(self.Nqubits)
            for k, gki in enumerate(self.couplings(i))
        )
        H += Hint + Hint.conj().T
        return H

    def qubit_state(self, bits: str | None = None) -> State:
        """Quantum state in which only the given qubits are excited.

        `str` is a string of 0's and 1's denoting which qubits are excited.
        We assume all photonic modes are in the vacuum state. If `str` is
        not given, we assume a state with 1's in the first qubits, where the
        number of 1's is given by the number of excitations we fixed initially.
        """
        # If no value is given, we fill with 1's the first qubits
        if bits is None:
            bits = ("1" * self.excitations) + ("0" * (self.Nqubits - self.excitations))
        elif len(bits) != self.Nqubits:
            raise Exception("Wrong qubit state to start the simulation.")
        if bits.count("1") != self.excitations:
            raise Exception(
                f"Bit string {bits} does not contain the required number of excitations."
            )
        # We create a tuple with the indices of all qubits that are excited. This
        # is used to consult our database of quantum states
        state = tuple(np.int64(index) for index, a in enumerate(bits) if a == "1")
        D = len(self.basis)
        output = np.zeros(D)
        output[self.basis[state]] = 1.0
        return output
    
def evolve(
    H, state, T, steps: int = 100, callback: Callable[[float, State_2], Any] | None = None
) -> tuple[float, Any]:
    def default_callback(t, state):
        return state

    if callback is None:
        callback = default_callback
    times = np.linspace(0, T, steps + 1)
    output = [None] * len(times)
    lastt = 0.0
    for i, t in enumerate(times):
        dt = t - lastt
        if dt:
            state = scipy.sparse.linalg.expm_multiply((-1j * dt) * H, state)
        output[i] = callback(t, state)
        lastt = t
    return times, np.asarray(output)