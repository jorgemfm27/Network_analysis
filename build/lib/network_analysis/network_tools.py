import numpy as np
from scipy.optimize import minimize


def Z_capacitor(Freq: np.array, C: float):
    assert len(Freq.shape), 'Frequency should be given as 1D array'
    w = 2*np.pi*Freq
    Z_in = -1j/(w*C)
    return Z_in

def Z_inductor(Freq: np.array, L: float):
    assert len(Freq.shape), 'Frequency should be given as 1D array'
    w = 2*np.pi*Freq
    Z_in = 1j*w*L
    return Z_in

def Z_term_transline(Freq: np.array, l: float, v: float, Z0: float, ZL: float = 0):
    assert len(Freq.shape), 'Frequency should be given as 1D array'
    beta = 2*np.pi*Freq/v
    Z_in = Z0*( ZL + 1j*Z0*np.tan(beta*l) )/( Z0 + 1j*ZL*np.tan(beta*l) )
    return Z_in

def Z_parallel(Za: np.array, Zb: np.array):
    Z_par = Za*Zb/(Za+Zb)
    return Z_par

def M_series_impedance(Z: np.array):
    assert len(Z.shape), 'Impedance should be given as 1D array'
    M = np.zeros((len(Z), 2, 2), dtype=np.complex_)
    for i in range(len(Z)):
        M[i] = np.array([[1, Z[i]],
                         [0,   1]])
    return M

def M_parallel_impedance(Z: np.array):
    assert len(Z.shape), 'Impedance should be given as 1D array'
    M = np.zeros((len(Z), 2, 2), dtype=np.complex_)
    for i in range(len(Z)):
        M[i] = np.array([[     1, 0],
                         [1/Z[i], 1]])
    return M

def M_series_transline(Freq: np.array, l: float, v: float, Z0: float):
    assert len(Freq.shape), 'Frequency should be given as 1D array'
    beta = 2*np.pi*Freq/v
    M = np.zeros((len(beta), 2, 2), dtype=np.complex_)
    for i in range(len(beta)):
        M[i] = np.array([[np.cos(beta[i]*l), 1j*Z0*np.sin(beta[i]*l)],
                         [1j/Z0*np.sin(beta[i]*l), np.cos(beta[i]*l)]])
    return M

def M_inductively_coupled_impedance(Freq: np.array, Z: float, L1: float, L2: float, M:float):
    '''
    An ABCD matrix modulating transmission through a parallel incuctively coupled impedance Z
         _________
    ----|--mmm----|-----
        |  mmm    |
        |  |      |
        | | |     |
        | |Z|     |
        | |_|     |
        |  |      |
        |__V______|
    '''
    w = 2*np.pi*Freq
    Z_in = 1j*w*L1 + (1j*w*M)**2 / (Z+1j*w*L2)
    # Compute ABCD matrix
    M = M_series_impedance(Z_in)
    return M

def multiply_matrices(M_list):
    for i, m in enumerate(M_list):
        assert len(m.shape) == 3 and m.shape[1:]==(2,2), f'M_list[{i}] must have shape (n, 2, 2)'
    # Initialize the result as the first array
    result = M_list[0]
    # Compute the dot product sequentially with each subsequent array
    for arr in M_list[1:]:
        result = np.einsum('nij,njk->nik', result, arr)
    return result

def extract_S_pars(M, Zgen):
    _shape = M.shape
    assert len(_shape) == 3 and _shape[1:]==(2,2), 'M should have shape (n, 2, 2)'
    A, B, C, D = M[:,0,0], M[:,0,1], M[:,1,0], M[:,1,1]
    # Compute scattering parameters
    S11 = (A + B/Zgen - C*Zgen - D)/( A + B/Zgen + C*Zgen + D )
    S12 = 2*(A*D - B*C)/( A + B/Zgen + C*Zgen + D )
    S21 = 2/( A + B/Zgen + C*Zgen + D )
    S22 = (-A + B/Zgen - C*Zgen + D)/( A + B/Zgen + C*Zgen + D )
    return S11, S12, S21, S22

def extract_Z_pars(M, ZL):
    _shape = M.shape
    assert len(_shape) == 3 and _shape[1:]==(2,2), 'M should have shape (n, 2, 2)'
    # Compute Z matrix
    Z11 = M[:,0,0]/M[:,1,0]
    Z12 = (M[:,0,0]*M[:,1,1] - M[:,0,1]*M[:,1,0])/M[:,1,0]
    Z21 = 1/M[:,1,0]
    Z22 = M[:,1,1]/M[:,1,0]
    # Compute input and output impedance
    Zin = Z11 - (Z12*Z21)/(Z22+ZL)
    Zout = Z22 - (Z12*Z21)/(Z11+ZL)
    return Z11, Z12, Z21, Z22, Zin, Zout

def get_fft_from_pulse(time_axis, pulse):
    # compute frequency axis
    dt = time_axis[1]-time_axis[0]
    n = time_axis.shape[-1]
    freq_axis = np.fft.fftfreq(n, d=dt)
    # compute fft
    fft = np.fft.fft(pulse)
    return freq_axis, fft

def get_pulse_from_fft(freq_axis, fft):
    # compute frequency axis
    df = freq_axis[1]-freq_axis[0]
    n = freq_axis.size
    time_axis = np.fft.fftfreq(n, d=df)
    # compute inverse fft
    pulse = np.fft.ifft(fft)
    # sort vectors
    time_axis = np.fft.fftshift(time_axis)
    time_axis -= np.min(time_axis)
    return time_axis, pulse

def square_pulse(time, pulse_duration, frequency, pulse_pad=0):
    return np.sin(time*frequency*2*np.pi+np.pi/5)*(np.heaviside(time-pulse_pad, 1)-np.heaviside(time-pulse_duration-pulse_pad, 1))

def _solve_nan(array):
    # interpolate nan values in array
    idxs_nan = np.where(np.isnan(array))[0]
    for i in idxs_nan:
        array[i] = np.mean([array[[i-1, i+1]]])
    return array



class Network():
    '''
    Class used to construct and model a 2 port network.
    '''
    def __init__(self, Zgen: float):
        # Elements along network
        self.elements = []
        # Loading of ports when extracting scattering and impedance parameters
        self.Zgen = Zgen

    def _add_element(self, M_function):
        '''
        Add an ABCD matrix element to network.
        M_function should have a frequency argument.
        '''
        self.elements.append(M_function)

    def _substitute_element(self, M_function, element_idx):
        '''
        Substitute an ABCD matrix element in existing network.
        M_function should have a frequency argument.
        '''
        self.elements[element_idx] = M_function

    def get_S_parameters(self, frequency):
        '''
        Compute the S parameters of the network
        over a frequency range.
        '''
        # convert frequency into array
        if isinstance(frequency, float):
            frequency = np.array([frequency])
        # List of ABCD matrices of elements in network
        M_elements = []
        for element in self.elements:
            # ABCD matrix of element
            m_element = element(frequency=frequency)
            M_elements.append(m_element)
        # Multiply all ABCD matrices
        M_system = multiply_matrices([*M_elements])
        # Compute scattering parameters
        S11, S12, S21, S22 = extract_S_pars(M_system, Zgen=self.Zgen)
        return S11, S12, S21, S22

    def get_Z_parameters(self, frequency):
        '''
        Compute the Z parameters of the network
        over a frequency range.
        '''
        # convert frequency into array
        if isinstance(frequency, float):
            frequency = np.array([frequency])
        # List of ABCD matrices of elements in network
        M_elements = []
        for element in self.elements:
            # ABCD matrix of element
            m_element = element(frequency=frequency)
            M_elements.append(m_element)
        # Multiply all ABCD matrices
        M_system = multiply_matrices([*M_elements])
        # Compute scattering parameters
        Z11, Z12, Z21, Z22, Zin, Zout = extract_Z_pars(M_system, ZL=self.Zgen)
        return Z11, Z12, Z21, Z22, Zin, Zout

    def get_signal_response(self, time, signal):
        '''
        Get the signal response of network to an input signal.
        Computes the scattered signal via S parameters.
        '''
        # express signal in frequency domain
        freq_axis, fft = get_fft_from_pulse(time_axis=time, pulse=signal)
        # compute Scattering parameters of network along frequency domain
        s11, s12, s21, s22 = self.get_S_parameters(frequency=freq_axis)
        # Resolve nans in scattering parameters
        s11, s12, s21, s22 = _solve_nan(s11), _solve_nan(s12), _solve_nan(s21), _solve_nan(s22)
        # compute scattered spectrum
        fft_11, fft_12, fft_21, fft_22 = s11*fft, s12*fft, s21*fft, s22*fft
        fft_11, fft_12, fft_21, fft_22 = np.conjugate(s11)*fft, np.conjugate(s12)*fft, np.conjugate(s21)*fft, np.conjugate(s22)*fft
        # recover signal in time domain
        time, signal_11 = get_pulse_from_fft(freq_axis=freq_axis, fft=fft_11)
        time, signal_12 = get_pulse_from_fft(freq_axis=freq_axis, fft=fft_12)
        time, signal_21 = get_pulse_from_fft(freq_axis=freq_axis, fft=fft_21)
        time, signal_22 = get_pulse_from_fft(freq_axis=freq_axis, fft=fft_22)
        return time, signal_11, signal_12, signal_21, signal_22

    def add_capacitance(self, C: float, element_type: str, element_idx=None):
        '''
        Add a capacitive element to the network.
        Can be made in series or in parallel.
        '''
        # Assertion
        assert element_type in ['series', 'parallel'], 'type must be "series" or "parallel"'
        # add a series capacitance element to network
        if element_type == 'series':
            Z = lambda frequency: Z_capacitor(Freq=frequency, C=C)
            M_C = lambda frequency: M_series_impedance(Z=Z(frequency))
        else:
            Z = lambda frequency: Z_capacitor(Freq=frequency, C=C)
            M_C = lambda frequency: M_parallel_impedance(Z=Z(frequency))
        # if element already exists
        if element_idx is None:
            self._add_element(M_C)
        else:
            self._substitute_element(M_C, element_idx)

    def add_inductance(self, L: float, element_type: str, element_idx=None):
        '''
        Add an inductive element to the network.
        Can be made in series or in parallel.
        '''
        # Assertion
        assert element_type in ['series', 'parallel'], 'type must be "series" or "parallel"'
        # add a series capacitance element to network
        if element_type == 'series':
            Z = lambda frequency: Z_inductor(Freq=frequency, L=L)
            M_L = lambda frequency: M_series_impedance(Z=Z(frequency))
        else:
            Z = lambda frequency: Z_inductor(Freq=frequency, L=L)
            M_L = lambda frequency: M_parallel_impedance(Z=Z(frequency))
        # if element already exists
        if element_idx is None:
            self._add_element(M_L)
        else:
            self._substitute_element(M_L, element_idx)

    def add_transmission_line(self, length: float, Z0: float, phase_velocity: float, element_idx=None):
        '''
        Add a series transmission line element to network.
        '''
        M_t = lambda frequency: M_series_transline(Freq=frequency, l=length, v=phase_velocity, Z0=Z0)
        # if element already exists
        if element_idx is None:
            self._add_element(M_t)
        else:
            self._substitute_element(M_t, element_idx)

    def add_capacitively_coupled_hanger(self, length: float, Z0: float, phase_velocity: float, Z_termination: float, C_coupling: float, element_idx=None):
        '''
        Add a parallel terminated transmission line
        in series with a capacitor element to network.
        '''
        # impedance of capacitor
        Z_C = lambda frequency: Z_capacitor(Freq=frequency, C=C_coupling)
        # impedance of terminated transmission line
        Z_t = lambda frequency: Z_term_transline(Freq=frequency, l=length, v=phase_velocity, Z0=Z0, ZL=Z_termination)
        # ABCD matrix of parallel elements
        M_t = lambda frequency: M_parallel_impedance(Z=Z_C(frequency)+Z_t(frequency))
        # if element already exists
        if element_idx is None:
            self._add_element(M_t)
        else:
            self._substitute_element(M_t, element_idx)

    def add_inductively_coupled_hanger(self, length: float, Z0: float, phase_velocity: float, Z_termination: float, L_line: float, L_hanger: float, M_inductance: float, element_idx=None):
        '''
        Add a parallel terminated transmission line
        inductively coupled to main line of network.
        '''
        # impedance of terminated transmission line
        Z_t = lambda frequency: Z_term_transline(Freq=frequency, l=length, v=phase_velocity, Z0=Z0, ZL=Z_termination)
        # ABCD matrix of parallel elements
        M_t = lambda frequency: M_inductively_coupled_impedance(Freq=frequency, Z=Z_t(frequency), L1=L_line, L2=L_hanger, M=M_inductance)
        # if element already exists
        if element_idx is None:
            self._add_element(M_t)
        else:
            self._substitute_element(M_t, element_idx)

    def find_resonance_parameters(self, frequency_bounds: tuple):
        '''
        Function used to detect resonator frequencies and kappas.
        This is done by searching for poles and zeros in complex
        frequency space.
        '''
        # initial guess is estimated with linear frequency sweep
        freq_axis = np.linspace(*frequency_bounds, int(1e3))
        s21 = np.abs(self.get_S_parameters(frequency=freq_axis)[2])
        f0 = freq_axis[np.argmin(s21)]
        # get tighter frequency bounds
        dist = np.min(np.abs(np.array(frequency_bounds)-f0))
        frequency_bounds = (f0-dist, f0+dist)
        # Find zero of S21
        def cost_func(freq):
            s21 = self.get_S_parameters(frequency=np.array([*freq]))[2]
            return np.log(np.abs(s21))[0]
        # run minimizer
        initial_guess = [f0]
        bounds = [frequency_bounds]
        f0 = minimize(cost_func, x0=initial_guess, options={'disp':False}, bounds=bounds, method='Powell').x[0]
        # Find pole of S21
        def cost_func(complex_freq):
            freq_re, freq_im = complex_freq
            s21 = self.get_S_parameters(frequency=np.array([ freq_re + freq_im*1j ]))[2]
            return np.log(1/np.abs(s21))
        # run minimizer
        initial_guess = [f0, 1e6]
        bounds = [frequency_bounds, (-50e6, +50e6)]
        _, k0 = minimize(cost_func, x0=initial_guess, options={'disp':False}, bounds=bounds, method='Powell').x
        return f0, k0










