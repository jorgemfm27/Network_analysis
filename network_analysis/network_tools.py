import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from . import circuit_design as cd
import qutip


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
    An ABCD matrix modeling transmission through a parallel incuctively coupled impedance Z
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

# def M_attenuator(Freq: np.array, attn_dB: float, Z0:float):
#     '''
#     An ABCD matrix for a standard T attenuator circuit:

#        R1       R1
#     --WWWW--|--WWWW--
#             Z
#             Z R2
#             |
#     ----------------
#     '''
#     assert len(Freq.shape), 'Frequency should be given as 1D array'
#     w = 2*np.pi*Freq
#     # attenuation linear
#     attn = 10**(attn_dB/20) #??????
#     attn = 10**(attn_dB/10)
#     R1 = Z0*((attn-1)/(attn+1))
#     R2 = Z0*((2*attn)/(attn**2-1))
#     # ABCD matrix coeficients
#     A = 1 + R1/R2
#     B = R1 * ( 2 + R1/R2)
#     C = 1/R2
#     D = 1 + R1/R2
#     M = np.zeros((len(Freq), 2, 2), dtype=np.complex_)
#     for i in range(len(Freq)):
#         M[i] = np.array([[A, B],
#                          [C, D]])
#     return M

def M_attenuator(Freq: np.array, attn_dB: float, Z0:float):
    '''
    An ABCD matrix for a standard Pi attenuator circuit:
           R2       
    ---|--WWWW--|---
       Z        Z   
    R1 Z        Z R1
       |        |   
    ----------------
    '''
    assert len(Freq.shape), 'Frequency should be given as 1D array'
    assert attn_dB <=40, 'Attenuation must be at most 40 dB'
    w = 2*np.pi*Freq
    # attenuation linear
    attn = 10**(attn_dB/20)
    R1 = Z0*((attn+1)/(attn-1))
    R2 = Z0/2*(attn-1/attn)
    # ABCD matrix coeficients
    A = 1 + R2/R1
    B = R2
    C = 2/R1 + R2/(R1**2)
    D = 1 + R2/R1
    M = np.zeros((len(Freq), 2, 2), dtype=np.complex_)
    for i in range(len(Freq)):
        M[i] = np.array([[A, B],
                         [C, D]])
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

def PSD_thermal(f, T):
    '''
    Power spectral density at temperature T in units J/Hz.
    Args:
        T  : temperature in Kelvin.
        Z0 : Impedance of system.
    '''
    h = 6.62607015e-34
    kB = 1.380649e-23
    eta = (h*f/(kB*T))/(np.exp(h*f/(kB*T))-1) # Bose-Einstein
    PSD = 4*kB*T*eta
    return PSD

def Vp_from_PdBm(P_dBm, R=50):
    '''
    Convert power in dBm to voltage peak.
    '''
    return np.sqrt(R*10**(P_dBm/10-3))

class Network():
    '''
    Class used to construct and model a 2 port network.
    Network can be built using add_<component> methods.
    The self.draw_network() method plots a schematic of the network bult.
    '''
    def __init__(self, Zgen: float):
        # Elements along network
        self.elements = []
        self.element_properties = {}
        # Loading of ports when extracting scattering and impedance parameters
        self.Zgen = Zgen

    def _add_element(self, M_function, name, properties=None):
        '''
        Add an ABCD matrix element to network.
        M_function should have a frequency argument.
        '''
        self.elements.append((name, M_function))
        if properties:
            element_idx = len(self.elements)-1
            self.element_properties[element_idx] = properties

    def _substitute_element(self, M_function, name, element_idx, properties=None):
        '''
        Substitute an ABCD matrix element in existing network.
        M_function should have a frequency argument.
        '''
        self.elements[element_idx] = (name, M_function)
        if properties:
            self.element_properties[element_idx] = properties

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
        for name, element in self.elements:
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
        for name, element in self.elements:
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

    def get_node_VI(self, node_idx: int, in_freq: float, in_amp: float = 1, in_phase: float = 0):
        '''
        Compute the voltage and current at a given node of the network for a given input field.
        Args:
            node_idx : node of circuit.
            in_freq  : input signal frequency.
            in_amp   : input signal amplitude (V).
            in_phase : input signal phase (deg).
        '''
        assert node_idx < len(self.elements)+1, f'Network contains only {len(self.elements)+1} nodes.'
        # convert frequency into array
        if isinstance(in_freq, float):
            in_freq = np.array([in_freq])
        # compute voltage and current at input node 0
        v0 = in_amp*np.exp(1j*in_phase*np.pi/180)*np.ones(in_freq.shape)
        # input current obtained from input impedance
        z11, z12, z21, z22, zin, zout = self.get_Z_parameters(frequency=in_freq)
        i0 = v0/(zin)
        A_0 = np.array([v0, i0]).T
        # compute voltage and current at node_idx
        # list of ABCD matrices of elements in network
        M_elements = [np.array([np.eye(2, dtype=complex) for _ in in_freq])]
        for name, element in self.elements[:node_idx]:
            m_element = element(frequency=in_freq)
            M_elements.append(m_element)
        # multiply all ABCD matrices
        M_system = multiply_matrices([*M_elements])
        M_system = np.array([np.linalg.inv(m) for m in M_system])
        # get node voltage and current
        A_node = np.einsum('nij,nj->ni', M_system, A_0)
        vnode, inode = A_node[:,0], A_node[:,1]
        return vnode, inode

    def get_thermal_psd_at_node(self, node_idx, frequency, plot=False):
        '''
        Calculate the cascaded psd from all finite temperature elements in the network
        '''
        # Start with a 0 PSD
        PSD = np.zeros(frequency.shape)
        if plot:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('frequency (Hz)')
            ax.set_ylabel('PSD ($\\mathrm{{\\:J\\:Hz^{-1}}}$)')
            ax.set_title(f'Cumulative PSD at node {node_idx}')
        # Loop through all nodes up to the selected node
        for idx, (element, _) in enumerate(self.elements[:node_idx]):
            _node_idx = idx+1
            if element == 'attenuator':
                temperature_K = self.element_properties[idx]['temperature_K']
                attn_dB = self.element_properties[idx]['attn_dB']
                attn = 10**(-attn_dB/10) # here we should compute attenuation in power
                # cascaded attenuation
                PSD = (1-attn)*PSD_thermal(frequency, temperature_K) + attn*PSD
                # plot current PSD
                if plot:
                    ax.plot(frequency, PSD, label=f'node {_node_idx} ({temperature_K} K)', color=f'C{idx}')
                    # ax.plot(frequency, (1-attn)*PSD_thermal(frequency, temperature_K),
                    #         color=f'C{idx}', ls='--', autoscale=False)
        if plot:
            ax.legend(frameon=False, loc=2, bbox_to_anchor=(1, 1))
        return PSD

    def _generate_thermal_noise_ensemble(self, frequency: float, PSD: float, shots: int = 10000):
        '''
        Get a thermal ensemble of voltage shots.
        Args:
            frequency : frequency over which PSD was sampled (Hz)
            PSD       : Power spectral density (J/Hz)
        '''
        if isinstance(shots, float):
            shots = int(shots)
        bandwidth = frequency.max() - frequency.min()
        PSD_V2 = PSD*self.Zgen
        # ensemble of thermal shots
        v_noise = np.zeros(shots, dtype=np.complex128)
        for i in range(shots):
            phase = np.random.random(101)*2*np.pi
            v_rms = np.sqrt(PSD_V2*bandwidth)
            v_noise[i] = np.sum(v_rms/np.sqrt(2)*np.exp(1j*phase))
        return v_noise

    def get_node_thermal_VI(self, node_idx: int, in_freq: float, in_amp: float = 1, in_phase: float = 0, shots = 10000):
        '''
        Compute the voltage and current at a given node of the network for an input coherent state
        and compute Wigner function in voltage.
        Args:
            node_idx : node of circuit.
            in_freq  : input signal frequency.
            in_amp   : input signal amplitude (V).
            in_phase : input signal phase (deg).
        '''
        v_node, i_node = self.get_node_VI(node_idx=node_idx, in_freq=in_freq, in_amp=in_amp, in_phase=in_phase)
        v_node, i_node = v_node[0], i_node[0]
        z_node = v_node/i_node
        assert (type(v_node) is complex) or (type(v_node) is np.complex128)
        # get thermal voltages ensemble
        bandwidth = 2.4e9
        frequency = np.linspace(-bandwidth/2, bandwidth/2, 101)+in_freq
        PSD = self.get_thermal_psd_at_node(node_idx=node_idx, frequency=frequency)
        v_noise = self._generate_thermal_noise_ensemble(frequency=frequency, PSD=PSD, shots=shots)
        v_thermal = v_noise + v_node
        # glot thermal shot distribution
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        heatmap, xedges, yedges = np.histogram2d(np.real(v_thermal)*1e6, np.imag(v_thermal)*1e6, bins=21,)
        ax.pcolormesh(xedges, yedges, heatmap, cmap='Blues')
        axlim = np.abs(np.concatenate((xedges, yedges))).max()
        ax.set_xlim(-axlim, axlim)
        ax.set_ylim(-axlim, axlim)
        ax.grid(ls='--', lw=1, color='gray', alpha=.25)
        ax.set_ylabel('Voltage Q ($\\mathrm{{\\mu}}$V)')
        ax.set_xlabel('Voltage I ($\\mathrm{{\\mu}}$V)')
        ax.set_title(f'Thermal voltage at node {node_idx}')

    def get_node_quantum_VI(self, node_idx: int, in_freq: float, in_amp: float = 1, in_phase: float = 0):
        '''
        Compute the voltage and current at a given node of the network for an input coherent state
        and compute Wigner function in voltage.
        Args:
            node_idx : node of circuit.
            in_freq  : input signal frequency.
            in_amp   : input signal amplitude (V).
            in_phase : input signal phase (deg).
        '''
        v_node, i_node = self.get_node_VI(node_idx=node_idx, in_freq=in_freq, in_amp=in_amp, in_phase=in_phase)
        v_node, i_node = v_node[0], i_node[0]
        # z_node = v_node/i_node
        z_node = self.Zgen
        assert (type(v_node) is complex) or (type(v_node) is np.complex128)
        # calculate vacuum state fluctuations of voltage
        h = 6.62607015e-34
        e = 1.602176634e-19
        Phi_0 = h/(2*e)  # flux quantum
        R_Q = h/(4*e**2) # resistance quantum
        V_zpf = (2*np.pi*in_freq)*Phi_0*np.sqrt(z_node/(R_Q*4*np.pi))
        # V_zpf = np.sqrt( h*in_freq/(2*self.Zgen) )
        # V_zpf = np.abs(V_zpf)
        # calculate equivalent coherent state
        alpha = (v_node/V_zpf)/2
        N = 200
        # if type(v_node) is np.ndarray:
        #     state = sum([qutip.coherent_dm(N, a) for a in alpha])/shots # Thermal mixture of coherent states
        # else:
        state = qutip.coherent_dm(N, alpha) # single coherent state
        x_vec = np.linspace(-10, 10, 61)
        W_func = qutip.wigner(state, x_vec, x_vec)
        # Plot wigner function
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        x_volt = 2*V_zpf*x_vec
        ax.pcolormesh(x_vec, x_vec, W_func, cmap='Blues', shading='nearest')
        ax.grid(ls='--', lw=1, color='gray', alpha=.25)
        ax.set_ylabel('$Y$ Quadrature')
        ax.set_xlabel('$X$ Quadrature')
        ax.set_title(f'Quantum voltage at node {node_idx}')
        ax.plot(np.real(alpha), np.imag(alpha), 'C3x' )
        # Vzpf axes
        ax_vy = ax.twinx()
        ax_vx = ax.twiny()
        ax_vy.set_ylabel('Voltage Q ($\\mathrm{{\\mu}}$V)')
        ax_vy.set_ylim([l*2*V_zpf*1e6 for l in ax.get_ylim()])
        ax_vx.set_xlabel('Voltage I ($\\mathrm{{\\mu}}$V)')
        ax_vx.set_xlim([l*2*V_zpf*1e6 for l in ax.get_xlim()])
        # ax_vx.axvline(V_zpf*1e6, color='C0', ls='--')
        # # ax_vy.axhline(np.imag(v_node*1e6), color='C3', ls='--')
        # ax_vx.axvline(np.real(v_node*1e6), color='C3', ls='--')
        # ax_vy.axhline(np.imag(v_node*1e6), color='C3', ls='--')

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
            self._add_element(M_C, name=f'capacitor_{element_type}')
        else:
            self._substitute_element(M_C, element_idx, name=f'capacitor_{element_type}')

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
            self._add_element(M_L, name=f'inductor_{element_type}')
        else:
            self._substitute_element(M_L, element_idx, name=f'inductor_{element_type}')

    def add_transmission_line(self, length: float, Z0: float, phase_velocity: float, element_idx=None):
        '''
        Add a series transmission line element to network.
        '''
        M_t = lambda frequency: M_series_transline(Freq=frequency, l=length, v=phase_velocity, Z0=Z0)
        # if element already exists
        if element_idx is None:
            self._add_element(M_t, name='transmission_line')
        else:
            self._substitute_element(M_t, element_idx, name='transmission_line')

    def add_attenuator(self, attn_dB: float, Z0: float, temperature_K: float, element_idx=None):
        '''
        Add a series attenuator element to network.
        '''
        M_a = lambda frequency: M_attenuator(Freq=frequency, attn_dB=attn_dB, Z0=Z0)
        # if element already exists
        if element_idx is None:
            self._add_element(M_a, name='attenuator', properties={'attn_dB': attn_dB, 'temperature_K': temperature_K})
        else:
            self._substitute_element(M_a, element_idx, name=f'attenuator_{attn_dB:.0f}')

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
            self._add_element(M_t, name='cap_resonator')
        else:
            self._substitute_element(M_t, element_idx, name='cap_resonator')

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
            self._add_element(M_t, name='ind_resonator')
        else:
            self._substitute_element(M_t, element_idx, name='ind_resonator')

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

    def draw_network(self):
        '''
        Draw schematic of network.
        '''
        fig, ax = plt.subplots(figsize=(1,1), dpi=100)
        ax.set_xlim(0, .9)
        ax.set_ylim(-.45, .45)
        ax.axis('off')
        for i, (name, element) in enumerate(self.elements):
            x_i = 1.3*i
            x_j = 1.3*(i+1)
            xij = np.mean([x_i, x_j])
            # plot capacitor in series
            if name == 'capacitor_series':
                cd.plot_capacitor(x_i, 0, x_j, 0, l_cap=.2, cap_dist=.2)
            # plot capacitor in parallel
            elif name == 'capacitor_parallel':
                ax.plot([x_i, x_j], [0, 0], color='k', solid_capstyle='round', linewidth=4, clip_on=False)
                cd.plot_capacitor(xij, 0, xij, -1.2, l_cap=.2, cap_dist=.2)
                cd.plot_ground(xij, -1., .2, horizontal=False)
            # plot inductor in series
            elif name == 'inductor_series':
                cd.plot_inductor(x_i, 0, x_j, 0, w_ind=.2, lpad=.25)
            # plot inductor in parallel
            elif name == 'inductor_parallel':
                ax.plot([x_i, x_j], [0, 0], color='k', solid_capstyle='round', linewidth=4, clip_on=False)
                cd.plot_inductor(xij, 0, xij, -1.2, w_ind=.2, lpad=.25)
                cd.plot_ground(xij, -1., .2, horizontal=False)
            # plot transmissionline
            elif name == 'transmission_line':
                cd.plot_transmission_line(x_i+.05, 0, 1.2, horizontal=True, radius=.2)
            # plot capacitively coupled resonator
            elif name == 'cap_resonator':
                ax.plot([x_i, x_j], [0, 0], color='k', solid_capstyle='round', linewidth=4, clip_on=False)
                cd.plot_capacitor(xij, 0, xij, -.35, l_cap=.1, cap_dist=.12)
                cd.plot_ground(xij, -1.1, .1, horizontal=False)
                cd.plot_transmission_line(xij, -.25, .9, horizontal=False, radius=.2)
            # plot inductively coupled resonator
            elif name == 'ind_resonator':
                cd.plot_inductor(x_j, 0, x_i, 0, w_ind=.08, lpad=.4)
                cd.plot_inductor(x_i+.2, -.3, x_j-.2, -.3, w_ind=.08, lpad=.2)
                cd.plot_ground(x_j-.2, -.3, .2, horizontal=False)
                cd.plot_transmission_line(x_i+.2, -.3, .9, horizontal=False, radius=.2)
            # plot inductively coupled resonator
            elif 'attenuator' in name:
                attn_dB = self.element_properties[i]['attn_dB']
                temperature_K = self.element_properties[i]['temperature_K']
                cd.plot_attenuator(xij, 0, h=.3, l=.6, attn_dB=attn_dB)
                ax.plot([x_i, x_j], [0, 0], color='k', solid_capstyle='round', linewidth=4, clip_on=False)
                ax.text(xij, -.5, f'{temperature_K} K', va='center', ha='center')
            else:
                ax.plot([x_i, x_j], [0, 0], color='k', solid_capstyle='round', linewidth=4, clip_on=False)
                ax.text(xij, 0, '?', ha='center', va='center')
            # plot node
            ax.plot([x_i], [0], color='k', marker='o', markersize=10, clip_on=False)
            ax.text(x_i, .3, f'{i}', size=12, va='center', ha='center')
        # final node
        ax.plot([x_j], [0], color='k', marker='o', markersize=10, clip_on=False)
        ax.text(x_j, .3, f'{i+1}', size=12, va='center', ha='center')
        plt.show()











