from . import circuit_design as cd
from .network_helpers import *
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import constants

# Main class start
class Network():
    '''
    Class used to construct and model a 2 port network. 
    A short summary of the main methods of the class:

    add_<element>:
        Components can be added to the network using the "add_<element>" methods. 
        These range from simple lumped elements such as capacitors, inductors,
        resistors, etc, to distributed elements (transmission lines) and composite
        elements (attenuators, amplifiers, resonators, transformers, etc.)
        Custom commercial elements (imported as s2p files) can also be added using
        "add_custom_component".

    sweep_element_parameter:
        Method used to sweep a certain parameter of a given element of the network.
        Usefull to perform sweeps.

    get_S_parameters:
        Get the scattering parameters of the resulting network for a given frequency
        axis.

    get_Z_parameters:
        Get the impedance parameters of the resulting network for a given frequency
        axis.
    
    get_S_versus_parameters:
        Get the scattering parameters of the resulting networkfor a given frequency
        axis and a given element parameter axis.

    get_node_VI:
        Get the complex voltage and current phasors of the network a given node
        for an input drive. 

    get_signal_response:
        Get the response of the network to a given input time-domain signal. Usefull 
        to assess transient effects and pulse distortion. Returns the scattered
        output signal in all directions (s11, s12, s21, s22).

    find_resonance_parameters:
        Find resonances in the network and return their respective frequency and 
        linewidth. Performed by looking for poles and zeros in the scattering 
        parameters. (WARNING: still a bit finicky).

    get_psd_at_node:
        Get the power spectral density describing noise at each node of the network,
        for an input node noise temperature. Some resistive elements (such as 
        attenuators, amplifiers, and custom elements) also support noise temperatures
        Usefull to estimate noise filtering and SNR of amplifier chains across
        the network. Also related to the methods:
            get_svv_at_output: get the voltage spectral density.
            get_sii_at_output: get the current spectral density.
            get_spp_at_output: get the flux spectral density.

    draw_network:
        Plot a schematic of the network built.
    '''
    def __init__(
        self,
        Zgen: float,
    ):
        # Elements along network
        self.elements = []
        self.element_properties = {}
        # Loading of ports when extracting scattering and impedance parameters
        self.Zgen = Zgen
        # Figure storing network plot
        self.fig = None

    def _add_element(
        self, 
        M_function: callable, 
        name: str, 
        properties: dict = None,
    ):
        '''
        Add an ABCD matrix element to network.
        M_function should have a frequency argument.
        '''
        self.elements.append((name, M_function))
        if properties:
            element_idx = len(self.elements)-1
            self.element_properties[element_idx] = properties

    def _substitute_element(
        self, 
        M_function: callable, 
        element_idx: int, 
        name: str, 
        properties: dict = None,
    ):
        '''
        Substitute an ABCD matrix element in existing network.
        M_function should have a frequency argument.
        '''
        self.elements[element_idx] = (name, M_function)
        if properties:
            self.element_properties[element_idx] = properties

    #################################################
    # Elements types of network
    #################################################
    def add_capacitance(
        self, 
        C: float, 
        element_type: str, 
        element_idx=None,
    ):
        '''
        Add a capacitor to the network.
        Args:
            C            : capacitance (Farads)
            element_type : arrangement of circuit element ("series" or "parallel")
            element_idx  : position index in network (used only to replace existing elements)
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
            self._add_element(M_C, name=f'capacitor_{element_type}', properties={'capacitance': C})
        else:
            self._substitute_element(M_C, element_idx, name=f'capacitor_{element_type}', properties={'capacitance': C})

    def add_inductance(
        self, 
        L: float, 
        element_type: str, 
        element_idx=None,
    ):
        '''
        Add an inductor to the network.
        Args:
            L            : Inductance (Henry)
            element_type : arrangement of circuit element ("series" or "parallel")
            element_idx  : position index in network (used only to replace existing elements)
        '''
        # Assertion
        assert element_type in ['series', 'parallel'], 'type must be "series" or "parallel"'
        # add a series inductance element to network
        if element_type == 'series':
            Z = lambda frequency: Z_inductor(Freq=frequency, L=L)
            M_L = lambda frequency: M_series_impedance(Z=Z(frequency))
        else:
            Z = lambda frequency: Z_inductor(Freq=frequency, L=L)
            M_L = lambda frequency: M_parallel_impedance(Z=Z(frequency))
        # if element already exists
        if element_idx is None:
            self._add_element(M_L, name=f'inductor_{element_type}', properties={'inductance': L})
        else:
            self._substitute_element(M_L, element_idx, name=f'inductor_{element_type}', properties={'inductance': L})

    def add_resistor(
        self, 
        R: float,
        element_type: str, 
        element_idx=None,
    ):
        '''
        Add a resistor to the network.
        Args:
            R            : Resistance (Ohms)
            element_type : arrangement of circuit element ("series" or "parallel")
            element_idx  : position index in network (used only to replace existing elements)
        '''
        # Assertion
        assert element_type in ['series', 'parallel'], 'type must be "series" or "parallel"'
        # add a series capacitance element to network
        if element_type == 'series':
            Z = lambda frequency: Z_resistor(Freq=frequency, R=R)
            M_C = lambda frequency: M_series_impedance(Z=Z(frequency))
        else:
            Z = lambda frequency: Z_resistor(Freq=frequency, R=R)
            M_C = lambda frequency: M_parallel_impedance(Z=Z(frequency))
        # if element already exists
        if element_idx is None:
            self._add_element(M_C, name=f'resistor_{element_type}', properties={'resistance': R})
        else:
            self._substitute_element(M_C, element_idx, name=f'resistor_{element_type}', properties={'resistance': R})

    def add_transmission_line(
        self, 
        length: float, 
        Z0: float, 
        phase_velocity: float, 
        element_idx=None,
    ):
        '''
        Add a series transmission line element to network.
        Args:
            length         : length of transmission line (meters)
            Z0             : characteristic impedance of transmission line (Ohms)
            phase_velocity : speed of light in transmission line (m/s)
            element_idx    : position index in network (used only to replace existing elements)
        '''
        M_t = lambda frequency: M_series_transline(Freq=frequency, l=length, v=phase_velocity, Z0=Z0)
        # if element already exists
        if element_idx is None:
            self._add_element(M_t, name='transmission_line', properties={'length': length, 'Z0': Z0, 'phase_velocity': phase_velocity})
        else:
            self._substitute_element(M_t, element_idx, name='transmission_line', properties={'length': length, 'Z0': Z0, 'phase_velocity': phase_velocity})

    def add_parallel_transmission_line(
        self, 
        length: float, 
        Z0: float, 
        phase_velocity: float, 
        Z_load: float, 
        element_idx=None,
    ):
        '''
        Add a parallel transmission line element to network.
        Args:
            length         : length of transmission line (meters)
            Z0             : characteristic impedance of transmission line (Ohms)
            phase_velocity : speed of light in transmission line (m/s)
            Z_load         : load impedance of at the end of the transmission line (Ohm)
            element_idx    : position index in network (used only to replace existing elements)
        '''
        M_t = lambda frequency: M_parallel_impedance(Z_term_transline(frequency, l=length, v=phase_velocity, Z0=Z0, ZL=Z_load))
        # if element already exists
        if element_idx is None:
            self._add_element(M_t, name='parallel_transmission_line', properties={'Z_load': Z_load, 'length': length, 'Z0': Z0, 'phase_velocity': phase_velocity})
        else:
            self._substitute_element(M_t, element_idx, name='parallel_transmission_line', properties={'Z_load': Z_load, 'length': length, 'Z0': Z0, 'phase_velocity': phase_velocity})

    def add_transformer(
        self, 
        L1: float, 
        L2: float, 
        M: float, 
        element_idx=None,
    ):
        '''
        Add a transformer element to the network:
        Args:
            L1          : Inductance on left side (Henry)
            L2          : Inductance on right side (Henry)
            M           : Mutual inductance between inductors (Henry)
            element_idx : position index in network (used only to replace existing elements)
        '''
        # ABCD matrix of transformer
        M_t = lambda frequency: M_transformer(Freq=frequency, L1=L1, L2=L2, M=M)
        # if element already exists
        if element_idx is None:
            self._add_element(M_t, name='transformer', properties={'L1': L1, 'L2': L2, 'M': M})
        else:
            self._substitute_element(M_t, element_idx, name='transformer', properties={'L1': L1, 'L2': L2, 'M': M})

    def add_capacitively_coupled_hanger(
        self, 
        length: float, 
        Z0: float, 
        phase_velocity: float, 
        Z_termination: float, 
        C_coupling: float, 
        element_idx=None,
    ):
        '''
        Add a parallel terminated transmission line in series with a capacitor element to network.
        Args:
            length         : length of transmission line (meters)
            Z0             : characteristic impedance of transmission line (Ohms)
            phase_velocity : speed of light in transmission line (m/s)
            Z_termination  : load impedance of at the end of the transmission line (Ohm)
            C_coupling     : capacitance (C)
            element_idx    : position index in network (used only to replace existing elements)
        '''
        # impedance of capacitor
        Z_C = lambda frequency: Z_capacitor(Freq=frequency, C=C_coupling)
        # impedance of terminated transmission line
        Z_t = lambda frequency: Z_term_transline(Freq=frequency, l=length, v=phase_velocity, Z0=Z0, ZL=Z_termination)
        # ABCD matrix of parallel elements
        M_t = lambda frequency: M_parallel_impedance(Z=Z_C(frequency)+Z_t(frequency))
        # if element already exists
        if element_idx is None:
            self._add_element(M_t, name='capacitively_coupled_hanger', properties={
                'length': length, 
                'Z0': Z0,
                'phase_velocity': phase_velocity,
                'C_coupling': C_coupling, 
                'Z_termination': Z_termination})
        else:
            self._substitute_element(M_t, element_idx, name='capacitively_coupled_hanger', properties={
                'length': length, 
                'Z0': Z0,
                'phase_velocity': phase_velocity,
                'C_coupling': C_coupling, 
                'Z_termination': Z_termination})

    def add_inductively_coupled_hanger(
        self, 
        length: float, 
        Z0: float, 
        phase_velocity: float, 
        Z_termination: float, 
        L_line: float, 
        L_hanger: float, 
        M_inductance: float, 
        element_idx=None,
    ):
        '''
        Add a parallel terminated transmission line inductively coupled to main line of network.
        Args:
            length         : length of transmission line (meters)
            Z0             : characteristic impedance of transmission line (Ohms)
            phase_velocity : speed of light in transmission line (m/s)
            Z_termination  : load impedance of at the end of the transmission line (Ohm)
            L_line         : inductance on main line (H)
            L_hanger       : inductance on parallel line (H)
            M_inductance   : mutual inductance between parallel and series inductance (H).
            element_idx    : position index in network (used only to replace existing elements)
        '''
        # impedance of terminated transmission line
        Z_t = lambda frequency: Z_term_transline(Freq=frequency, l=length, v=phase_velocity, Z0=Z0, ZL=Z_termination)
        # ABCD matrix of parallel elements
        M_t = lambda frequency: M_inductively_coupled_impedance(Freq=frequency, Z=Z_t(frequency), L1=L_line, L2=L_hanger, M=M_inductance)
        # if element already exists
        if element_idx is None:
            self._add_element(M_t, name='inductively_coupled_hanger', properties={
                'length': length, 
                'Z0': Z0,
                'phase_velocity': phase_velocity,
                'L_line': L_line,
                'L_hanger': L_hanger,
                'M_inductance': M_inductance,
                'Z_termination': Z_termination})
        else:
            self._substitute_element(M_t, element_idx, name='inductively_coupled_hanger', properties={
                'length': length, 
                'Z0': Z0,
                'phase_velocity': phase_velocity,
                'L_line': L_line,
                'L_hanger': L_hanger,
                'M_inductance': M_inductance,
                'Z_termination': Z_termination})

    def add_attenuator(
        self, 
        attn_dB: float, 
        Z0: float, 
        temperature_K: float, 
        element_idx=None,
    ):
        '''
        Add an attenuator to the network.
        (This component is implemented as a pi-network of resistors).
        Args:
            attn_dB       : power attenuation of attenuator (dB)
            Z0            : characteristic impedance of attenuator (Ohms)
            temperature_K : temperature of component (Kelvin)
            element_idx   : position index in network (used only to replace existing elements)
        '''
        M_a = lambda frequency: M_attenuator(Freq=frequency, attn_dB=attn_dB, Z0=Z0)
        # if element already exists
        if element_idx is None:
            self._add_element(M_a, name='attenuator', properties={'attn_dB': attn_dB, 'temperature_K': temperature_K})
        else:
            self._substitute_element(M_a, element_idx, name=f'attenuator_{attn_dB:.0f}', properties={'attn_dB': attn_dB, 'temperature_K': temperature_K})

    def add_amplifier(
        self, 
        gain_dB: float, 
        temperature_K: float, 
        Z0: float, 
        element_idx=None,
    ):
        '''
        Add an amplifier to the network.
        (This component is implemented as a pi-network of resistors).
        Args:
            gain_dB       : power gain of amplifier (dB)
            Z0            : characteristic impedance of attenuator (Ohms)
            temperature_K : temperature of component (Kelvin)
            element_idx   : position index in network (used only to replace existing elements)
        '''
        M_a = lambda frequency: M_amplifier(Freq=frequency, gain_dB=gain_dB, Z0=Z0)
        # if element already exists
        if element_idx is None:
            self._add_element(M_a, name='amplifier', properties={'gain_dB': gain_dB, 'temperature_K': temperature_K})
        else:
            self._substitute_element(M_a, element_idx, name=f'amplifier_{gain_dB:.0f}')

    def add_SQUID(
        self, 
        Ej1: float, 
        Ej2: float, 
        phi: float, 
        element_type: str, 
        element_idx=None,
    ):
        '''
        Add a SQUID element to the network.
        (This component is implemented as a variable inductance).
        Args:
            Ej1/Ej2       : Josephson energies (Hz)
            phi           : external flux threading squid (flux quantum phi_0)
            element_idx   : position index in network (used only to replace existing elements)
        '''
        # Assertion
        assert element_type in ['series', 'parallel'], 'type must be "series" or "parallel"'
        # Compute Josephson inductance
        Ej_eff = effective_josephson_energy(Ej1, Ej2, phi)
        L_eff = josephson_inductance(Ej_eff)
        # add a series inductance element to network
        if element_type == 'series':
            Z = lambda frequency: Z_inductor(Freq=frequency, L=L_eff)
            M_J = lambda frequency: M_series_impedance(Z=Z(frequency))
        else:
            Z = lambda frequency: Z_inductor(Freq=frequency, L=L_eff)
            M_J = lambda frequency: M_parallel_impedance(Z=Z(frequency))
        # if element already exists
        if element_idx is None:
            self._add_element(M_J, name=f'SQUID_{element_type}', properties={'Ej1': Ej1, 'Ej2': Ej2, 'phi': phi})
        else:
            self._substitute_element(M_J, element_idx, name=f'SQUID_{element_type}', properties={'Ej1': Ej1, 'Ej2': Ej2, 'phi': phi})

    def add_custom_component(
        self, 
        name: str, 
        temperature_K: float, 
        element_idx=None, 
        plot_params: bool = False,
    ):
        '''
        Add a component from S matrix data available in .s2p file.
        See .\circuit_components\ for avaliable components.
        Args:
            name          : name of component (must match name of file found in the directory)
            temperature_K : temperature of component (Kelvin)
            element_idx   : position index in network (used only to replace existing elements)
            plot_params   : plot parameters used to model component.
        '''
        # Parse s2p file data to scattering paramaters
        component_data_file = os.path.abspath(os.path.join(__file__, '..', 'circuit_components', f'{name}.s2p'))
        data = parse_s2p(component_data_file)
        # Plot data from s2p file
        if data['param_type'].lower() == 's':
            freq = np.array(data['frequency'])
            S11 = np.array(data['11'])
            S12 = np.array(data['21']) # we make s21 = s12 to avoid numerical instabilities
            S21 = np.array(data['21']) # (this is only a good assumption for reciprocal components)
            S22 = np.array(data['22'])
            Z0 = float(data['char_imp'])
            # convert s params to ABCD params
            _A =        ((1 + S11)*(1 - S22) + S12*S21 )/(2*S21)
            _B =   Z0 * ((1 + S11)*(1 + S22) - S12*S21 )/(2*S21)
            _C = 1/Z0 * ((1 - S11)*(1 - S22) - S12*S21 )/(2*S21)
            _D =        ((1 - S11)*(1 + S22) + S12*S21 )/(2*S21)
            if plot_params:
                fig, axs = plt.subplots(figsize=(8, 5),ncols=2, nrows=2, sharex='col')
                # plot S params mag
                axs = axs.flatten()
                axs[0].plot(freq, 20*np.log10(np.abs(S11)), color='C0', alpha=1, ls='-', zorder=+1)
                axs[1].plot(freq, 20*np.log10(np.abs(S12)), color='C0', alpha=1, ls='-', zorder=+1)
                axs[2].plot(freq, 20*np.log10(np.abs(S21)), color='C0', alpha=1, ls='-', zorder=+1)
                axs[3].plot(freq, 20*np.log10(np.abs(S22)), color='C0', alpha=1, ls='-', zorder=+1)
                # # plot S params phase
                # axt = [ax.twinx() for ax in axs]
                # axt[0].plot(freq, np.angle(S11), color='C1', alpha=.5, ls='--', zorder=-1)
                # axt[1].plot(freq, np.angle(S12), color='C1', alpha=.5, ls='--', zorder=-1)
                # axt[2].plot(freq, np.angle(S21), color='C1', alpha=.5, ls='--', zorder=-1)
                # axt[3].plot(freq, np.angle(S22), color='C1', alpha=.5, ls='--', zorder=-1)
                set_xlabel(axs[2], 'frequency', unit='Hz')
                set_xlabel(axs[3], 'frequency', unit='Hz')
                set_ylabel(axs[0], '|S_{11}|', unit='dB')
                set_ylabel(axs[1], '|S_{12}|', unit='dB')
                set_ylabel(axs[2], '|S_{21}|', unit='dB')
                set_ylabel(axs[3], '|S_{22}|', unit='dB')
        # Assemble M matrix function
        def M_CC(frequency):
            # interpolate for different frequency values
            from scipy.interpolate import interp1d
            A = interp1d(freq, _A, kind='linear')
            B = interp1d(freq, _B, kind='linear')
            C = interp1d(freq, _C, kind='linear')
            D = interp1d(freq, _D, kind='linear')
            # assemble matrix
            if True:
                # deal with negative frequencies
                _frequency = np.abs(frequency)
                __A = A(_frequency)
                __B = B(_frequency)
                __C = C(_frequency)
                __D = D(_frequency)
                # conjugate terms of negative frequency
                idx = np.where(frequency<0)
                __A[idx] = np.conj(__A[idx])
                __B[idx] = np.conj(__B[idx])
                __C[idx] = np.conj(__C[idx])
                __D[idx] = np.conj(__D[idx])
                return np.moveaxis(np.array([[__A, __B],
                                             [__C, __D]]), -1, 0)
            else:
                raise ValueError('frequency out of component range!')
        # if element already exists
        element_type = next((line for line in data["metadata"] if "type" in line.lower()), 'unknown commercial component')
        model = next((line for line in data["metadata"] if "model" in line.lower()), 'unknown commercial component').split(' ')[1]
        if element_idx is None:
            self._add_element(M_CC, name='commercial_component', properties={'type': element_type, 
                                                                             'model': model, 
                                                                             'temperature_K': temperature_K})
        else:
            self._substitute_element(M_CC, element_idx, name='commercial_component', properties={'type': element_type, 
                                                                                                 'model': model,
                                                                                                 'temperature_K': temperature_K})

    #################################################
    # Methods for properties/behaviors of network
    #################################################
    def get_S_parameters(
        self, 
        frequency: list, 
        plot: bool | str = False, 
        **kw,
    ):
        '''
        Compute the scattering parameters of the network over a frequency range.
        Args:
            frequency  : frequency array over which to evaluate scattering params.
            plot       : plot scattering parameters. Can be given as bool or one of
                         these strings ('s11', 's12', 's21', 's22').
            xscale     : axis scale of plot ('linear', 'log').
            yscale     : axis scale of plot ('linear', 'log', 'dB').
            plot_phase : plot magnitude and phase of each s parameter.
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
        # plot parameters
        if plot:
            yscale = kw.get('yscale', 'linear')
            xscale = kw.get('xscale', 'linear')
            plot_phase = kw.get('plot_phase', False)
            # plot single s parameter
            if isinstance(plot, str):
                assert plot.lower() in ['s11', 's12', 's21', 's22']
                s_param_dict = {'s11': S11, 's12': S12, 's21': S21, 's22': S22}
                fig, ax = plt.subplots(figsize=(4,3))
                s_param = np.abs(s_param_dict[plot.lower()])
                if yscale == 'dB':
                    s_param = 20*np.log10(s_param)
                    set_ylabel(ax, f'$|S_{{{plot.lower().split("s")[-1]}}}|$ (dB)')
                else:
                    ax.set_yscale(yscale)
                    set_ylabel(ax, f'$|S_{{{plot.lower().split("s")[-1]}}}|$')
                ax.plot(frequency, s_param)
                ax.set_xscale(xscale)
                set_xlabel(ax, 'frequency', 'Hz')
                if plot_phase:
                    axt = ax.twinx()
                    axt.plot(frequency, np.angle(s_param_dict[plot.lower()]), ls='--', color='C2', alpha=.5, zorder=-1)
                    set_ylabel(axt, 'phase', unit='rad')
                    axt.set_yticks([-np.pi, 0, np.pi])
                    axt.set_yticklabels(['$-\\pi$', '0', '$\\pi$'])
                fig.tight_layout()
                ax.set_title('Scattering parameter')
            else:
                # plot all s parameters
                fig, axs = plt.subplots(figsize=(8,5), ncols=2, nrows=2, sharex='col')#, sharey='row')
                for ax, s_param, name in zip(axs.flatten(), [S11, S12, S21, S22], ['S_{11}', 'S_{12}', 'S_{21}', 'S_{22}']):
                    if yscale == 'dB':
                        s_param = 20*np.log10(np.abs(s_param))
                        set_ylabel(ax, f'$|{name}|$ (dB)')
                    else:
                        s_param = np.abs(s_param)
                        ax.set_yscale(yscale)
                        set_ylabel(ax, f'$|{name}|$')
                    ax.plot(frequency, s_param)
                    ax.set_xscale(xscale)
                    if name in ['S_{21}', 'S_{22}']:
                        set_xlabel(ax, 'frequency', 'Hz')
                fig.suptitle('Scattering parameters')
                fig.tight_layout()
        return S11, S12, S21, S22

    def get_Z_parameters(
        self, 
        frequency: list, 
        Z_load: float = None, 
        plot: bool | str = False, 
        **kw,
    ):
        '''
        Compute the Z parameters of the network over a frequency range.
        Args:
            frequency  : frequency array over which to evaluate scattering params
            Z_load     : load impedance used to compute input and output impedance of network.
            plot       : plot scattering parameters. Can be given as bool or one of
                         these strings ('z11', 'z12', 'z21', 'z22', 'zin', 'zout').
            xscale     : axis scale of plot ('linear', 'log').
            yscale     : axis scale of plot ('linear', 'log', 'dB').
            plot_phase : plot magnitude and phase of each z parameter.
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
        # (Z_load here is used to estimate input and output impedances)
        if Z_load is None:
            Z_load = self.Zgen
        Z11, Z12, Z21, Z22, Zin, Zout = extract_Z_pars(M_system, ZL=Z_load)
        # plot parameters
        if plot:
            yscale = kw.get('yscale', 'linear')
            xscale = kw.get('xscale', 'linear')
            # plot single z parameter
            if isinstance(plot, str):
                assert plot.lower() in ['z11', 'z12', 'z21', 'z22', 'zin', 'zout']
                z_param_dict = {'z11': Z11, 'z12': Z12, 'z21': Z21, 'z22': Z22, 'zin': Zin, 'zout': Zout}
                fig, ax = plt.subplots(figsize=(4,3))
                z_param = z_param_dict[plot.lower()]
                if yscale == 'dB':
                    z_param = 10*np.log10(z_param)
                    set_ylabel(ax, f'$Z_{{{plot.lower().split("s")[-1]}}}$ (dB$\Omega$)')
                else:
                    ax.set_yscale(yscale)
                    set_ylabel(ax, f'$Z_{{{plot.lower().split("s")[-1]}}}$', unit='$\Omega$')
                ax.plot(frequency, np.abs(np.real(z_param)), label='$\\mathrm{{Re}}[z]$', color='C0')
                ax.plot(frequency, np.abs(np.imag(z_param)), label='$\\mathrm{{Im}}[z]$', color='C2')
                ax.legend(frameon=False)
                ax.set_xscale(xscale)
                set_xlabel(ax, 'frequency', 'Hz')
                fig.tight_layout()
            else:
                # plot all z parameters
                fig, axs = plt.subplots(figsize=(8,7.5), ncols=2, nrows=3, sharex='col')
                for ax, z_param, name in zip(axs.flatten(), [Zin, Zout, Z11, Z12, Z21, Z22], ['Z_\\mathrm{in}', 'Z_\\mathrm{out}', 'Z_{11}', 'Z_{12}', 'Z_{21}', 'Z_{22}']):
                    if yscale == 'dB':
                        z_param = 10*np.log10(np.abs(np.real(z_param))) + 10j*np.log10(np.abs(np.imag(z_param)))
                        set_ylabel(ax, f'${name}$ (dB $\Omega$)')
                    else:
                        ax.set_yscale(yscale)
                        set_ylabel(ax, f'${name}$', unit='$\Omega$')
                    ax.plot(frequency, np.abs(np.real(z_param)), label='$\\mathrm{{Re}}[z]$', color='C0')
                    ax.plot(frequency, np.abs(np.imag(z_param)), label='$\\mathrm{{Im}}[z]$', color='C2')
                    ax.set_xscale(xscale)
                    if name in ['Z_{21}', 'Z_{22}']:
                        set_xlabel(ax, 'frequency', 'Hz')
                axs[0,1].legend(frameon=False, loc=2, bbox_to_anchor=(1.01, 1))
                fig.suptitle('Impedance parameters')
                fig.tight_layout()
        return Z11, Z12, Z21, Z22, Zin, Zout

    def get_node_VI(
        self, 
        node_idx: int, 
        in_freq: float, 
        in_amp: float = 1, 
        in_phase: float = 0, 
        Z_load : float = None,
    ):
        '''
        Compute the voltage and current phasors at a node of the network for a given input field,
        on port 0 and a load impedance <Z_load> on the output port.
        Args:
            node_idx : node of circuit.
            in_freq  : input signal phasor frequency (Hz).
            in_amp   : input signal phasor amplitude (V).
            in_phase : input signal phasor phase (deg).
            Z_load   : load impedance on ouput port (Ohm).
        '''
        assert node_idx < len(self.elements)+1, f'Network contains only {len(self.elements)+1} nodes.'
        if Z_load is None:
            Z_load = self.Zgen
        # convert frequency into array
        if isinstance(in_freq, float):
            in_freq = np.array([in_freq])
        # compute voltage and current at input node 0
        v0 = in_amp*np.exp(1j*in_phase*np.pi/180)*np.ones(in_freq.shape)
        # input current obtained from input impedance
        z11, z12, z21, z22, zin, zout = self.get_Z_parameters(frequency=in_freq, Z_load=Z_load)
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

    def get_node_power(
        self, 
        node_idx: int, 
        frequency: float, 
        in_power_dBm: float = 0, 
        Z_load: float = None, 
        plot: bool = False,
    ):
        '''
        Compute the power phasor at a node of the network for a given input field,
        on port 0 and a load impedance <Z_load> on the output port.
        Args:
            node_idx     : node of circuit.
            frequency    : input signal frequency (Hz).
            in_power_dBm : input signal power (dBm).
            Z_load       : load impedance on ouput port (Ohm).
            plot         : make fancy plot of power at each node.
        '''
        assert node_idx < len(self.elements)+1, f'Network contains only {len(self.elements)+1} nodes.'
        # default to characteristic impedance
        if Z_load is None:
            Z_load = self.Zgen
        # convert frequency into array
        if isinstance(frequency, float):
            frequency = np.array([frequency])
        # compute Vpk of signal
        in_amp = Vp_from_PdBm(in_power_dBm)
        # get voltage and current phasors at node
        vnode, inode = self.get_node_VI(node_idx=node_idx, in_freq=frequency, in_amp=in_amp, Z_load=Z_load)
        # get voltage and current phasors at node
        pnode_W = np.abs(vnode*inode)/2
        # Convert W to dBm
        pnode_dBm = 10*np.log10(pnode_W*1e3)
        # plot power vs node of network
        if plot:
            assert len(frequency) == 1, 'Can only plot for a fixed frequency'
            n_nodes = len(self.elements)+1
            Powers_dBm = [ 
                self.get_node_power(node_idx=i, frequency=frequency, in_power_dBm=in_power_dBm, Z_load=Z_load, plot=False)[0] \
                for i in range(n_nodes)
                ]
            self.draw_network()
            ax = plt.gcf().add_subplot()
            ax.plot(Powers_dBm, 'C0o-')
            ax.set_xticks(np.arange(n_nodes))
            ax.set_xticklabels([])
            ax.set_ylabel('Power at node (dBm)')
            ax.grid()
            move_subplot(ax, -.047, +1.5/5, scale_x=1+(n_nodes-1)*1.5, scale_y=5)
        return pnode_dBm

    def get_signal_response(
        self, 
        time: list, 
        signal: list, 
        plot: bool = False, 
        **kw,
    ):
        '''
        Get the response of the network to an input signal.
        Performed by computing the scattered signal using S matrix of the network.
        Args:
            time   : frequency array over which to evaluate scattering params
            signal : load impedance used to compute input and output impedance of network
        '''
        # assertion
        assert len(time) == len(signal), 'time and signal axis must have same dimensions!'
        # express signal in frequency domain
        flip = kw.get('flip', False)
        freq_axis, fft = get_fft_from_pulse(time_axis=time, pulse=signal, flip=flip)
        # compute Scattering parameters of network along frequency domain
        s11, s12, s21, s22 = self.get_S_parameters(frequency=freq_axis)
        # resolve nans in scattering parameters
        s11, s12, s21, s22 = solve_nan(s11), solve_nan(s12), solve_nan(s21), solve_nan(s22)
        # compute scattered spectrum
        fft_11, fft_12, fft_21, fft_22 = s11*fft, s12*fft, s21*fft, s22*fft
        # recover signal in time domain
        _time, signal_11 = get_pulse_from_fft(freq_axis=freq_axis, fft=fft_11, flip=flip)
        _time, signal_12 = get_pulse_from_fft(freq_axis=freq_axis, fft=fft_12, flip=flip)
        _time, signal_21 = get_pulse_from_fft(freq_axis=freq_axis, fft=fft_21, flip=flip)
        _time, signal_22 = get_pulse_from_fft(freq_axis=freq_axis, fft=fft_22, flip=flip)
        # assert the signal is real (apart from numerical errors)
        # assert all(np.imag(signal_11)<1e-6)
        # assert all(np.imag(signal_12)<1e-6)
        # assert all(np.imag(signal_21)<1e-6)
        # assert all(np.imag(signal_22)<1e-6)
        # plot input and output signal
        if plot:
            yscale = kw.get('yscale', 'linear')
            xscale = kw.get('xscale', 'linear')
            # plot single s parameter
            if isinstance(plot, str):
                assert plot.lower() in ['s11', 's12', 's21', 's22']
                signal_dict = {'s11': signal_11, 's12': signal_12, 's21': signal_21, 's22': signal_22}
                fig, ax = plt.subplots(figsize=(4,3))
                output_signal = signal_dict[plot.lower()]
                # plot input signal
                ax.plot(time, signal, color='gray', alpha=.5, label='input signal')
                # plot output signal
                ax.plot(_time, np.real(output_signal), label='output signal')
                ax.set_xscale(xscale)
                ax.set_yscale(yscale)
                set_xlabel(ax, 'time', unit='s')
                set_ylabel(ax, f'$\\mathrm{{signal}}_{{{plot.lower().split("s")[-1]}}}$')
                ax.set_title('Scattered signal')
            else:
                # plot all s parameters
                fig, axs = plt.subplots(figsize=(8,5), ncols=2, nrows=2, sharex='col')
                for ax, output_signal, name in zip(axs.flatten(), [signal_11, signal_12, signal_21, signal_22], ['11', '12', '21', '22']):
                    # plot input signal
                    ax.plot(time, signal, color='gray', alpha=.5, label='input signal')
                    # plot output signal
                    ax.plot(time, np.real(output_signal), label='output signal')
                    ax.set_xscale(xscale)
                    ax.set_yscale(yscale)
                    set_ylabel(ax, f'$\\mathrm{{signal}}_{{{name}}}$')
                    if name in ['21', '22']:
                        set_xlabel(ax, 'time', unit='s')
                fig.suptitle('Scattered signal')
            fig.tight_layout()
        return _time, signal_11, signal_12, signal_21, signal_22

    def get_psd_at_node(
        self, 
        frequency: list, 
        node_idx: int = None, 
        initial_node_temp: float = 300, 
        plot: bool = False, 
        add_rates : bool = False, 
        **kw,
    ):
        '''
        Calculate the cascaded power spectral density from all finite temperature elements in the network.
        Args:
            frequency         : frequency array over which to evaluate psd
            node_idx          : node of circuit.
            initial_node_temp : temperature at input of the network (node 0)
            add_rates         : include noise from both absorption and emission
            plot              : plot psd at avery node until node_idx
        '''
        # default to last node
        if node_idx is None:
            node_idx = len(self.elements)
        # Start with a 300 K PSD
        PSD = PSD_thermal_quantum(frequency, T=initial_node_temp)
        if plot:
            fig, ax = plt.subplots(figsize=(4, 6))
            norm = matplotlib.colors.Normalize(vmin=0, vmax=node_idx)
            cmap = matplotlib.colormaps['plasma']
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('frequency (Hz)')
            ax.set_ylabel('PSD, $S$, ($\\mathrm{{\\:W\\:Hz^{-1}}}$)')
            ax.set_title(f'Cumulative PSD at node {node_idx}')
            ax.plot(frequency, PSD, label=f'node 0 ({quantity_to_string(initial_node_temp, "K")})', color=cmap(norm(0)))
            # add temperature axis
            axt = ax.twinx()
            axt.set_yscale('log')
            set_ylabel(axt, 'Temperature (K)')
            def _update_temperature_axis():
                # change ticks to equivalent Johnson-nyquist noise temperature
                _range = ax.get_ylim()
                kB = constants.Boltzmann
                temp_range = [val/(4*kB) for val in _range]
                axt.set_ylim(temp_range)
        # Loop through all nodes up to the selected node
        for idx, (element, _) in enumerate(self.elements[:node_idx]):
            _node_idx = idx+1
            if element == 'attenuator':
                temperature_K = self.element_properties[idx]['temperature_K']
                attn_dB = self.element_properties[idx]['attn_dB']
                attn = 10**(-attn_dB/10) # here we should compute attenuation in power
                # cascaded attenuation
                PSD = (1-attn)*PSD_thermal_quantum(frequency, temperature_K) + attn*PSD
                # add both emission and absorption rate
                if add_rates:
                    PSD += (1-attn)*PSD_thermal_quantum(-frequency, temperature_K)
            elif element == 'amplifier':
                temperature_K = self.element_properties[idx]['temperature_K']
                gain_dB = self.element_properties[idx]['gain_dB']
                gain = 10**(gain_dB/10) # here we should compute gain in power
                # cascaded attenuation
                PSD = PSD_thermal_quantum(frequency, temperature_K) + gain*PSD
                # add both emission and absorption rate
                if add_rates:
                    PSD += PSD_thermal_quantum(-frequency, temperature_K)
            elif element == 'commercial_component':
                temperature_K = self.element_properties[idx]['temperature_K']
                # Get attenuation of component versus frequency
                _, _, s21, _ = extract_S_pars(self.elements[idx][1](frequency), Zgen=self.Zgen)
                attn = np.abs(s21)**2
                PSD = (1-attn)*PSD_thermal_quantum(frequency, temperature_K) + attn*PSD
                # add both emission and absorption rate
                if add_rates:
                    PSD += (1-attn)*PSD_thermal_quantum(-frequency, temperature_K)
            else:
                temperature_K = initial_node_temp
            # plot current PSD
            if plot:
                ax.plot(frequency, PSD, label=f'node {_node_idx} ({quantity_to_string(temperature_K, "K")})', color=cmap(norm(_node_idx)))
        if plot:
            # add a label to plot
            label_tuple = kw.get('label', None)
            if label_tuple:
                name, temperature = label_tuple
                axt.axhline(temperature, color='0.8', ls='--')
                axt.text(frequency.max(), temperature, name, va='bottom', ha='right', color='0.8')
            ax.legend(frameon=False, loc=2, bbox_to_anchor=(1.2, 1))
            _update_temperature_axis()
        else:
            return PSD

    def get_svv_at_output(
        self, 
        frequency: list, 
        Z_load: float = None, 
        add_rates: bool = False, 
        plot: bool = False,
    ):
        '''
        Calculate spectral density of squared voltage 
        in units of V^2/Hz (Volt squared per Hertz).
        (used for Fermi's golden rule).
        Args:
            frequency : frequency, should be given as array
            Z_load    : load impedance at input of network
            plot      : plot Svv at node_idx
        '''

        # Compute scattering parameters
        # (Z_load here is used to estimate input and output impedances)
        if Z_load is None:
            Z_load = self.Zgen
        # get power spectral density
        n_elements = len(self.elements)
        psd = self.get_psd_at_node(node_idx=n_elements, frequency=frequency, add_rates=add_rates)
        # calculate output impedance
        _, _, _, _, _, zout = self.get_Z_parameters(frequency=frequency, Z_load=Z_load)
        # calculate Svv
        s_vv = psd*np.real(zout)
        # plot
        if plot:
            fig, ax = plt.subplots(figsize=(5,3.5))
            ax.plot(frequency, s_vv)
            ax.set_xscale('log')
            ax.set_yscale('log')
            set_xlabel(ax, 'frequency (Hz)')
            set_ylabel(ax, '$S_{{VV}}$ ($\\mathrm{{V^2\\:Hz^{{-1}}}}$)')
            ax.set_title('Voltage spectral density')
        return s_vv

    def get_sii_at_output(
        self, 
        frequency: list, 
        Z_load: float = None, 
        add_rates: bool = False, 
        plot: bool = False,
    ):
        '''
        Calculate spectral density of squared current
        in units of A^2/Hz (Ampere squared per Hertz).
        (used for Fermi's golden rule)
        Args:
            frequency : frequency, should be given as array
            Z_load    : load impedance at input of network
            plot      : plot Sii at node_idx
        '''

        # Compute scattering parameters
        # (Z_load here is used to estimate input and output impedances)
        if Z_load is None:
            Z_load = self.Zgen
        # get power spectral density
        n_elements = len(self.elements)
        psd = self.get_psd_at_node(node_idx=n_elements, frequency=frequency, add_rates=add_rates)
        # calculate output impedance
        _, _, _, _, _, zout = self.get_Z_parameters(frequency=frequency, Z_load=Z_load)
        # calculate Sii
        s_ii = psd/np.real(zout)
        # plot
        if plot:
            fig, ax = plt.subplots(figsize=(5,3.5))
            ax.plot(frequency, s_ii)
            ax.set_xscale('log')
            ax.set_yscale('log')
            set_xlabel(ax, 'frequency (Hz)')
            set_ylabel(ax, '$S_{{II}}$ ($\\mathrm{{A^2\\:Hz^{{-1}}}}$)')
            ax.set_title('Current spectral density')
        return s_ii

    def get_spp_at_output(
        self, 
        frequency: list, 
        M_henry: float, 
        Z_load: float = None, 
        add_rates: bool = False, 
        plot: bool = False,
    ):
        '''
        Calculate spectral density of squared flux from mutual inductive coupling
        in units of phi0^2/Hz (flux quantum squared per Hertz).
        (used for Fermi's golden rule)
        Args:
            frequency : frequency, should be given as array
            M_henry   : Mutual inductance of line
            Z_load    : load impedance at input of network
            plot      : plot Svv at node_idx
        '''
        # Compute scattering parameters
        # (Z_load here is used to estimate input and output impedances)
        if Z_load is None:
            Z_load = self.Zgen
        # get current spectral density
        s_ii = self.get_sii_at_output(frequency=frequency, Z_load=Z_load, add_rates=add_rates)
        # convert mutual inductance from henry (Webber/amp) to phi0/amp
        phi_0 = constants.physical_constants['mag. flux quantum'][0]
        M_phi0 = M_henry/phi_0
        # calculate flux spectral density
        s_pp = M_phi0**2 * s_ii
        # plot
        if plot:
            fig, ax = plt.subplots(figsize=(5,3.5))
            ax.plot(frequency, s_pp)
            ax.set_xscale('log')
            ax.set_yscale('log')
            set_xlabel(ax, 'frequency (Hz)')
            set_ylabel(ax, '$S_{{\\Phi\\Phi}}$ ($\\mathrm{{{{\\Phi_0}}^2\\:Hz^{{-1}}}}$)')
            ax.set_title('Flux spectral density')
        return s_pp

    def find_resonance_parameters(
        self, 
        frequency_bounds: tuple, 
        kappa_bounds: tuple = (-5e6, +5e6),
    ):
        '''
        Function used to detect resonance modes.
        This is done by searching for poles and zeros in complex
        frequency space. 
        Returns the resonance frequency and kappa of the mode.
        '''
        # initial guess is estimated with linear frequency sweep
        freq_axis = np.linspace(*frequency_bounds, int(1e4))
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
        bounds = [frequency_bounds, kappa_bounds]
        _, k0 = minimize(cost_func, x0=initial_guess, options={'disp':False}, bounds=bounds, method='Powell').x
        return f0, k0

    def sweep_element_parameter(
        self, 
        value: float, 
        element_idx: int, 
        element_property: str,
    ):
        '''
        Sweeps a parameter of a given element in the network.
        Args:
            value            : value of parameter.
            element_idx      : position index in network.
            element_property : parameter to sweep in the form of string.
        '''
        # Make sure element exists
        assert element_property in self.element_properties[element_idx].keys(), 'property not in element!'
        # compile element properties except for the sweep
        properties = {key: val for key, val in self.element_properties[element_idx].items() if key != element_property}
        # get element name
        element_name = self.elements[element_idx][0]
        if ('_series' in element_name):
            element_name = element_name.split('_series')[0]
            properties['element_type'] = 'series'
        elif ('_parallel' in element_name):
            element_name = element_name.split('_parallel')[0]
            properties['element_type'] = 'parallel'
        # find "add_element" function
        add_methods = [ method for method in dir(self) if 'add' in method ]
        method_name = [ method for method in add_methods if f'add_{element_name}' in method ][0]
        method = getattr(self, method_name)
        # change property value
        properties[element_property] = value
        # change method
        method(element_idx=element_idx, **properties)

    def get_S_versus_parameters(
        self, 
        frequency: list, 
        values: list, 
        element_idx: int, 
        element_property: str, 
        plot: bool = False, 
        **kw,
    ):
        '''
        Get the scattering parameters of the resulting networkfor a given frequency
        axis and a given element parameter axis.
        Args:
            frequency        : frequency array over which to evaluate scattering params.
            values           : values of parameter as 1D array.
            element_idx      : position index in network.
            element_property : parameter to sweep in the form of string.
            plot             : plot scattering parameters. Can be given as bool or one of
                               these strings ('s11', 's12', 's21', 's22').
            xscale           : axis scale of plot ('linear', 'log').
            yscale           : axis scale of plot ('linear', 'log', 'dB').
            plot_phase       : plot magnitude and phase of each s parameter.
        '''
        if type(values) is list:
            values = np.array(values)
        if type(frequency) is list:
            frequency = np.array(frequency)
        S11, S12, S21, S22 = np.zeros((len(values),len(frequency)), dtype=complex),\
                             np.zeros((len(values),len(frequency)), dtype=complex),\
                             np.zeros((len(values),len(frequency)), dtype=complex),\
                             np.zeros((len(values),len(frequency)), dtype=complex)
        for i, val in enumerate(values):
            # sweep element parameter
            self.sweep_element_parameter(value=val, element_idx=element_idx, element_property=element_property)
            S11[i], S12[i], S21[i], S22[i] = self.get_S_parameters(frequency=frequency)
        # plot parameters
        if plot:
            yscale = kw.get('yscale', 'linear')
            xscale = kw.get('xscale', 'linear')
            cmap = kw.get('cmap', 'viridis')
            plot_phase = kw.get('plot_phase', False)
            unit = kw.get('unit', None)
            # plot single s parameter
            if isinstance(plot, str):
                assert plot.lower() in ['s11', 's12', 's21', 's22']
                s_param_dict = {'s11': S11, 's12': S12, 's21': S21, 's22': S22}
                fig, ax = plt.subplots(figsize=(4,3))
                s_param = np.abs(s_param_dict[plot.lower()])
                if yscale == 'dB':
                    s_param = 20*np.log10(s_param)
                    set_ylabel(ax, f'$|S_{{{plot.lower().split("s")[-1]}}}|$ (dB)')
                else:
                    ax.set_yscale(yscale)
                    set_ylabel(ax, f'$|S_{{{plot.lower().split("s")[-1]}}}|$')
                for i, val in enumerate(values):
                    c_percentage = (val-values.min())/(values.max()-values.min())
                    plot_line_cmap(frequency, s_param[i], color_percentage=c_percentage, cmap=cmap)
                ax.set_xscale(xscale)
                set_xlabel(ax, 'frequency', 'Hz')
                if plot_phase:
                    axt = ax.twinx()
                    axt.plot(frequency, np.angle(s_param_dict[plot.lower()]), ls='--', color='C2', alpha=.5, zorder=-1)
                    set_ylabel(axt, 'phase', unit='rad')
                    axt.set_yticks([-np.pi, 0, np.pi])
                    axt.set_yticklabels(['$-\\pi$', '0', '$\\pi$'])
                fig.tight_layout()
                ax.set_title('Scattering parameter')
                get_cbar(ax=ax, orientation='vertical', pos=None, dims=None, label=element_property, unit=unit, vmin=values.min(), vmax=values.max())
            else:
                # plot all s parameters
                fig, axs = plt.subplots(figsize=(8,5), ncols=2, nrows=2, sharex='col')#, sharey='row')
                for ax, s_param, name in zip(axs.flatten(), [S11, S12, S21, S22], ['S_{11}', 'S_{12}', 'S_{21}', 'S_{22}']):
                    if yscale == 'dB':
                        s_param = 20*np.log10(np.abs(s_param))
                        set_ylabel(ax, f'$|{name}|$ (dB)')
                    else:
                        s_param = np.abs(s_param)
                        ax.set_yscale(yscale)
                        set_ylabel(ax, f'$|{name}|$')
                    for i, val in enumerate(values):
                        c_percentage = (val-values.min())/(values.max()-values.min())
                        plot_line_cmap(frequency, s_param[i], ax=ax, color_percentage=c_percentage, cmap=cmap)
                    ax.set_xscale(xscale)
                    if name in ['S_{21}', 'S_{22}']:
                        set_xlabel(ax, 'frequency', 'Hz')
                fig.suptitle('Scattering parameters')
                fig.tight_layout()
                get_cbar(ax=ax, orientation='vertical', pos=(1.1, 0), dims=(.05, 2.1), label=element_property, unit=unit, vmin=values.min(), vmax=values.max())
        return S11, S12, S21, S22

    #################################################
    # Plot network
    #################################################
    def draw_network(
        self,
    ):
        '''
        Draw schematic of network.
        '''
        fig, ax = plt.subplots(figsize=(1,1))
        ax.set_xlim(0, .9)
        ax.set_ylim(-.45, .45)
        ax.axis('off')
        for i, (name, element) in enumerate(self.elements):
            x_i = 1.3*i
            x_j = 1.3*(i+1)
            xij = np.mean([x_i, x_j])
            # plot capacitor in series
            if name == 'capacitor_series':
                capacitance = self.element_properties[i]['capacitance']
                cd.plot_capacitor(x_i, 0, x_j, 0, l_cap=.2, cap_dist=.2)
                ax.text(xij, +.5, quantity_to_string(capacitance, 'F'), va='center', ha='center', color='gray')
            # plot capacitor in parallel
            elif name == 'capacitor_parallel':
                capacitance = self.element_properties[i]['capacitance']
                ax.plot([x_i, x_j], [0, 0], color='k', solid_capstyle='round', linewidth=4, clip_on=False)
                cd.plot_capacitor(xij, 0, xij, -1.2, l_cap=.2, cap_dist=.2)
                cd.plot_ground(xij, -1., .2, horizontal=False)
                ax.text(xij-.3, -.6, quantity_to_string(capacitance, 'F'), va='center', ha='right', color='gray')
            # plot inductor in series
            elif name == 'inductor_series':
                inductance = self.element_properties[i]['inductance']
                cd.plot_inductor(x_i, 0, x_j, 0, w_ind=.2, lpad=.25)
                ax.text(xij, +.5, quantity_to_string(inductance, 'H'), va='center', ha='center', color='gray')
            # plot inductor in parallel
            elif name == 'inductor_parallel':
                inductance = self.element_properties[i]['inductance']
                ax.plot([x_i, x_j], [0, 0], color='k', solid_capstyle='round', linewidth=4, clip_on=False)
                cd.plot_inductor(xij, 0, xij, -1.2, w_ind=.2, lpad=.25)
                cd.plot_ground(xij, -1., .2, horizontal=False)
                ax.text(xij-.3, -.6, quantity_to_string(inductance, 'H'), va='center', ha='right', color='gray')
            # plot resistor in series
            elif name == 'resistor_series':
                resistance = self.element_properties[i]['resistance']
                cd.plot_resistor(x_i, 0, x_j, 0, w_ind=-.3, lpad=.3)
                ax.text(xij, +.5, quantity_to_string(resistance, '$\Omega$'), va='center', ha='center', color='gray')
            # plot inductor in parallel
            elif name == 'resistor_parallel':
                resistance = self.element_properties[i]['resistance']
                ax.plot([x_i, x_j], [0, 0], color='k', solid_capstyle='round', linewidth=4, clip_on=False)
                cd.plot_resistor(xij, 0, xij, -1.2, w_ind=.3, lpad=.3)
                cd.plot_ground(xij, -1., .2, horizontal=False)
                ax.text(xij-.3, -.6, quantity_to_string(resistance, '$\Omega$'), va='center', ha='right', color='gray')
            # plot transmission line
            elif name == 'transmission_line':
                length = self.element_properties[i]['length']
                cd.plot_transmission_line(x_i+.05, 0, 1.2, horizontal=True, radius=.2)
                ax.text(xij, +.5, quantity_to_string(length, 'm', decimal_place=2), va='center', ha='center', color='gray')
            # plot parallel transmission line
            elif name == 'parallel_transmission_line':
                length = self.element_properties[i]['length']
                ax.plot([x_i, x_j], [0, 0], color='k', solid_capstyle='round', linewidth=4, clip_on=False)
                cd.plot_ground(xij, -1.1, .1, horizontal=False)
                ax.text(xij-.3, -.6, quantity_to_string(length, 'm', decimal_place=2), va='center', ha='right', color='gray')
                Z_load = self.element_properties[i]['Z_load']
                if Z_load == 0:
                    cd.plot_transmission_line(xij, 0, .9+.2, horizontal=False, radius=.2)
                elif np.abs(Z_load) > 1e5:
                    cd.plot_transmission_line(xij, 0, .8, horizontal=False, radius=.2)
                    ax.plot([xij, xij], [-.8, -1.05], color='k', ls='None', marker='o', linewidth=4, clip_on=False)
                else:
                    cd.plot_transmission_line(xij, 0, .7, horizontal=False, radius=.2)
                    cd.plot_resistor(x0=xij, y0=-.7, x1=xij, y1=-1.1, w_ind=.2)
            # plot transformer
            elif name == 'transformer':
                L1 = self.element_properties[i]['L1']
                L2 = self.element_properties[i]['L2']
                M = self.element_properties[i]['M']
                ax.plot([x_i, xij-.3], [0, 0], color='k', solid_capstyle='round', linewidth=4, clip_on=False)
                ax.plot([x_j, xij+.3], [0, 0], color='k', solid_capstyle='round', linewidth=4, clip_on=False)
                cd.plot_inductor(xij-.3, 0, xij-.3, -1.2, w_ind=.125, lpad=.25)
                cd.plot_inductor(xij+.3, -1.2, xij+.3, 0, w_ind=.125, lpad=.25)
                cd.plot_ground(xij-.3, -1., .2, horizontal=False)
                cd.plot_ground(xij+.3, -1., .2, horizontal=False)
                ax.plot([xij-.05, xij-.05], [-.25, -.95], color='k', solid_capstyle='round', linewidth=2, clip_on=False)
                ax.plot([xij+.05, xij+.05], [-.25, -.95], color='k', solid_capstyle='round', linewidth=2, clip_on=False)
                ax.text(xij, -.1, quantity_to_string(M, 'H'), va='bottom', ha='center', color='gray', fontsize=8)
                ax.text(xij-.5, -.6, quantity_to_string(L1, 'H'), va='center', ha='right', color='gray', fontsize=8)
                ax.text(xij+.5, -.6, quantity_to_string(L2, 'H'), va='center', ha='left', color='gray', fontsize=8)
            # plot capacitively coupled resonator
            elif name == 'capacitively_coupled_hanger':
                ax.plot([x_i, x_j], [0, 0], color='k', solid_capstyle='round', linewidth=4, clip_on=False)
                cd.plot_capacitor(xij, 0, xij, -.35, l_cap=.1, cap_dist=.12)
                cd.plot_ground(xij, -1.1, .1, horizontal=False)
                cd.plot_transmission_line(xij, -.25, .9, horizontal=False, radius=.2)
                length = self.element_properties[i]['length']
                cap = self.element_properties[i]['C_coupling']
                ax.text(xij-.2, -.2, quantity_to_string(cap, 'F', decimal_place=1), va='center', ha='right', color='gray')
                ax.text(xij-.3, -.7, quantity_to_string(length, 'm', decimal_place=1), va='center', ha='right', color='gray')
            # plot inductively coupled resonator
            elif name == 'inductively_coupled_hanger':
                cd.plot_inductor(x_j, 0, x_i, 0, w_ind=.08, lpad=.4)
                cd.plot_inductor(x_i+.2, -.3, x_j-.2, -.3, w_ind=.08, lpad=.2)
                cd.plot_ground(x_j-.2, -.3, .2, horizontal=False)
                ax.plot([xij-.2, xij+.2], [-.135, -.135], color='gray', lw=1, clip_on=False, solid_capstyle='round')
                ax.plot([xij-.2, xij+.2], [-.165, -.165], color='gray', lw=1, clip_on=False, solid_capstyle='round')
                length = self.element_properties[i]['length']
                L_line = self.element_properties[i]['L_line']
                L_hang = self.element_properties[i]['L_hanger']
                M_ind = self.element_properties[i]['M_inductance']
                ax.text(xij-.15, -.85, quantity_to_string(length, 'm', decimal_place=1), va='center', ha='left', color='gray')
                ax.text(xij, +.25, quantity_to_string(L_line, 'H', decimal_place=1), va='center', ha='center', color='gray', fontsize=7)
                ax.text(xij, -.5, quantity_to_string(L_hang, 'H', decimal_place=1), va='center', ha='center', color='gray', fontsize=7)
                ax.text(xij+.275, -.15, quantity_to_string(M_ind, 'H', decimal_place=1), va='center', ha='left', color='gray', fontsize=7)
                Z_termination = self.element_properties[i]['Z_termination']
                if Z_termination == 0:
                    cd.plot_transmission_line(x_i+.2, -.3, .9, horizontal=False, radius=.2)
                    cd.plot_ground(x_i+.2, -1.1, .1, horizontal=False)
                elif np.abs(Z_termination) > 1e5:
                    cd.plot_transmission_line(x_i+.2, -.3, .9, horizontal=False, radius=.2)
                    # ax.plot([x_i+.2], [-1.2], color='k', ls='None', marker='o', markersize=6, clip_on=False)
                else:
                    cd.plot_transmission_line(x_i+.2, -.3, .9, horizontal=False, radius=.2) # FIX this
            # plot SQUID in series
            elif name == 'SQUID_series':
                phi = self.element_properties[i]['phi']
                cd.plot_SQUID(x_i, 0, x_j, 0, w=.6, l_junction=.125)
                ax.text(xij, 0, '\n'.join(quantity_to_string(phi, '$\\Phi_0$').split(' ')), va='center', ha='center', color='gray', fontsize=7)
            # plot SQUID in parallel
            elif name == 'SQUID_parallel':
                phi = self.element_properties[i]['phi']
                ax.plot([x_i, x_j], [0, 0], color='k', solid_capstyle='round', linewidth=4, clip_on=False)
                cd.plot_SQUID(xij, 0, xij, -1.2, w=.6, l_junction=.125)
                cd.plot_ground(xij, -1., .2, horizontal=False)
                ax.text(xij, -.6, '\n'.join(quantity_to_string(phi, '$\\Phi_0$').split(' ')), va='center', ha='center', color='gray', fontsize=7)
            # plot attenuator
            elif 'attenuator' in name:
                attn_dB = self.element_properties[i]['attn_dB']
                temperature_K = self.element_properties[i]['temperature_K']
                cd.plot_attenuator(xij, 0, h=.3, l=.6, attn_dB=attn_dB)
                ax.plot([x_i, x_j], [0, 0], color='k', solid_capstyle='round', linewidth=4, clip_on=False)
                ax.text(xij, -.5, quantity_to_string(temperature_K, 'K'), va='center', ha='center')
            # plot attenuator
            elif 'amplifier' in name:
                gain_dB = self.element_properties[i]['gain_dB']
                temperature_K = self.element_properties[i]['temperature_K']
                cd.plot_amplifier(xij, 0, h=.7, l=.7, gain_dB=gain_dB)
                ax.plot([x_i, x_j], [0, 0], color='k', solid_capstyle='round', linewidth=4, clip_on=False)
                ax.text(xij, -.5, quantity_to_string(temperature_K, 'K'), va='center', ha='center')
            # plot commercial component
            elif 'commercial_component' in name:
                _type = self.element_properties[i]['type']
                _model = self.element_properties[i]['model']
                if 'low pass' in _type.lower():
                    cd.plot_low_pass(xij, 0, h=.4, l=.8, name=_model)
                elif 'high pass' in _type.lower():
                    cd.plot_high_pass(xij, 0, h=.4, l=.8, name=_model)
                ax.plot([x_i, x_j], [0, 0], color='k', solid_capstyle='round', linewidth=4, clip_on=False)
                try:
                    temperature_K = self.element_properties[i]['temperature_K']
                    ax.text(xij, -.5, quantity_to_string(temperature_K, 'K'), va='center', ha='center')
                except exception as e:
                    print(e)
                    print(f'Could not find temperature for component {_type} {_model}!')
            else:
                ax.plot([x_i, x_j], [0, 0], color='k', solid_capstyle='round', linewidth=4, clip_on=False)
                ax.text(xij, 0, '?', ha='center', va='center')
            # plot node
            ax.plot([x_i], [0], color='k', marker='o', markersize=10, clip_on=False)
            ax.text(x_i, .3, f'{i}', size=12, va='center', ha='center')
        # final node
        ax.plot([x_j], [0], color='k', marker='o', markersize=10, clip_on=False)
        ax.text(x_j, .3, f'{i+1}', size=12, va='center', ha='center')
        self.fig = fig
        # plt.show()

# end