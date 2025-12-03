import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import constants

##############################################
# networks helper functions
##############################################

def Z_capacitor(Freq: np.array, C: float):
    '''
    Impedance of capacitor.
    Args:
        Freq : frequency in Hz
        C    : capacitance in F
    '''
    w = 2*np.pi*Freq
    Z_in = -1j/(w*C)
    return Z_in

def Z_inductor(Freq: np.array, L: float):
    '''
    Impedance of inductor.
    Args:
        Freq : frequency in Hz
        L    : inductance in H
    '''
    w = 2*np.pi*Freq
    Z_in = 1j*w*L
    return Z_in

def Z_resistor(Freq: np.array, R: float):
    '''
    Impedance of resistor.
    Args:
        Freq : frequency in Hz
        R    : resistance in Ohms
    '''
    w = 2*np.pi*Freq
    Z_in = np.ones(w.shape, dtype=np.complex_)*R
    return Z_in

def Z_term_transline(Freq: np.array, l: float, v: float, Z0: float, ZL: float = 0):
    '''
    Input impedance of terminated transmission line.
    Args:
        length : length of transmission line in meters
        v      : speed of light in transmission line in m/s
        Z0     : characteristic impedance of transmission line in Ohms
        ZL     : (termination) load impedance in Ohms
    '''
    beta = 2*np.pi*Freq/v
    Z_in = Z0*( ZL + 1j*Z0*np.tan(beta*l) )/( Z0 + 1j*ZL*np.tan(beta*l) )
    return Z_in

def Z_parallel(Za: np.array, Zb: np.array):
    '''
    Effective impedance of two parallel impedances.
    Args:
        Za/Zb : impedance of parallel elements
    '''
    Z_par = Za*Zb/(Za+Zb)
    return Z_par

def M_series_impedance(Z: np.array):
    '''
    ABCD matrix for a series impedance.
    Args:
        Z : impedance of element in series
    '''
    # Define coefficient functions for ABCD
    def A(x): return np.ones(x.shape, dtype=complex)
    def B(x): return x
    def C(x): return np.zeros(x.shape, dtype=complex)
    def D(x): return np.ones(x.shape, dtype=complex)
    # generate M matrices for each impedance
    M = np.stack([
        np.stack([A(Z), B(Z)], axis=-1),
        np.stack([C(Z), D(Z)], axis=-1)
    ], axis=-2)
    return M

def M_parallel_impedance(Z: np.array):
    '''
    ABCD matrix for a parallel impedance.
    Args:
        Z : impedance of element in parallel
    '''
    # Define coefficient functions for ABCD
    def A(x): return np.ones(x.shape, dtype=complex)
    def B(x): return np.zeros(x.shape, dtype=complex)
    def C(x): return 1/x
    def D(x): return np.ones(x.shape, dtype=complex)
    # generate M matrices for each impedance
    M = np.stack([
        np.stack([A(Z), B(Z)], axis=-1),
        np.stack([C(Z), D(Z)], axis=-1)
    ], axis=-2)
    return M

def M_series_transline(Freq: np.array, l: float, v: float, Z0: float):
    '''
    ABCD matrix for a series transmission line.
    Args:
        Freq   : frequency in Hz
        length : length of transmission line in meters
        v      : speed of light in transmission line in m/s
        Z0     : characteristic impedance of transmission line in Ohms
    '''
    beta = 2*np.pi*Freq/v
    # Define coefficient functions for ABCD
    def A(x): return np.cos(x*l).astype(complex)
    def B(x): return 1j*Z0*np.sin(x*l)
    def C(x): return 1j/Z0*np.sin(x*l)
    def D(x): return np.cos(x*l).astype(complex)
    # generate M matrices for each impedance
    M = np.stack([
        np.stack([A(beta), B(beta)], axis=-1),
        np.stack([C(beta), D(beta)], axis=-1)
    ], axis=-2)
    return M

def M_inductively_coupled_impedance(Freq: np.array, Z: float, L1: float, L2: float, M:float):
    '''
    ABCD matrix of a parallel impedance that is mutual iductively coupled:
         _________
    ----|--mmm----|-----
        |  mmm    |
        |  |      |
        | | |     |
        | |Z|     |
        | |_|     |
        |  |      |
        |__V______|
    Args:
        Freq : frequency in Hz
        Z    : impedance of parallel component
        L1   : inductance of main line
        L2   : inductance of impedance line
        M    : mutual inductance between inductors
    '''
    w = 2*np.pi*Freq
    Z_ = 1j*w*L1 + (1j*w*M)**2 / (Z+1j*w*L2)
    # Compute ABCD matrix
    M = M_series_impedance(Z_)
    return M

def M_transformer(Freq: np.array, L1: float, L2: float, M: float):
    '''
    ABCD matrix of a series transformer:
            M
    o----| || |-----o
         3 || 3
     L1  3 || 3  L2
         3 || 3
         | || |
    Args:
        Freq : frequency in Hz
        Z    : impedance of parallel component
        L1   : inductance of main line
        L2   : inductance of impedance line
        M    : mutual inductance between inductors
    '''
    # T network transformation
    La = L1-M
    Lb = L2-M
    Lc = M
    # impedance
    Za = Z_inductor(Freq, L=La)
    Zb = Z_inductor(Freq, L=Lb)
    Zc = Z_inductor(Freq, L=Lc)
    # T network ABCD matrix
    def A(x): return 1+Z_inductor(x, L=La)/Z_inductor(x, L=Lc)
    def B(x): return Z_inductor(x, L=La)+Z_inductor(x, L=Lb)+Z_inductor(x, L=La)*Z_inductor(x, L=Lb)/Z_inductor(x, L=Lc)
    def C(x): return 1/Z_inductor(x, L=Lc)
    def D(x): return 1+Z_inductor(x, L=Lb)/Z_inductor(x, L=Lc)
    # generate M matrices for each impedance
    M = np.stack([
        np.stack([A(Freq), B(Freq)], axis=-1),
        np.stack([C(Freq), D(Freq)], axis=-1)
    ], axis=-2)
    return M

def M_attenuator(Freq: np.array, attn_dB: float, Z0:float):
    '''
    ABCD matrix of a standard Pi attenuator circuit:
           R2       
    ---|--WWWW--|---
       Z        Z   
    R1 Z        Z R1
       |        |   
    ----------------
    Args:
        Freq    : frequency in Hz
        attn_dB : power attenuation in dB
        Z0      : characteristic impedance in Ohms
    '''
    assert attn_dB <=40, 'Attenuation must be at most 40 dB'
    w = 2*np.pi*Freq
    # attenuation linear
    attn = 10**(attn_dB/20)
    # deal with 0 dB attenuators
    if attn_dB == 0:
        # ABCD matrix coefficients
        A = 1
        B = 0
        C = 0
        D = 1
    else:
        R1 = Z0*((attn+1)/(attn-1))
        R2 = Z0/2*(attn-1/attn)
        # ABCD matrix coefficients
        A_coef = 1 + R2/R1
        B_coef = R2
        C_coef = 2/R1 + R2/(R1**2)
        D_coef = 1 + R2/R1
    # ABCD coefficient functions
    def A(x): return np.ones(x.shape, dtype=complex)*A_coef
    def B(x): return np.ones(x.shape, dtype=complex)*B_coef
    def C(x): return np.ones(x.shape, dtype=complex)*C_coef
    def D(x): return np.ones(x.shape, dtype=complex)*D_coef
    # generate M matrices for each impedance
    M = np.stack([
        np.stack([A(Freq), B(Freq)], axis=-1),
        np.stack([C(Freq), D(Freq)], axis=-1)
    ], axis=-2)
    return M

def M_amplifier(Freq: np.array, gain_dB: float, Z0:float):
    '''
    An ABCD matrix for an amplifier.
    (Using a pi attenuator network with negative attenuation. 
     May lord have mercy on us...)
    Args:
        Freq    : frequency in Hz
        gain_dB : power gain in dB
        Z0      : characteristic impedance in Ohms
    '''
    assert gain_dB <=50, 'Gain must be at most 50 dB'
    # attenuation linear
    gain = 10**(-gain_dB/20)
    R1 = Z0*((gain+1)/(gain-1))
    R2 = Z0/2*(gain-1/gain)
    # ABCD matrix coefficients
    A_coef = 1 + R2/R1
    B_coef = R2
    C_coef = 2/R1 + R2/(R1**2)
    D_coef = 1 + R2/R1
    # ABCD coefficient functions
    def A(x): return np.ones(x.shape, dtype=complex)*A_coef
    def B(x): return np.ones(x.shape, dtype=complex)*B_coef
    def C(x): return np.ones(x.shape, dtype=complex)*C_coef
    def D(x): return np.ones(x.shape, dtype=complex)*D_coef
    # generate M matrices for each impedance
    M = np.stack([
        np.stack([A(Freq), B(Freq)], axis=-1),
        np.stack([C(Freq), D(Freq)], axis=-1)
    ], axis=-2)
    return M

def multiply_matrices(M_list):
    '''
    Method used to concatenate a series of ABCD matrices.
    Args:
        M_list : List of ABCD matrix instances
    '''
    for i, m in enumerate(M_list):
        assert m.shape[-2:]==(2,2), f'M_list[{i}] must have shape (n1, n2, ..., 2, 2)'
    # Initialize the result as the first array
    result = M_list[0]
    # Compute the dot product sequentially with each subsequent array
    for arr in M_list[1:]:
        result = np.matmul(result, arr)
    return result

def extract_S_pars(M, Zgen):
    '''
    Method used to calculate scattering parameters from an ABCD matrix.
    (Pozar page 192)
    Args:
        M    : ABCD matrix used
        Zgen : port impedance used to compute scattering coefficients
    '''
    _shape = M.shape
    assert _shape[-2:]==(2,2), 'M should have shape (n, 2, 2)'
    A, B, C, D = M[...,0,0], M[...,0,1], M[...,1,0], M[...,1,1]
    # Compute scattering parameters
    S11 = (A + B/Zgen - C*Zgen - D)/( A + B/Zgen + C*Zgen + D )
    S12 = 2*(A*D - B*C)/( A + B/Zgen + C*Zgen + D )
    S21 = 2/( A + B/Zgen + C*Zgen + D )
    S22 = (-A + B/Zgen - C*Zgen + D)/( A + B/Zgen + C*Zgen + D )
    return S11, S12, S21, S22

def extract_Z_pars(M, ZL):
    '''
    Method used to calculate impedance matrix parameters 
    as well as input/ouput impedance from an ABCD matrix.
    (Pozar page 192)
    Args:
        M  : ABCD matrix used
        ZL : load impedance used to compute input/output impedance coefficients
    '''
    _shape = M.shape
    assert _shape[-2:]==(2,2), 'M should have shape (n, 2, 2)'
    # Compute Z matrix
    Z11 = M[...,0,0]/M[...,1,0]
    Z12 = (M[...,0,0]*M[...,1,1] - M[...,0,1]*M[...,1,0])/M[...,1,0]
    Z21 = 1/M[...,1,0]
    Z22 = M[...,1,1]/M[...,1,0]
    # Compute input and output impedance
    Zin = Z11 - (Z12*Z21)/(Z22+ZL)
    Zout = Z22 - (Z12*Z21)/(Z11+ZL)
    return Z11, Z12, Z21, Z22, Zin, Zout

def get_fft_from_pulse(time_axis, pulse, flip=False, plot=False):
    '''
    Method used to compute the FFT and correspoding frequency axis
    from a pulse in the time domain.
    Args:
        time_axis : time coordinates of pulse
        pulse     : pulse coordinates 
    '''
    # compute frequency axis
    dt = time_axis[1]-time_axis[0]
    n = time_axis.size
    freq_axis = np.fft.fftfreq(n, d=dt)
    # compute fft
    if flip:
        pulse = np.flip(pulse)
    fft = np.fft.fft(pulse)
    # plot
    if plot:
        fig, ax = plt.subplots(figsize=(4,3))
        ax.plot(np.sort(freq_axis), np.abs(fft[np.argsort(freq_axis)])**2)
        ax.set_yscale('log')
        set_xlabel(ax, 'frequency', 'Hz')
        set_ylabel(ax, 'FFT')
    return freq_axis, fft

def get_pulse_from_fft(freq_axis, fft, flip=False):
    '''
    Method used to compute a pulse in the time domain 
    from a FFT and corresponding frequency axis.
    Args:
        freq_axis : frequency coordinates of FFT
        fft       : fast Fourier transform coefficients
    '''
    # compute frequency axis
    df = freq_axis[1]-freq_axis[0]
    n = freq_axis.size
    time_axis = np.fft.fftfreq(n, d=df)
    # compute inverse fft
    pulse = np.fft.ifft(fft)
    if flip:
        pulse = np.flip(pulse)
    # sort time axis
    time_axis = np.sort(time_axis)
    time_axis -= time_axis.min()
    # pulse -= pulse[0]
    return time_axis, pulse

def square_pulse(time, pulse_duration, frequency, pulse_pad=0, phase=0):
    '''
    Generate a square pulse with a carrier frequency:
                  ____________________
    <-pulse_pad->|                    |
    _____________|<--pulse_duration-->|____________
    
    Args: 
        time           : time domain axis of pulse
        pulse_duration : duration of square pulse
        frequency      : carrier frequency of pulse
        pulse_pad      : zero amplitude padding of pulse
    '''
    return np.cos(time*frequency*2*np.pi+phase)*(np.heaviside(time-pulse_pad, 1)-np.heaviside(time-pulse_duration-pulse_pad, 1))

def Vp_from_PdBm(P_dBm, R=50):
    '''
    Convert power in dBm to voltage peak.
    '''
    return np.sqrt(R*10**(P_dBm/10-3))*np.sqrt(2)

def PdBm_from_Vp(Vp, R=50):
    '''
    Convert power in dBm to voltage peak.
    '''
    Vrms = Vp/np.sqrt(2)
    return 10*(np.log10(Vrms**2/R)+3)

def solve_nan(array):
    '''
    Interpolate nan values in array.
    '''
    idxs_nan = np.where(np.isnan(array))[0]
    for i in idxs_nan:
        p1 = np.mean([array[[i-1, i+1]]])
        p2 = np.mean([array[[i-2, i+2]]])
        if p1<p2:
            array[i] = 0
        else:
            array[i] = np.mean([array[[i-1, i+1]]])
    return array

##############################################
# Physics helper functions
##############################################

def PSD_thermal_quantum(f, T):
    '''
    Power spectral density describing thermal and quantum 
    (Johnson-Nyquist) at temperature T in units W/Hz.
    Args:
        f : frequency in Hz
        T : temperature in Kelvin
    '''
    h = constants.h
    kB = constants.Boltzmann
    # Bose-Einstein distribution
    eta = (h*f/(kB*T))/(np.exp(h*f/(kB*T))-1)
    # power spectral density
    PSD = 4*kB*T*eta
    return PSD

def PSD_quantum(f):
    '''
    Power spectral density of quantum (Nyquist) noise in units W/Hz.
    Args:
        T : temperature in Kelvin.
        f : frequency in Hz.
    '''
    h = constants.h
    # power spectral density (single sided)
    PSD = h*f/2 * (np.sign(f)+1)/2
    return PSD

def PSD_thermal_bose(f, T):
    '''
    Power spectral density describing thermal 
    (Johnson) noise at temperature T in units W/Hz.
    Args:
        f : frequency in Hz
        T : temperature in Kelvin
    '''
    h = constants.h
    kB = constants.Boltzmann
    # Bose-Einstein distribution
    eta = (h*f/(kB*T))/(np.exp(h*f/(kB*T))-1)
    # power spectral density
    PSD = 4*kB*T*eta
    return PSD

def effective_josephson_energy(Ej1_norm, Ej2_norm, phi):
    Ej_sum = Ej1_norm+Ej2_norm
    d = (Ej2_norm-Ej1_norm)/Ej_sum
    Ej_eff = Ej_sum*np.cos(np.pi*phi)*np.sqrt(1 + (d*np.tan(np.pi*phi))**2)
    return Ej_eff
    
def josephson_inductance(Ej_norm):
    h = constants.h
    phi_0 = constants.physical_constants['mag. flux quantum'][0]
    Ej = Ej_norm*h
    Lj = phi_0**2/((2*np.pi)**2 * Ej)
    return abs(Lj)

##############################################
# Plotting helper functions
##############################################
SI_PREFIXES = dict(zip(range(-24, 25, 3), 'yzafpnμm kMGTPEZY'))
SI_PREFIXES[0] = ""
SI_UNITS = ',m,s,g,W,J,V,A,F,H,T,Hz,$\Omega$,S,N,C,px,b,B,K,Bar,Vpeak,Vpp,Vp,Vrms,$\Phi_0$,A/s'.split(
    ',')

def SI_prefix_and_scale_factor(val, unit=None):
    """
    Takes in a value and unit and if applicable returns the proper
    scale factor and SI prefix.
    Args:
        val (float) : the value
        unit (str)  : the unit of the value
    returns:
        scale_factor (float) : scale_factor needed to convert value
        unit (str)           : unit including the prefix
    """

    if unit in SI_UNITS:
        try:
            with np.errstate(all="ignore"):
                prefix_power = np.log10(abs(val))//3 * 3
                prefix = SI_PREFIXES[prefix_power]
                # Greek symbols not supported in tex
                if plt.rcParams['text.usetex'] and prefix == 'μ':
                    prefix = r'$\mu$'

            return 10 ** -prefix_power, prefix + unit
        except (KeyError, TypeError):
            pass

    return 1, unit if unit is not None else ""

def quantity_to_string(val, unit, error=None, decimal_place=2):
    scale_factor, prefix = SI_prefix_and_scale_factor(val, unit)
    scaled_val = val*scale_factor
    if error: # write number with uncertainty
        try:
            decimal_place = -int(np.round(np.log10(error*scale_factor)))
            if decimal_place < 0:
                uncert = int(scaled_val%(10**-decimal_place)*(10**(decimal_place+1)))
                scaled_val = (scaled_val//(10**-decimal_place))
                quantity_str = '{:.0f}({}) {}'.format(scaled_val, uncert, prefix)
            elif decimal_place == 0:
                uncert = int(scaled_val%(10**-decimal_place)*(10**(decimal_place+1)))
                scaled_val = (scaled_val//(10**-decimal_place))
                quantity_str = '{:.0f}.({}) {}'.format(scaled_val, uncert, prefix)
            else:
                uncert = int(scaled_val%(10**-decimal_place)*(10**(decimal_place+1)))
                scaled_val = (scaled_val//(10**-decimal_place))*(10**-decimal_place)
                quantity_str = '{:.{}f}({}) {}'.format(scaled_val, decimal_place, uncert, prefix)
        except:
            print('Could not write uncertainty!')
            decimal_place = 0
            quantity_str = '{:.{}f} {}'.format(scaled_val, decimal_place, prefix)
    else:
        quantity_str = '{:.{}g} {}'.format(scaled_val, decimal_place+1, prefix)
    return quantity_str

def set_xlabel(axis, label, unit=None, latexify_ticks=False, **kw):
    """
    Add a unit aware x-label to an axis object.
    Args:
        axis: matplotlib axis object to set label on
        label: the desired label
        unit:  the unit
        **kw : keyword argument to be passed to matplotlib.set_xlabel
    """
    if unit is not None and unit != '':
        if axis.get_xscale() == 'log':
            axis.set_xlabel(label + ' ({})'.format(unit), **kw)
        else:
            xticks = axis.get_xticks()
            scale_factor, unit = SI_prefix_and_scale_factor(
                val=max(abs(xticks)), unit=unit)
            tick_str = '{:.4g}' if not latexify_ticks else r'${:.4g}$'
            formatter = matplotlib.ticker.FuncFormatter(
                lambda x, pos: tick_str.format(x * scale_factor))
            axis.xaxis.set_major_formatter(formatter)
            axis.set_xlabel(label + ' ({})'.format(unit), **kw)
    else:
        axis.set_xlabel(label, **kw)
    return axis

def set_ylabel(axis, label, unit=None, latexify_ticks=False, **kw):
    """
    Add a unit aware y-label to an axis object.
    Args:
        axis: matplotlib axis object to set label on
        label: the desired label
        unit:  the unit
        **kw : keyword argument to be passed to matplotlib.set_ylabel
    """
    if unit is not None and unit != '':
        if axis.get_yscale() == 'log':
            axis.set_xlabel(label + ' ({})'.format(unit), **kw)
        else:
            yticks = axis.get_yticks()
            scale_factor, unit = SI_prefix_and_scale_factor(
                val=max(abs(yticks)), unit=unit)
            tick_str = '{:.6g}' if not latexify_ticks else r'${:.6g}$'
            formatter = matplotlib.ticker.FuncFormatter(
                lambda x, pos: tick_str.format(x * scale_factor))
            axis.yaxis.set_major_formatter(formatter)
            axis.set_ylabel(label + ' ({})'.format(unit), **kw)
    else:
        axis.set_ylabel(label, **kw)
    return axis

def set_cbarlabel(cbar, label, unit=None, **kw):
    """
    Add a unit aware z-label to a colorbar object

    Args:
        cbar: colorbar object to set label on
        label: the desired label
        unit:  the unit
        **kw : keyword argument to be passed to cbar.set_label
    """
    if unit is not None and unit != '':
        zticks = cbar.get_ticks()[1:-1]
        scale_factor, unit = SI_prefix_and_scale_factor(
            val=max(abs(zticks)), unit=unit)
        cbar.set_ticks(zticks)
        cbar.set_ticklabels([f'{v:.3g}' for v in zticks*scale_factor])
        cbar.set_label(label + ' ({})'.format(unit))
    else:
        cbar.set_label(label, **kw)
    return cbar

def move_subplot(ax, dx: float = 0, dy: float = 0, scale_x: float = 1, scale_y: float = 1):
    '''
    Move subplot by dx and dy
    '''
    pos = ax.get_position()
    pos = [pos.x0+dx*pos.width*scale_x, pos.y0+dy*pos.height*scale_y, pos.width*scale_x, pos.height*scale_y]
    ax.set_position(pos)

def plot_line_cmap(x, y, color_percentage: float, ax = None, cmap: str = 'viridis', **kw):
    '''
    Plot a line with single color colored by a colormap.
    '''
    # figure
    if not ax:
        ax = plt.gca()
    else:
        fig = plt.gcf()
    # define colormap
    cmap = matplotlib.colormaps[cmap]
    color = cmap(color_percentage)
    # plot
    ax.plot(x, y, color=color, **kw)

def get_cbar(mappeable=None, ax = None, orientation: str = 'vertical', pos: tuple = None, dims: tuple = None, label: str = None, unit: str = None, **kw):
    '''
    Add a cbar to subplot.
    Args:
        pos  : colorbar position tuple in relative axis units.
        dims : colorbar dimensions tuple (lx, ly) in relative axis units.
    '''
    assert orientation in ['vertical', 'horizontal']
    cmap = kw.get('cmap', None)
    if mappeable is None:
        vmin = kw.get('vmin', 0)
        vmax = kw.get('vmax', 1)
        norm = kw.get('norm', matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
        # mappeable = matplotlib.collections.QuadMesh(np.array([[[0,0],[1,1]]]), norm=norm, cmap=cmap)
        mappeable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()
    cbar_ax = fig.add_subplot(111)
    _pos = ax.get_position()
    if dims is None:
        if orientation == 'vertical':
            dims = (0.05 ,1)
        else:
            dims = (1, 0.05)
    if pos is None:
        if orientation == 'vertical':
            pos = (1.05 ,0)
        else:
            pos = (0, -0.1)
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')
    cbar_ax.set_position([_pos.x0+_pos.width*pos[0], _pos.y0+_pos.height*pos[1], _pos.width*dims[0], _pos.height*dims[1]])
    cbar = fig.colorbar(mappeable, cax=cbar_ax, orientation=orientation)
    if label:
        set_cbarlabel(cbar, label, unit=unit)
    return cbar

def parse_s2p(file_path):
    """
    Parse an S2P file into a dictionary with metadata and S-parameter data.
    Parameters:
        file_path (str): Path to the S2P file.
    Returns:
        dict: A dictionary with the frequency data, S-parameters, and metadata.
    """
    s2p_data = {
        "metadata": [],
        "frequency": [],
        "11": [],
        "12": [],
        "21": [],
        "22": []
    }
    with open(file_path, "r") as file:
        lines = file.readlines()
    for line in lines:
        line = line.strip()
        # Skip comments and empty lines
        if not line or line.startswith("#") or line.startswith("!"):
            if line.startswith("!"):
                # Parse metadata from the header
                s2p_data["metadata"].append(line[1:].strip())
            if line.startswith("#"):
                s2p_data['frq_unit'], s2p_data['param_type'], s2p_data['data_format'], _, s2p_data['char_imp'] = line[1:].split()
            continue
        # Split numerical data
        values = list(map(float, line.split()))
        # Find the key by value
        scale_factor = 10**next((k for k, v in SI_PREFIXES.items() if v == s2p_data['frq_unit'].split('Hz')[0]), 0)
        s2p_data["frequency"].append(values[0]*scale_factor)
        if s2p_data['data_format'].lower() == 'db':
            s2p_data["11"].append(10**(values[1]/20)*np.exp(1j*np.pi*values[2]/180))
            s2p_data["21"].append(10**(values[3]/20)*np.exp(1j*np.pi*values[4]/180))
            s2p_data["12"].append(10**(values[5]/20)*np.exp(1j*np.pi*values[6]/180))
            s2p_data["22"].append(10**(values[7]/20)*np.exp(1j*np.pi*values[8]/180))
        elif s2p_data['data_format'].lower() == 'ma':
            s2p_data["11"].append(values[1]*np.exp(1j*np.pi*values[2]/180))
            s2p_data["21"].append(values[3]*np.exp(1j*np.pi*values[4]/180))
            s2p_data["12"].append(values[5]*np.exp(1j*np.pi*values[6]/180))
            s2p_data["22"].append(values[7]*np.exp(1j*np.pi*values[8]/180))
        elif s2p_data['data_format'].lower() == 'ri':
            s2p_data["11"].append(complex(values[1], values[2]))
            s2p_data["21"].append(complex(values[3], values[4]))
            s2p_data["12"].append(complex(values[5], values[6]))
            s2p_data["22"].append(complex(values[7], values[8]))
    return s2p_data