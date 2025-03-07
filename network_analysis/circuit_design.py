import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon


def plot_capacitor(x0, y0, x1, y1, l_cap=.1, cap_dist=.15, **kwargs):
    '''
    Plot capacitor between (x0, y0) and (x1, y1)
    '''
    # get current axis
    ax = plt.gca()
    # Plot settings
    _plot_args = dict(color=kwargs.get('color', 'k'), lw=kwargs.get('lw', 4), solid_capstyle='round', clip_on=False)
    _plot_args = {**_plot_args,**kwargs}
    # compute displacement
    dx = x1-x0
    dy = y1-y0
    theta = np.arctan2(dy, dx)
    L = np.sqrt(dx**2 + dy**2)/2
    # Connections
    ax.plot([x0, x0+(L-cap_dist/2)*np.cos(theta)], [y0, y0+(L-cap_dist/2)*np.sin(theta)], **_plot_args)
    ax.plot([x1, x1-(L-cap_dist/2)*np.cos(theta)], [y1, y1-(L-cap_dist/2)*np.sin(theta)], **_plot_args)
    # Capacitor
    ax.plot([x0+(L-cap_dist/2)*np.cos(theta)+l_cap*np.sin(theta), x0+(L-cap_dist/2)*np.cos(theta)-l_cap*np.sin(theta)],
            [y0+(L-cap_dist/2)*np.sin(theta)-l_cap*np.cos(theta), y0+(L-cap_dist/2)*np.sin(theta)+l_cap*np.cos(theta)], **_plot_args)
    ax.plot([x1-(L-cap_dist/2)*np.cos(theta)-l_cap*np.sin(theta), x1-(L-cap_dist/2)*np.cos(theta)+l_cap*np.sin(theta)],
            [y1-(L-cap_dist/2)*np.sin(theta)+l_cap*np.cos(theta), y1-(L-cap_dist/2)*np.sin(theta)-l_cap*np.cos(theta)], **_plot_args)

def plot_inductor(x0, y0, x1, y1, w_ind=.05, lpad=0, **kwargs):
    '''
    Plot inductor coil between (x0, y0) and (x1, y1).
    Args:
        w_ind : Width of inductor coil
    '''
    # get current axis
    ax = plt.gca()
    # Plot settings
    _plot_args = dict(color=kwargs.get('color', 'k'), lw=kwargs.get('lw', 4), solid_capstyle='round', clip_on=False)
    # displacement
    dx = x1-x0
    dy = y1-y0
    l = np.sqrt(dx**2 + dy**2)
    phi = np.arctan2(dy, dx)
    # horizontal inductor
    theta = np.linspace(np.pi, 6*np.pi, 201)
    x = np.linspace(lpad, l-w_ind-lpad, 201) + ((np.cos(theta)+1)/2)*w_ind
    y = -2*w_ind*np.sin(theta)/2
    x = np.array([0]+list(x)+[l])
    y = np.array([0]+list(y)+[0])
    X = (x*np.cos(phi)-y*np.sin(phi))+x0
    Y = (y*np.cos(phi)+x*np.sin(phi))+y0
    # Plot rotated inductor
    ax.plot(X, Y, **_plot_args)

def plot_resistor(x0, y0, x1, y1, w_ind=.05, lpad=0, **kwargs):
    '''
    Plot inductor coil between (x0, y0) and (x1, y1).
    Args:
        w_ind : Width of inductor coil
    '''
    # get current axis
    ax = plt.gca()
    # Plot settings
    _plot_args = dict(color=kwargs.get('color', 'k'), lw=kwargs.get('lw', 4), solid_capstyle='round', clip_on=False)
    # displacement
    dx = x1-x0
    dy = y1-y0
    l = np.sqrt(dx**2 + dy**2)
    phi = np.arctan2(dy, dx)
    # horizontal inductor
    x = np.linspace(lpad, l-lpad, 7)
    y = w_ind*np.resize([1,-1], len(x))/2
    y[0],y[-1] = 0,0
    x = np.array([0]+list(x)+[l])
    y = np.array([0]+list(y)+[0])
    X = (x*np.cos(phi)-y*np.sin(phi))+x0
    Y = (y*np.cos(phi)+x*np.sin(phi))+y0
    # Plot rotated inductor
    ax.plot(X, Y, **_plot_args)

def plot_attenuator(x0, y0, h=.1, l=.3, attn_dB=None, **kwargs):
    '''
    Plot attenuator.
    '''
    # get current axis
    ax = plt.gca()
    # Plot settings
    _plot_args = dict(boxstyle="Round, pad=0.1", 
                      zorder=3,
                      ec='k', fc=kwargs.get('color', '#607D8B'),
                      lw=kwargs.get('lw', 4), clip_on=False)
    # displacement
    rectangle = FancyBboxPatch((x0-l/2, y0-h/2), l, h, **_plot_args)
    # Plot rotated inductor
    ax.add_patch(rectangle)
    # if attn_dB:
    ax.text(x0, y0, f'-{attn_dB:.0f} dB', va='center', ha='center', zorder=3, size=12, color='w')

def plot_amplifier(x0, y0, h=.1, l=.3, gain_dB=None, **kwargs):
    '''
    Plot amplifier
    '''
    # get current axis
    ax = plt.gca()
    # Plot settings
    _plot_args = dict(joinstyle='round', zorder=3,
                      ec='k', fc=kwargs.get('color', '#607D8B'),
                      lw=kwargs.get('lw', 4), clip_on=False)
    # displacement
    triangle = Polygon([(x0-l/2, y0-h/2), (x0-l/2, y0+h/2), (x0+l/2, y0)], **_plot_args)
    # Plot rotated inductor
    ax.add_patch(triangle)
    # # if gain_dB:
    ax.text(x0-l/10, y0, f'{gain_dB} dB', va='center', ha='center', zorder=3, size=9, color='w')

def plot_low_pass(x0, y0, h=.1, l=.35, name=None, **kwargs):
    '''
    Plot low pass filter.
    '''
    # get current axis
    ax = plt.gca()
    # Plot settings
    _plot_args = dict(boxstyle="Round, pad=0.1", 
                      zorder=3,
                      ec='k', fc=kwargs.get('color', '#607D8B'),
                      lw=kwargs.get('lw', 4), clip_on=False)
    # displacement
    rectangle = FancyBboxPatch((x0-l/2, y0-h/2), l, h, **_plot_args)
    # Plot rotated inductor
    ax.add_patch(rectangle)
    # plot low pass sign
    _plot_args = dict(color=kwargs.get('color', 'w'), lw=kwargs.get('lw', 2), solid_capstyle='round', clip_on=False, zorder=3)
    x = [x0-l*.30, x0+l*.10, x0+l*.30]
    y = [y0+h*.35, y0+h*.35, y0-h*.05]
    ax.plot(x, y, **_plot_args)
    ax.text(x0, y0-h*.43, name, va='center', ha='center', zorder=3, size=8, color='w')

def plot_high_pass(x0, y0, h=.1, l=.35, name=None, **kwargs):
    '''
    Plot high pass filter.
    '''
    # get current axis
    ax = plt.gca()
    # Plot settings
    _plot_args = dict(boxstyle="Round, pad=0.1", 
                      zorder=3,
                      ec='k', fc=kwargs.get('color', '#607D8B'),
                      lw=kwargs.get('lw', 4), clip_on=False)
    # displacement
    rectangle = FancyBboxPatch((x0-l/2, y0-h/2), l, h, **_plot_args)
    # Plot rotated inductor
    ax.add_patch(rectangle)
    # plot low pass sign
    _plot_args = dict(color=kwargs.get('color', 'w'), lw=kwargs.get('lw', 2), solid_capstyle='round', clip_on=False, zorder=3)
    x = [x0-l*.30, x0-l*.10, x0+l*.30]
    y = [y0-h*.05, y0+h*.35, y0+h*.35]
    ax.plot(x, y, **_plot_args)
    ax.text(x0, y0-h*.43, name, va='center', ha='center', zorder=3, size=8, color='w')

def plot_transmission_line(x, y, l, horizontal=True, radius=.1, **kwargs):
    '''
    Plot transmission line.
    '''
    # get current axis
    ax = plt.gca()
    # Plot settings
    _plot_args = dict(color='k', lw=kwargs.get('lw', 4), solid_capstyle='round', clip_on=False)
    _plot_args = {**_plot_args, **kwargs}
    # horizontal line
    if horizontal:
        theta = np.linspace(0, 2*np.pi, 101)
        # elipsis
        ax.plot(radius*(np.cos(theta)/2+1)+x, radius*np.sin(theta)+y, **_plot_args)
        # half elipsis
        ax.plot(radius*(np.sin(theta/2)/2-1)+x+l, radius*np.cos(theta/2)+y, **_plot_args)
        # conection
        ax.plot([x+radius, x+l-radius], [y+radius, y+radius], **_plot_args)
        ax.plot([x+radius, x+l-radius], [y-radius, y-radius], **_plot_args)
        # input/output
        ax.plot([x, x+radius], [y, y], **_plot_args)
        ax.plot([x+l-radius/2, x+l], [y, y], **_plot_args)
    else:
        theta = np.linspace(0, 2*np.pi, 101)
        # elipsis
        ax.plot(radius*np.cos(theta)+x, radius*(np.sin(theta)/2-1)+y, **_plot_args)
        # half elipsis
        ax.plot(radius*np.cos(theta/2)+x, radius*(1-np.sin(theta/2)/2)+y-l, **_plot_args)
        # # conection
        ax.plot([x+radius, x+radius], [y-radius, y+radius-l], **_plot_args)
        ax.plot([x-radius, x-radius], [y-radius, y+radius-l], **_plot_args)
        # # input/output
        ax.plot([x, x], [y, y-radius], **_plot_args)
        ax.plot([x, x], [y-l+radius/2, y-l], **_plot_args)

def plot_ground(x, y, l, lpad=0.1, horizontal=False, **kwargs):
    '''
    Draw ground conection
    '''
    # get current axis
    ax = plt.gca()
    # Plot settings
    _plot_args = dict(color=kwargs.get('color', 'k'), lw=kwargs.get('lw', 4), solid_capstyle='round', clip_on=False)
    # Capacitor direction
    if horizontal:
        ax.plot([x, x+l], [y, y], **_plot_args, zorder=-1)
        # ground
        ax.plot([x-l-np.sign(l)*.00, x-l-np.sign(l)*.00], [ y+lpad, y-lpad], **_plot_args, zorder=-1)
        ax.plot([x-l-np.sign(l)*lpad/2, x-l-np.sign(l)*lpad/2], [ y+lpad/2, y-lpad/2], **_plot_args, zorder=-1)
        ax.plot([x-l-np.sign(l)*lpad, x-l-np.sign(l)*lpad], [ y+lpad/8, y-lpad/8], **_plot_args, zorder=-1)
    # Vertical capacitor
    if not horizontal:
        ax.plot([x, x], [y, y-l], **_plot_args, zorder=-1)
        # ground
        ax.plot([ x+lpad, x-lpad], [y-l-np.sign(l)*.00, y-l-np.sign(l)*.00], **_plot_args, zorder=-1)
        ax.plot([ x+lpad/2, x-lpad/2], [y-l-np.sign(l)*lpad/2, y-l-np.sign(l)*lpad/2], **_plot_args, zorder=-1)
        ax.plot([ x+lpad/8, x-lpad/8], [y-l-np.sign(l)*lpad, y-l-np.sign(l)*lpad], **_plot_args, zorder=-1)

def plot_drive(x0, y0, x1, y1, amp=0.2, periods=8, **kwargs):
    '''
    Draw microwave drive
    '''
    # get current axis
    ax = plt.gca()
    # Plot settings
    _plot_args = dict(color=kwargs.get('color', 'k'), lw=kwargs.get('lw', 4), solid_capstyle='round', clip_on=False)
    # compute displacement
    dx = x1-x0
    dy = y1-y0
    theta = np.arctan2(dy, dx)
    L = np.sqrt(dx**2 + dy**2)/2
    # midpoint
    x_avg, y_avg = (x0+x1)/2, (y0+y1)/2
    # waveform
    phi = np.linspace(0, 2*np.pi, 101)*periods
    x = np.linspace(-L, L, len(phi))
    y = np.exp(-(x/(L/2))**2)  *  np.sin(phi)*amp
    # plot
    ax.plot(x*np.cos(theta)-y*np.sin(theta) + x_avg, y*np.cos(theta)+x*np.sin(theta) + y_avg, **_plot_args)
    # arrow
    _plot_args = dict(color=kwargs.get('color', 'k'), lw=0, clip_on=False)
    l_arrow = .1
    pts = np.array([[x1+l_arrow*np.cos(theta), x1+l_arrow/2*np.cos(theta+np.pi/2), x1, x1+l_arrow/2*np.cos(theta-np.pi/2)],
                    [y1+l_arrow*np.sin(theta), y1+l_arrow/2*np.sin(theta+np.pi/2), y1, y1+l_arrow/2*np.sin(theta-np.pi/2)]]).T
    ax.add_patch(Polygon(pts, **_plot_args))




