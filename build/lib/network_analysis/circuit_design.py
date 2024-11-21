import matplotlib.pyplot as plt
import numpy as np


def draw_fluxonium(x1, y1, mirror=False, highlight_capacitor=False, ground_caps=True, **kwargs):
    '''
    Draws a fluxonium circuit.
    '''
    # get current axis
    ax = plt.gca()
    # Plot settings
    _plot_args = dict(color='k', lw=kwargs.get('lw', 4), solid_capstyle='round', clip_on=False)
    # Coordinates of loop
    _x = np.array([ -.4, -.5, -.5, +.5, +.5, +.4])
    _y = np.array([ -.5, -.5, +.5, +.5, -.5, -.5])
    # Fluxonium
    if mirror:
        _y *= -1
    ax.plot(x1+_x, y1+_y, **{**_plot_args, 'color':'#e0e1dd' if highlight_capacitor else 'k'})
    # Inductance
    _phase = np.linspace(0, 2*np.pi, 1001)
    _y_l = 0.1*np.sin(_phase*2.5)*(2*mirror-1)+_y[0]
    _x_l = 0.1*np.cos(_phase*2.5)-_phase/(2*np.pi)*0.6-.3
    ax.plot(x1+_x_l+.6, y1+_y_l, **{**_plot_args, 'color':'#e0e1dd' if highlight_capacitor else 'k'})
    # Josephson junction
    ax.plot([x1-.1, x1+.1], [y1-.1+_y[2], y1+.1+_y[2]], **{**_plot_args, 'color':'#e0e1dd' if highlight_capacitor else 'k'}) # Junction
    ax.plot([x1-.1, x1+.1], [y1+.1+_y[2], y1-.1+_y[2]], **{**_plot_args, 'color':'#e0e1dd' if highlight_capacitor else 'k'}) #
    # Capacitor
    ax.plot([x1-.5, x1-.1], [y1, y1], **_plot_args)         #
    ax.plot([x1+.5, x1+.1], [y1, y1], **_plot_args)         #
    ax.plot([x1-.1, x1-.1], [y1+.15, y1-.15], **_plot_args) # Capacitor
    ax.plot([x1+.1, x1+.1], [y1+.15, y1-.15], **_plot_args) #
    # Plot grounding caps at nodes
    plot_ground_capacitance(x1+.5, y1-.5, l=.5, horizontal=False)
    plot_ground_capacitance(x1-.5, y1-.5, l=.5, horizontal=False)
    ax.plot([x1-.5, x1-.5], [y1, y1-.5], **{**_plot_args, 'solid_capstyle':None})
    ax.plot([x1+.5, x1+.5], [y1, y1-.5], **{**_plot_args, 'solid_capstyle':None})


def plot_ground_capacitance(x, y, l, l_cap=.1, cap_dist=.15, horizontal=True, **kwargs):
    '''
    Draw ground capacitance
    '''
    # get current axis
    ax = plt.gca()
    # Plot settings
    _plot_args = dict(color='k', lw=kwargs.get('lw', 4), solid_capstyle='round', clip_on=False)
    # Capacitor direction
    if horizontal:
        ax.plot([x, x-l/2+cap_dist/2, x-l/2+cap_dist/2, x-l/2+cap_dist/2], [y, y, y+l_cap, y-l_cap], **_plot_args, zorder=-1)
        ax.plot([x-l*4/3, x-l*4/3, x-l*4/3, x-l*7/3], [ y+l_cap, y-l_cap, y, y], **_plot_args, zorder=-1)
        # ground
        ax.plot([x-l-np.sign(l)*.00, x-l-np.sign(l)*.00], [ y+.10, y-.10], **_plot_args, zorder=-1)
        ax.plot([x-l-np.sign(l)*.05, x-l-np.sign(l)*.05], [ y+.05, y-.05], **_plot_args, zorder=-1)
        ax.plot([x-l-np.sign(l)*.10, x-l-np.sign(l)*.10], [ y+.01, y-.01], **_plot_args, zorder=-1)
    # Vertical capacitor
    if not horizontal:
        ax.plot([x, x, x+l_cap, x-l_cap], [y, y-l/2+cap_dist/2, y-l/2+cap_dist/2, y-l/2+cap_dist/2], **_plot_args, zorder=-1)
        ax.plot([ x+l_cap, x-l_cap, x, x], [y-l/2-cap_dist/2, y-l/2-cap_dist/2, y-l/2-cap_dist/2, y-l], **_plot_args, zorder=-1)
        # ground
        ax.plot([ x+.10, x-.10], [y-l-np.sign(l)*.00, y-l-np.sign(l)*.00], **_plot_args, zorder=-1)
        ax.plot([ x+.05, x-.05], [y-l-np.sign(l)*.05, y-l-np.sign(l)*.05], **_plot_args, zorder=-1)
        ax.plot([ x+.01, x-.01], [y-l-np.sign(l)*.10, y-l-np.sign(l)*.10], **_plot_args, zorder=-1)


def plot_capacitor(x0, y0, x1, y1, l_cap=.1, cap_dist=.15, **kwargs):
    '''
    Plot capacitor between (x0, y0) and (x1, y1)
    '''
    # get current axis
    ax = plt.gca()
    # Plot settings
    _plot_args = dict(color='k', lw=kwargs.get('lw', 4), solid_capstyle='round', clip_on=False)
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


def plot_inductor(x0, y0, x1, y1, w_ind=.1, **kwargs):
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
    x = np.linspace(0, l-w_ind, 201) + ((np.cos(theta)+1)/2)*w_ind
    y = -w_ind*np.sin(theta)/2
    # Plot rotated inductor
    ax.plot((x*np.cos(phi)-y*np.sin(phi))+x0, (y*np.cos(phi)+x*np.sin(phi))+y0, **_plot_args)


def plot_transmission_line(x, y, l, horizontal=True, radius=.1, **kwargs):
    '''
    Plot transmission line.
    '''
    # get current axis
    ax = plt.gca()
    # Plot settings
    _plot_args = dict(color='k', lw=kwargs.get('lw', 4), solid_capstyle='round', clip_on=False)
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


def plot_ground(x, y, l, horizontal=False, **kwargs):
    '''
    Draw ground conection
    '''
    # get current axis
    ax = plt.gca()
    # Plot settings
    _plot_args = dict(color='k', lw=kwargs.get('lw', 4), solid_capstyle='round', clip_on=False)
    # Capacitor direction
    if horizontal:
        ax.plot([x, x+l], [y, y], **_plot_args, zorder=-1)
        # ground
        ax.plot([x-l-np.sign(l)*.00, x-l-np.sign(l)*.00], [ y+.10, y-.10], **_plot_args, zorder=-1)
        ax.plot([x-l-np.sign(l)*.05, x-l-np.sign(l)*.05], [ y+.05, y-.05], **_plot_args, zorder=-1)
        ax.plot([x-l-np.sign(l)*.10, x-l-np.sign(l)*.10], [ y+.01, y-.01], **_plot_args, zorder=-1)
    # Vertical capacitor
    if not horizontal:
        ax.plot([x, x], [y, y-l], **_plot_args, zorder=-1)
        # ground
        ax.plot([ x+.10, x-.10], [y-l-np.sign(l)*.00, y-l-np.sign(l)*.00], **_plot_args, zorder=-1)
        ax.plot([ x+.05, x-.05], [y-l-np.sign(l)*.05, y-l-np.sign(l)*.05], **_plot_args, zorder=-1)
        ax.plot([ x+.01, x-.01], [y-l-np.sign(l)*.10, y-l-np.sign(l)*.10], **_plot_args, zorder=-1)


def plot_drive(x0, y0, x1, y1, amp=0.2, **kwargs):
    '''
    Draw microwave drive
    '''
    # get current axis
    ax = plt.gca()
    # Plot settings
    _plot_args = dict(color='k', lw=kwargs.get('lw', 4), solid_capstyle='round', clip_on=False)
    # compute displacement
    dx = x1-x0
    dy = y1-y0
    theta = np.arctan2(dy, dx)
    L = np.sqrt(dx**2 + dy**2)/2

    phi = np.linspace(0, 2*np.pi, 101)*10

    x = np.linspace(0, L, len(phi))
    y = np.sin(phi)

    ax.plot(x, y, **_plot_args)


