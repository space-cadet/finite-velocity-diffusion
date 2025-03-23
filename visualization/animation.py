import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional, Callable
import matplotlib.animation as animation


def create_1d_animation(
    x: np.ndarray,
    solutions: List[np.ndarray],
    times: List[float],
    title: str = "Finite Velocity Diffusion Evolution",
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (10, 6),
    interval: int = 200,
    save_path: Optional[str] = None
) -> FuncAnimation:
    """
    Create an animation of 1D solution evolution.
    
    Parameters:
    -----------
    x : np.ndarray
        Spatial coordinates
    solutions : List[np.ndarray]
        List of solution arrays at different times
    times : List[float]
        List of time points corresponding to solutions
    title : str
        Animation title
    ylim : Tuple[float, float], optional
        y-axis limits
    figsize : Tuple[int, int]
        Figure size
    interval : int
        Time between frames in milliseconds
    save_path : str, optional
        Path to save the animation (if None, won't save)
        
    Returns:
    --------
    FuncAnimation
        Matplotlib animation object
    """
    fig, ax = plt.subplots(figsize=figsize)
    line, = ax.plot([], [], 'b-', lw=2)
    
    if ylim is None:
        ymin = min(np.min(sol) for sol in solutions) * 1.1
        ymax = max(np.max(sol) for sol in solutions) * 1.1
        ylim = (ymin, ymax)
    
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(ylim)
    ax.set_xlabel('Position')
    ax.set_ylabel('Concentration')
    ax.grid(True)
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    def animate(i):
        line.set_data(x, solutions[i])
        time_text.set_text(f'Time: {times[i]:.3f}')
        return line, time_text
    
    ani = FuncAnimation(fig, animate, frames=len(solutions),
                        init_func=init, blit=True, interval=interval)
    
    if save_path:
        writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(save_path, writer=writer)
    
    return ani


def create_2d_animation(
    X: np.ndarray,
    Y: np.ndarray,
    solutions: List[np.ndarray],
    times: List[float],
    title: str = "2D Finite Velocity Diffusion Evolution",
    figsize: Tuple[int, int] = (10, 8),
    interval: int = 200,
    colormap: str = 'viridis',
    save_path: Optional[str] = None
) -> FuncAnimation:
    """
    Create an animation of 2D solution evolution.
    
    Parameters:
    -----------
    X, Y : np.ndarray
        Meshgrid of spatial coordinates
    solutions : List[np.ndarray]
        List of 2D solution arrays at different times
    times : List[float]
        List of time points corresponding to solutions
    title : str
        Animation title
    figsize : Tuple[int, int]
        Figure size
    interval : int
        Time between frames in milliseconds
    colormap : str
        Matplotlib colormap name
    save_path : str, optional
        Path to save the animation (if None, won't save)
        
    Returns:
    --------
    FuncAnimation
        Matplotlib animation object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Find global min and max for consistent colorbar
    vmin = min(np.min(sol) for sol in solutions)
    vmax = max(np.max(sol) for sol in solutions)
    
    # Initial plot
    im = ax.pcolormesh(X, Y, solutions[0], vmin=vmin, vmax=vmax, 
                        shading='auto', cmap=colormap)
    
    fig.colorbar(im, ax=ax, label='Concentration')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(title)
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                        color='white', fontweight='bold')
    
    def animate(i):
        im.set_array(solutions[i].ravel())
        time_text.set_text(f'Time: {times[i]:.3f}')
        return [im, time_text]
    
    ani = FuncAnimation(fig, animate, frames=len(solutions),
                        interval=interval, blit=False)
    
    if save_path:
        writer = animation.FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(save_path, writer=writer)
    
    return ani
