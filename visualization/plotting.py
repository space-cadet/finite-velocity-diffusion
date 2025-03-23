import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_1d_comparison(
    x: np.ndarray,
    u_finite: np.ndarray,
    u_classical: np.ndarray,
    title: str = "Finite Velocity vs Classical Diffusion",
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[Figure, Axes]:
    """
    Plot a comparison between finite velocity and classical diffusion in 1D.
    
    Parameters:
    -----------
    x : np.ndarray
        Spatial coordinates
    u_finite : np.ndarray
        Solution for finite velocity diffusion
    u_classical : np.ndarray
        Solution for classical diffusion
    title : str
        Title for the plot
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    Tuple[Figure, Axes]
        Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, u_finite, 'b-', label='Finite Velocity')
    ax.plot(x, u_classical, 'r--', label='Classical')
    ax.set_xlabel('Position')
    ax.set_ylabel('Concentration')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    return fig, ax


def plot_2d_heatmap(
    X: np.ndarray,
    Y: np.ndarray,
    u: np.ndarray,
    title: str = "2D Finite Velocity Diffusion",
    figsize: Tuple[int, int] = (10, 8),
    colorbar_label: str = "Concentration"
) -> Tuple[Figure, Axes]:
    """
    Plot a 2D heatmap of the solution.
    
    Parameters:
    -----------
    X, Y : np.ndarray
        Spatial coordinate meshgrids
    u : np.ndarray
        Solution array
    title : str
        Title for the plot
    figsize : Tuple[int, int]
        Figure size
    colorbar_label : str
        Label for the colorbar
        
    Returns:
    --------
    Tuple[Figure, Axes]
        Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(X, Y, u, shading='auto')
    plt.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(title)
    
    return fig, ax


def plot_time_series(
    times: List[float],
    solutions: List[np.ndarray],
    x: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Time Evolution",
    figsize: Tuple[int, int] = (12, 8)
) -> Tuple[Figure, Axes]:
    """
    Plot the time evolution of the solution.
    
    Parameters:
    -----------
    times : List[float]
        List of time points
    solutions : List[np.ndarray]
        List of solution arrays corresponding to each time point
    x : np.ndarray
        Spatial coordinates
    labels : List[str], optional
        Custom labels for each time point
    title : str
        Title for the plot
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    Tuple[Figure, Axes]
        Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is None:
        labels = [f"t = {t:.3f}" for t in times]
    
    for i, (t, u) in enumerate(zip(times, solutions)):
        ax.plot(x, u, label=labels[i])
    
    ax.set_xlabel('Position')
    ax.set_ylabel('Concentration')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    return fig, ax


def plot_parameter_comparison(
    x: np.ndarray,
    solutions: List[np.ndarray],
    parameter_values: List[float],
    parameter_name: str = "D",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> Tuple[Figure, Axes]:
    """
    Plot solutions for different parameter values.
    
    Parameters:
    -----------
    x : np.ndarray
        Spatial coordinates
    solutions : List[np.ndarray]
        List of solution arrays for different parameter values
    parameter_values : List[float]
        Values of the varied parameter
    parameter_name : str
        Name of the parameter being varied
    title : str, optional
        Title for the plot
    figsize : Tuple[int, int]
        Figure size
        
    Returns:
    --------
    Tuple[Figure, Axes]
        Matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if title is None:
        title = f"Comparison for Different {parameter_name} Values"
    
    for i, (param, u) in enumerate(zip(parameter_values, solutions)):
        ax.plot(x, u, label=f"{parameter_name} = {param}")
    
    ax.set_xlabel('Position')
    ax.set_ylabel('Concentration')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    return fig, ax
