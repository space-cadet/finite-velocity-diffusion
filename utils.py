import numpy as np
from typing import Tuple, Union, Optional

def gaussian_initial_condition(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    amplitude: float = 1.0,
    center: Union[float, Tuple[float, float]] = 0.0,
    sigma: Union[float, Tuple[float, float]] = 1.0
) -> np.ndarray:
    """
    Create a Gaussian initial condition for the simulation.
    
    Parameters:
    -----------
    x : np.ndarray
        x-coordinates
    y : np.ndarray, optional
        y-coordinates for 2D case
    amplitude : float
        Amplitude of the Gaussian
    center : float or tuple of float
        Center position(s) of the Gaussian
    sigma : float or tuple of float
        Standard deviation(s) of the Gaussian
        
    Returns:
    --------
    np.ndarray
        Initial condition array
    """
    if y is None:
        # 1D case
        return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
    else:
        # 2D case
        if isinstance(center, (int, float)):
            center = (center, center)
        if isinstance(sigma, (int, float)):
            sigma = (sigma, sigma)
            
        X, Y = np.meshgrid(x, y)
        return amplitude * np.exp(
            -((X - center[0])**2 / (2 * sigma[0]**2) + 
              (Y - center[1])**2 / (2 * sigma[1]**2))
        )

def classical_diffusion_solution(
    x: np.ndarray,
    t: float,
    D: float,
    initial_amplitude: float = 1.0,
    initial_center: float = 0.0,
    initial_sigma: float = 1.0
) -> np.ndarray:
    """
    Compute the analytical solution of the classical diffusion equation
    for a Gaussian initial condition.
    
    Parameters:
    -----------
    x : np.ndarray
        x-coordinates
    t : float
        Time
    D : float
        Diffusion coefficient
    initial_amplitude : float
        Initial amplitude of the Gaussian
    initial_center : float
        Initial center position of the Gaussian
    initial_sigma : float
        Initial standard deviation of the Gaussian
        
    Returns:
    --------
    np.ndarray
        Solution at time t
    """
    sigma_t = np.sqrt(initial_sigma**2 + 2*D*t)
    return (initial_amplitude * initial_sigma / sigma_t) * np.exp(
        -(x - initial_center)**2 / (2 * sigma_t**2)
    )

def calculate_propagation_speed(D: float, tau: float) -> float:
    """
    Calculate the finite propagation speed for telegrapher's equation.
    
    In the finite velocity diffusion (telegrapher's equation), the
    propagation speed is given by sqrt(D/tau).
    
    Parameters:
    -----------
    D : float
        Diffusion coefficient
    tau : float
        Relaxation time
        
    Returns:
    --------
    float
        Propagation speed
    """
    return np.sqrt(D / tau)

# Physical constants and default parameters
DEFAULT_PARAMETERS = {
    'D': 1.0,        # Diffusion coefficient
    'tau': 1.0,      # Relaxation time
    'nx': 100,       # Number of x points
    'ny': 100,       # Number of y points
    'dx': 0.1,       # x step size
    'dy': 0.1,       # y step size
    'dt': 0.001,     # Time step size
    'x_min': 0.0,    # Minimum x value
    'x_max': 10.0,   # Maximum x value
    'y_min': 0.0,    # Minimum y value
    'y_max': 10.0,   # Maximum y value
} 