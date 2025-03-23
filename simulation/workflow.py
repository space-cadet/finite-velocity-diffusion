"""
Simulation workflow functions for the finite velocity diffusion solver.

This module provides high-level functions to run simulations with different
parameters and configurations, separating the simulation logic from the UI.
"""

import numpy as np
from typing import Dict, Tuple, List, Callable, Optional, Union, Any
from dataclasses import dataclass

from solver.one_dimensional import FiniteVelocityDiffusion1D
from solver.two_dimensional import FiniteVelocityDiffusion2D
from utils import (
    gaussian_initial_condition,
    classical_diffusion_solution,
    calculate_propagation_speed,
    DEFAULT_PARAMETERS
)


@dataclass
class SimulationConfig:
    """Configuration for a finite velocity diffusion simulation."""
    # Physical parameters
    D: float = DEFAULT_PARAMETERS['D']  # Diffusion coefficient
    tau: float = DEFAULT_PARAMETERS['tau']  # Relaxation time
    
    # Domain parameters
    nx: int = DEFAULT_PARAMETERS['nx']  # Number of x grid points
    ny: int = DEFAULT_PARAMETERS['ny']  # Number of y grid points
    dx: float = DEFAULT_PARAMETERS['dx']  # x step size
    dy: float = DEFAULT_PARAMETERS['dy']  # y step size
    x_min: float = DEFAULT_PARAMETERS['x_min']  # Minimum x value
    x_max: float = DEFAULT_PARAMETERS['x_max']  # Maximum x value
    y_min: float = DEFAULT_PARAMETERS['y_min']  # Minimum y value
    y_max: float = DEFAULT_PARAMETERS['y_max']  # Maximum y value
    
    # Time parameters
    dt: float = DEFAULT_PARAMETERS['dt']  # Time step size
    num_steps: int = 100  # Number of time steps
    
    # Initial condition parameters
    amplitude: float = 1.0  # Initial amplitude
    sigma: float = 1.0  # Initial standard deviation
    
    # Visualization parameters
    show_evolution: bool = False  # Whether to show time evolution
    evolution_steps: int = 20  # Number of time points to show
    
    def validate(self) -> None:
        """
        Validate the simulation configuration parameters.
        
        Raises:
            ValueError: If any parameters are invalid or the configuration is unstable.
        """
        # Check physical parameters
        if self.D <= 0:
            raise ValueError("Diffusion coefficient D must be positive")
        if self.tau <= 0:
            raise ValueError("Relaxation time tau must be positive")
        
        # Check domain parameters
        if self.nx <= 1 or self.ny <= 1:
            raise ValueError("Number of grid points must be greater than 1")
        if self.dx <= 0 or self.dy <= 0:
            raise ValueError("Spatial step sizes must be positive")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Domain max values must be greater than min values")
        
        # Check time parameters
        if self.dt <= 0:
            raise ValueError("Time step must be positive")
        if self.num_steps <= 0:
            raise ValueError("Number of time steps must be positive")
        
        # Check Courant number for stability
        courant_x = np.sqrt(self.D * self.dt / (self.tau * self.dx**2))
        courant_y = np.sqrt(self.D * self.dt / (self.tau * self.dy**2))
        
        if courant_x > 1 or courant_y > 1:
            raise ValueError(
                f"Unstable configuration: Courant numbers ({courant_x:.2f}, {courant_y:.2f}) > 1. "
                "Reduce dt or increase dx/dy."
            )


@dataclass
class SimulationResult:
    """Results from a finite velocity diffusion simulation."""
    # Spatial grid
    x: np.ndarray  # x coordinates
    y: Optional[np.ndarray] = None  # y coordinates (for 2D)
    X: Optional[np.ndarray] = None  # X meshgrid (for 2D)
    Y: Optional[np.ndarray] = None  # Y meshgrid (for 2D)
    
    # Solution
    u: np.ndarray  # Final solution
    u_classical: Optional[np.ndarray] = None  # Classical diffusion solution (if computed)
    
    # Time evolution
    times: Optional[List[float]] = None  # Time points
    solutions: Optional[List[np.ndarray]] = None  # Solutions at each time point
    
    # Metadata
    config: SimulationConfig  # Configuration used for the simulation
    dimension: str  # "1D" or "2D"
    final_time: float  # Final simulation time
    
    @property
    def propagation_speed(self) -> float:
        """Calculate the finite propagation speed for the simulation."""
        return calculate_propagation_speed(self.config.D, self.config.tau)
    
    @property
    def courant_number(self) -> Union[float, Tuple[float, float]]:
        """Calculate the Courant number(s) for the simulation."""
        if self.dimension == "1D":
            return np.sqrt(self.config.D * self.config.dt / (self.config.tau * self.config.dx**2))
        else:
            courant_x = np.sqrt(self.config.D * self.config.dt / (self.config.tau * self.config.dx**2))
            courant_y = np.sqrt(self.config.D * self.config.dt / (self.config.tau * self.config.dy**2))
            return (courant_x, courant_y)


def run_1d_simulation(
    config: SimulationConfig,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> SimulationResult:
    """
    Run a 1D finite velocity diffusion simulation.
    
    Parameters:
        config: SimulationConfig object containing simulation parameters
        progress_callback: Optional callback function that takes (progress_fraction, status_message)
                          Used to update progress in UI
    
    Returns:
        SimulationResult object containing the simulation results
    """
    # Initialize the solver
    solver = FiniteVelocityDiffusion1D(
        nx=config.nx,
        dx=config.dx,
        dt=config.dt,
        D=config.D,
        tau=config.tau,
        x_min=config.x_min,
        x_max=config.x_max
    )
    
    # Set initial condition
    initial_center = (solver.x[-1] + solver.x[0]) / 2  # Center of domain
    initial_condition = gaussian_initial_condition(
        solver.x,
        amplitude=config.amplitude,
        center=initial_center,
        sigma=config.sigma
    )
    solver.set_initial_condition(initial_condition)
    
    # Storage for time evolution
    x = solver.x
    times = []
    solutions = []
    
    if config.show_evolution:
        # Calculate step size for evolution plots/animation
        step_size = max(1, config.num_steps // config.evolution_steps)
        
        # Run solver and capture intermediate results
        for i in range(0, config.num_steps + 1, step_size):
            if i == 0:
                # Initial condition
                times.append(0)
                solutions.append(solver.u.copy())
            else:
                # Solve for steps
                solver.solve(step_size)
                times.append(i * config.dt)
                solutions.append(solver.u.copy())
                
                # Update progress
                if progress_callback:
                    progress = min(1.0, i / config.num_steps)
                    progress_callback(progress, f"Computing solution: {int(progress * 100)}% complete")
        
        # Final result is the last solution
        u = solutions[-1]
    else:
        # Solve normally
        if progress_callback:
            progress_callback(0.0, "Computing solution...")
        x, u = solver.solve(config.num_steps)
        if progress_callback:
            progress_callback(1.0, "Solution complete.")
    
    # Compute classical diffusion solution for comparison
    final_time = config.num_steps * config.dt
    u_classical = classical_diffusion_solution(
        x,
        final_time,
        config.D,
        initial_amplitude=config.amplitude,
        initial_center=initial_center,
        initial_sigma=config.sigma
    )
    
    # Create and return result object
    result = SimulationResult(
        x=x,
        u=u,
        u_classical=u_classical,
        times=times if config.show_evolution else None,
        solutions=solutions if config.show_evolution else None,
        config=config,
        dimension="1D",
        final_time=final_time
    )
    
    return result


def run_2d_simulation(
    config: SimulationConfig,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> SimulationResult:
    """
    Run a 2D finite velocity diffusion simulation.
    
    Parameters:
        config: SimulationConfig object containing simulation parameters
        progress_callback: Optional callback function that takes (progress_fraction, status_message)
                          Used to update progress in UI
    
    Returns:
        SimulationResult object containing the simulation results
    """
    # Initialize the solver
    solver = FiniteVelocityDiffusion2D(
        nx=config.nx,
        ny=config.ny,
        dx=config.dx,
        dy=config.dy,
        dt=config.dt,
        D=config.D,
        tau=config.tau,
        x_min=config.x_min,
        x_max=config.x_max,
        y_min=config.y_min,
        y_max=config.y_max
    )
    
    # Set initial condition
    initial_center = (
        (solver.x[-1] + solver.x[0]) / 2,  # Center of x domain
        (solver.y[-1] + solver.y[0]) / 2   # Center of y domain
    )
    initial_condition = gaussian_initial_condition(
        solver.x,
        solver.y,
        amplitude=config.amplitude,
        center=initial_center,
        sigma=(config.sigma, config.sigma)
    )
    solver.set_initial_condition(initial_condition)
    
    # Storage for time evolution
    X, Y = solver.X, solver.Y
    times = []
    solutions = []
    
    if config.show_evolution:
        # Calculate step size for evolution plots/animation
        step_size = max(1, config.num_steps // config.evolution_steps)
        
        # Run solver and capture intermediate results
        for i in range(0, config.num_steps + 1, step_size):
            if i == 0:
                # Initial condition
                times.append(0)
                solutions.append(solver.u.copy())
            else:
                # Solve for steps
                solver.solve(step_size)
                times.append(i * config.dt)
                solutions.append(solver.u.copy())
                
                # Update progress
                if progress_callback:
                    progress = min(1.0, i / config.num_steps)
                    progress_callback(progress, f"Computing solution: {int(progress * 100)}% complete")
        
        # Final result is the last solution
        u = solutions[-1]
    else:
        # Solve normally
        if progress_callback:
            progress_callback(0.0, "Computing solution...")
        X, Y, u = solver.solve(config.num_steps)
        if progress_callback:
            progress_callback(1.0, "Solution complete.")
    
    # Create and return result object
    result = SimulationResult(
        x=solver.x,
        y=solver.y,
        X=X,
        Y=Y,
        u=u,
        times=times if config.show_evolution else None,
        solutions=solutions if config.show_evolution else None,
        config=config,
        dimension="2D",
        final_time=config.num_steps * config.dt
    )
    
    return result


def run_simulation(
    config: SimulationConfig,
    dimension: str,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> SimulationResult:
    """
    Run a finite velocity diffusion simulation in either 1D or 2D.
    
    Parameters:
        config: SimulationConfig object containing simulation parameters
        dimension: "1D" or "2D" to specify the simulation dimension
        progress_callback: Optional callback function for progress updates
    
    Returns:
        SimulationResult object containing the simulation results
        
    Raises:
        ValueError: If dimension is not "1D" or "2D"
    """
    # Validate configuration
    config.validate()
    
    # Run appropriate simulation based on dimension
    if dimension == "1D":
        return run_1d_simulation(config, progress_callback)
    elif dimension == "2D":
        return run_2d_simulation(config, progress_callback)
    else:
        raise ValueError(f"Invalid dimension: {dimension}. Must be '1D' or '2D'.")
