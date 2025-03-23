"""
One-dimensional finite velocity diffusion solver.

This module implements a numerical solver for the one-dimensional 
finite velocity diffusion equation (telegrapher's equation).
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict, Any, cast

class FiniteVelocityDiffusion1D:
    """
    Solver for the 1D finite velocity diffusion equation.
    
    This class implements a numerical solver for the telegrapher's equation:
    τ∂²u/∂t² + ∂u/∂t = D∇²u
    
    The solver uses a finite difference method with a second-order accurate
    central difference scheme for spatial derivatives.
    """
    
    def __init__(
        self,
        nx: int = 100,
        dx: float = 0.1,
        dt: float = 0.001,
        D: float = 1.0,
        tau: float = 1.0,
        x_min: float = 0.0,
        x_max: float = 10.0
    ) -> None:
        """
        Initialize the 1D finite velocity diffusion solver.
        
        Parameters:
        -----------
        nx : int
            Number of spatial points
        dx : float
            Spatial step size
        dt : float
            Time step size
        D : float
            Diffusion coefficient
        tau : float
            Relaxation time
        x_min : float
            Left boundary of spatial domain
        x_max : float
            Right boundary of spatial domain
            
        Raises:
        -------
        ValueError
            If the configuration is numerically unstable (Courant number > 1)
        """
        self.nx: int = nx
        self.dx: float = dx
        self.dt: float = dt
        self.D: float = D
        self.tau: float = tau
        
        # Spatial grid
        self.x: np.ndarray = np.linspace(x_min, x_max, nx)
        
        # Initialize solution arrays
        self.u: np.ndarray = np.zeros(nx)  # Current solution
        self.u_prev: np.ndarray = np.zeros(nx)  # Previous time step
        self.u_prev2: np.ndarray = np.zeros(nx)  # Two time steps ago
        self.u_next: np.ndarray = np.zeros(nx)  # Next time step (temporary storage)
        
        # Courant number for stability
        self.courant: float = np.sqrt(D * dt / (tau * dx**2))
        
        # Check stability condition
        if self.courant > 1:
            raise ValueError(
                f"Unstable configuration: Courant number {self.courant:.2f} > 1. "
                "Reduce dt or increase dx."
            )
    
    def set_initial_condition(self, initial_condition: np.ndarray) -> None:
        """
        Set the initial condition for the simulation.
        
        Parameters:
        -----------
        initial_condition : np.ndarray
            Array of shape (nx,) containing initial values
            
        Raises:
        -------
        ValueError
            If the initial condition array has incorrect shape
        """
        if initial_condition.shape != (self.nx,):
            raise ValueError(f"Initial condition must have shape ({self.nx},)")
        self.u = initial_condition.copy()
        self.u_prev = initial_condition.copy()
        self.u_prev2 = initial_condition.copy()
    
    def set_boundary_conditions(
        self,
        left_boundary: Optional[float] = None,
        right_boundary: Optional[float] = None
    ) -> None:
        """
        Set boundary conditions for the simulation.
        
        Parameters:
        -----------
        left_boundary : float, optional
            Value at the left boundary (x = x_min)
        right_boundary : float, optional
            Value at the right boundary (x = x_max)
        """
        if left_boundary is not None:
            self.u[0] = left_boundary
            self.u_prev[0] = left_boundary
            self.u_prev2[0] = left_boundary
        
        if right_boundary is not None:
            self.u[-1] = right_boundary
            self.u_prev[-1] = right_boundary
            self.u_prev2[-1] = right_boundary
    
    def step(self) -> None:
        """
        Perform one time step of the simulation.
        
        This method advances the solution by one time step using a finite difference
        scheme for the telegrapher's equation.
        """
        # Compute spatial derivatives using central differences
        d2u_dx2 = np.zeros_like(self.u)
        d2u_dx2[1:-1] = (self.u[2:] - 2*self.u[1:-1] + self.u[:-2]) / (self.dx**2)
        
        # Apply boundary conditions for second derivative
        d2u_dx2[0] = d2u_dx2[1]  # Neumann boundary at left
        d2u_dx2[-1] = d2u_dx2[-2]  # Neumann boundary at right
        
        # Update solution using finite difference scheme
        self.u_next = (
            2*self.u - self.u_prev + 
            (self.dt**2/self.tau) * (self.D * d2u_dx2 - (self.u - self.u_prev)/self.dt)
        )
        
        # Update solution arrays
        self.u_prev2 = self.u_prev.copy()
        self.u_prev = self.u.copy()
        self.u = self.u_next.copy()
    
    def solve(self, num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the equation for a given number of time steps.
        
        Parameters:
        -----------
        num_steps : int
            Number of time steps to perform
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Arrays containing the spatial grid and final solution
        """
        for _ in range(num_steps):
            self.step()
        return self.x, self.u
    
    def get_propagation_speed(self) -> float:
        """
        Calculate the finite propagation speed for the simulation.
        
        In the finite velocity diffusion (telegrapher's equation), the
        propagation speed is given by sqrt(D/tau).
        
        Returns:
        --------
        float
            Propagation speed
        """
        return np.sqrt(self.D / self.tau)
    
    def get_solution_state(self) -> Dict[str, np.ndarray]:
        """
        Get the current state of the solution.
        
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary containing the current solution state
        """
        return {
            "x": self.x,
            "u": self.u.copy(),
            "u_prev": self.u_prev.copy(),
            "u_prev2": self.u_prev2.copy()
        }
    
    def set_solution_state(self, state: Dict[str, np.ndarray]) -> None:
        """
        Set the solution state from a previously saved state.
        
        Parameters:
        -----------
        state : Dict[str, np.ndarray]
            Dictionary containing the solution state
            
        Raises:
        -------
        ValueError
            If the state arrays have incorrect shapes
        """
        if state["x"].shape != self.x.shape:
            raise ValueError(f"Spatial grid mismatch: expected {self.x.shape}, got {state['x'].shape}")
        
        for key in ["u", "u_prev", "u_prev2"]:
            if key in state and state[key].shape != (self.nx,):
                raise ValueError(f"{key} has incorrect shape: expected ({self.nx},), got {state[key].shape}")
        
        # Set the solution state
        self.u = state["u"].copy()
        self.u_prev = state["u_prev"].copy()
        self.u_prev2 = state["u_prev2"].copy()
