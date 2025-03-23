"""
Two-dimensional finite velocity diffusion solver.

This module implements a numerical solver for the two-dimensional 
finite velocity diffusion equation (telegrapher's equation).
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Dict, Any

class FiniteVelocityDiffusion2D:
    """
    Solver for the 2D finite velocity diffusion equation.
    
    This class implements a numerical solver for the telegrapher's equation:
    τ∂²u/∂t² + ∂u/∂t = D∇²u
    
    The solver uses a finite difference method with a second-order accurate
    central difference scheme for spatial derivatives.
    """
    
    def __init__(
        self,
        nx: int = 100,
        ny: int = 100,
        dx: float = 0.1,
        dy: float = 0.1,
        dt: float = 0.001,
        D: float = 1.0,
        tau: float = 1.0,
        x_min: float = 0.0,
        x_max: float = 10.0,
        y_min: float = 0.0,
        y_max: float = 10.0
    ) -> None:
        """
        Initialize the 2D finite velocity diffusion solver.
        
        Parameters:
        -----------
        nx, ny : int
            Number of spatial points in x and y directions
        dx, dy : float
            Spatial step sizes in x and y directions
        dt : float
            Time step size
        D : float
            Diffusion coefficient
        tau : float
            Relaxation time
        x_min, x_max : float
            Boundaries of spatial domain in x direction
        y_min, y_max : float
            Boundaries of spatial domain in y direction
            
        Raises:
        -------
        ValueError
            If the configuration is numerically unstable (Courant number > 1)
        """
        self.nx: int = nx
        self.ny: int = ny
        self.dx: float = dx
        self.dy: float = dy
        self.dt: float = dt
        self.D: float = D
        self.tau: float = tau
        
        # Spatial grids
        self.x: np.ndarray = np.linspace(x_min, x_max, nx)
        self.y: np.ndarray = np.linspace(y_min, y_max, ny)
        self.X: np.ndarray
        self.Y: np.ndarray
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize solution arrays
        self.u: np.ndarray = np.zeros((ny, nx))  # Current solution
        self.u_prev: np.ndarray = np.zeros((ny, nx))  # Previous time step
        self.u_prev2: np.ndarray = np.zeros((ny, nx))  # Two time steps ago
        self.u_next: np.ndarray = np.zeros((ny, nx))  # Next time step (temporary storage)
        
        # Courant numbers for stability
        self.courant_x: float = np.sqrt(D * dt / (tau * dx**2))
        self.courant_y: float = np.sqrt(D * dt / (tau * dy**2))
        
        # Check stability condition
        if self.courant_x > 1 or self.courant_y > 1:
            raise ValueError(
                f"Unstable configuration: Courant numbers ({self.courant_x:.2f}, {self.courant_y:.2f}) > 1. "
                "Reduce dt or increase dx/dy."
            )
    
    def set_initial_condition(self, initial_condition: np.ndarray) -> None:
        """
        Set the initial condition for the simulation.
        
        Parameters:
        -----------
        initial_condition : np.ndarray
            Array of shape (ny, nx) containing initial values
            
        Raises:
        -------
        ValueError
            If the initial condition array has incorrect shape
        """
        if initial_condition.shape != (self.ny, self.nx):
            raise ValueError(f"Initial condition must have shape ({self.ny}, {self.nx})")
        self.u = initial_condition.copy()
        self.u_prev = initial_condition.copy()
        self.u_prev2 = initial_condition.copy()
    
    def set_boundary_conditions(
        self,
        left: Optional[float] = None,
        right: Optional[float] = None,
        top: Optional[float] = None,
        bottom: Optional[float] = None
    ) -> None:
        """
        Set boundary conditions for the simulation.
        
        Parameters:
        -----------
        left, right : float, optional
            Values at the left and right boundaries
        top, bottom : float, optional
            Values at the top and bottom boundaries
        """
        if left is not None:
            self.u[:, 0] = left
            self.u_prev[:, 0] = left
            self.u_prev2[:, 0] = left
        
        if right is not None:
            self.u[:, -1] = right
            self.u_prev[:, -1] = right
            self.u_prev2[:, -1] = right
        
        if top is not None:
            self.u[0, :] = top
            self.u_prev[0, :] = top
            self.u_prev2[0, :] = top
        
        if bottom is not None:
            self.u[-1, :] = bottom
            self.u_prev[-1, :] = bottom
            self.u_prev2[-1, :] = bottom
    
    def step(self) -> None:
        """
        Perform one time step of the simulation.
        
        This method advances the solution by one time step using a finite difference
        scheme for the telegrapher's equation.
        """
        # Compute spatial derivatives using central differences
        d2u_dx2 = np.zeros_like(self.u)
        d2u_dy2 = np.zeros_like(self.u)
        
        # x-direction derivatives
        d2u_dx2[:, 1:-1] = (self.u[:, 2:] - 2*self.u[:, 1:-1] + self.u[:, :-2]) / (self.dx**2)
        
        # y-direction derivatives
        d2u_dy2[1:-1, :] = (self.u[2:, :] - 2*self.u[1:-1, :] + self.u[:-2, :]) / (self.dy**2)
        
        # Apply boundary conditions for second derivatives
        d2u_dx2[:, 0] = d2u_dx2[:, 1]  # Neumann boundary at left
        d2u_dx2[:, -1] = d2u_dx2[:, -2]  # Neumann boundary at right
        d2u_dy2[0, :] = d2u_dy2[1, :]  # Neumann boundary at top
        d2u_dy2[-1, :] = d2u_dy2[-2, :]  # Neumann boundary at bottom
        
        # Laplacian
        laplacian = d2u_dx2 + d2u_dy2
        
        # Update solution using finite difference scheme
        self.u_next = (
            2*self.u - self.u_prev + 
            (self.dt**2/self.tau) * (self.D * laplacian - (self.u - self.u_prev)/self.dt)
        )
        
        # Update solution arrays
        self.u_prev2 = self.u_prev.copy()
        self.u_prev = self.u.copy()
        self.u = self.u_next.copy()
    
    def solve(self, num_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the equation for a given number of time steps.
        
        Parameters:
        -----------
        num_steps : int
            Number of time steps to perform
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Arrays containing the spatial grids (X, Y) and final solution
        """
        for _ in range(num_steps):
            self.step()
        return self.X, self.Y, self.u
    
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
            "y": self.y,
            "X": self.X,
            "Y": self.Y,
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
            raise ValueError(f"x grid mismatch: expected {self.x.shape}, got {state['x'].shape}")
        
        if state["y"].shape != self.y.shape:
            raise ValueError(f"y grid mismatch: expected {self.y.shape}, got {state['y'].shape}")
        
        for key in ["u", "u_prev", "u_prev2"]:
            if key in state and state[key].shape != (self.ny, self.nx):
                raise ValueError(f"{key} has incorrect shape: expected ({self.ny}, {self.nx}), got {state[key].shape}")
        
        # Set the solution state
        self.u = state["u"].copy()
        self.u_prev = state["u_prev"].copy()
        self.u_prev2 = state["u_prev2"].copy()
