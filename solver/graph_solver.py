"""
Graph-based diffusion solvers.

This module implements diffusion processes on graph structures, including
both ordinary diffusion (heat equation) and finite-velocity diffusion 
(telegrapher's equation).
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Callable, Optional, Union


class OrdinaryGraphDiffusion:
    """
    Implements ordinary diffusion (heat equation) on graphs.
    
    The heat equation on graphs is:
    ∂u/∂t = D ∇²ᵍ u
    
    where ∇²ᵍ is the graph Laplacian.
    """
    
    def __init__(
        self,
        graph: nx.Graph,
        diffusion_coefficient: float = 1.0,
        dt: float = 0.01,
        normalize_laplacian: bool = False
    ):
        """
        Initialize the ordinary graph diffusion solver.
        
        Parameters:
        -----------
        graph : nx.Graph
            The graph on which to solve the diffusion equation
        diffusion_coefficient : float
            Diffusion coefficient (D)
        dt : float
            Time step size
        normalize_laplacian : bool
            Whether to use the normalized Laplacian instead of the standard Laplacian
        """
        self.graph = graph
        self.D = diffusion_coefficient
        self.dt = dt
        self.normalize_laplacian = normalize_laplacian
        
        # Calculate the graph Laplacian matrix
        if normalize_laplacian:
            self.laplacian = nx.normalized_laplacian_matrix(graph).todense()
        else:
            self.laplacian = nx.laplacian_matrix(graph).todense()
        
        # Initialize solution
        self.nodes = list(graph.nodes())
        self.node_indices = {node: i for i, node in enumerate(self.nodes)}
        self.u = np.zeros(len(self.nodes))
        
        # Storage for time evolution
        self.time_points = [0.0]
        self.solutions = {}
        
    def set_initial_condition(self, initial_values: Dict):
        """
        Set the initial condition for the diffusion process.
        
        Parameters:
        -----------
        initial_values : Dict
            Dictionary mapping nodes to initial values
        """
        self.u = np.zeros(len(self.nodes))
        for node, value in initial_values.items():
            if node in self.node_indices:
                self.u[self.node_indices[node]] = value
        
        # Store initial condition
        self.solutions[0.0] = self.get_node_values()
        
    def step(self):
        """
        Perform one time step of the simulation.
        
        For ordinary diffusion, we use the explicit Euler method:
        u(t+dt) = u(t) + D * dt * ∇²ᵍ u(t)
        """
        # Calculate the Laplacian term
        laplacian_term = -self.D * np.dot(self.laplacian, self.u)
        
        # Update solution
        self.u = self.u + self.dt * laplacian_term
        
        # Update time and store solution
        current_time = self.time_points[-1] + self.dt
        self.time_points.append(current_time)
        self.solutions[current_time] = self.get_node_values()
        
        return current_time
    
    def run(self, num_steps: int, store_interval: int = 1):
        """
        Run the simulation for a specified number of time steps.
        
        Parameters:
        -----------
        num_steps : int
            Number of time steps to run
        store_interval : int
            Interval at which to store solutions (1 = every step)
            
        Returns:
        --------
        Dict
            Dictionary of solutions at each stored time point
        """
        for i in range(num_steps):
            current_time = self.step()
            
            # Remove solution if not at store interval
            if (i + 1) % store_interval != 0 and current_time != self.time_points[-1]:
                del self.solutions[current_time]
                
        return self.solutions
    
    def get_node_values(self) -> Dict:
        """
        Get current values at each node.
        
        Returns:
        --------
        Dict
            Dictionary mapping nodes to their current values
        """
        return {node: self.u[i] for node, i in self.node_indices.items()}
    
    def get_solutions(self) -> Dict:
        """
        Get all stored solutions.
        
        Returns:
        --------
        Dict
            Dictionary mapping time points to solutions
        """
        return self.solutions


class FiniteVelocityGraphDiffusion:
    """
    Implements finite-velocity diffusion (telegrapher's equation) on graphs.
    
    The telegrapher's equation on graphs is:
    τ ∂²u/∂t² + ∂u/∂t = D ∇²ᵍ u
    
    where ∇²ᵍ is the graph Laplacian and τ is the relaxation time.
    """
    
    def __init__(
        self,
        graph: nx.Graph,
        diffusion_coefficient: float = 1.0,
        relaxation_time: float = 0.1,
        dt: float = 0.01,
        normalize_laplacian: bool = False
    ):
        """
        Initialize the finite-velocity graph diffusion solver.
        
        Parameters:
        -----------
        graph : nx.Graph
            The graph on which to solve the diffusion equation
        diffusion_coefficient : float
            Diffusion coefficient (D)
        relaxation_time : float
            Relaxation time (τ) that determines the wave-like behavior
        dt : float
            Time step size
        normalize_laplacian : bool
            Whether to use the normalized Laplacian instead of the standard Laplacian
        """
        self.graph = graph
        self.D = diffusion_coefficient
        self.tau = relaxation_time
        self.dt = dt
        self.normalize_laplacian = normalize_laplacian
        
        # Calculate the graph Laplacian matrix
        if normalize_laplacian:
            self.laplacian = nx.normalized_laplacian_matrix(graph).todense()
        else:
            self.laplacian = nx.laplacian_matrix(graph).todense()
        
        # Initialize solution
        self.nodes = list(graph.nodes())
        self.node_indices = {node: i for i, node in enumerate(self.nodes)}
        self.u = np.zeros(len(self.nodes))
        self.u_prev = np.zeros(len(self.nodes))  # Previous time step
        
        # Storage for time evolution
        self.time_points = [0.0]
        self.solutions = {}
        
        # Check stability
        self._check_stability()
        
    def _check_stability(self):
        """
        Check if the chosen parameters satisfy stability conditions.
        
        For the telegrapher's equation on graphs, the stability condition
        is more complex and depends on the eigenvalues of the Laplacian.
        This is a simplified check.
        """
        # Maximum eigenvalue of the Laplacian provides the worst-case scenario
        try:
            # For small graphs, we can compute the max eigenvalue directly
            if len(self.nodes) < 1000:
                max_eigenvalue = max(abs(np.linalg.eigvals(self.laplacian)))
                if self.dt > np.sqrt(2 * self.tau / (self.D * max_eigenvalue)):
                    print(f"Warning: The chosen time step may lead to instability.")
                    print(f"Recommended dt <= {np.sqrt(2 * self.tau / (self.D * max_eigenvalue)):.6f}")
            else:
                # For larger graphs, we use a heuristic
                # Maximum degree is often a good approximation for the largest eigenvalue
                # of the unnormalized Laplacian
                max_degree = max(dict(self.graph.degree()).values())
                if not self.normalize_laplacian and self.dt > np.sqrt(2 * self.tau / (self.D * max_degree)):
                    print(f"Warning: The chosen time step may lead to instability.")
                    print(f"Recommended dt <= {np.sqrt(2 * self.tau / (self.D * max_degree)):.6f}")
        except:
            print("Could not compute stability condition. Proceed with caution.")
        
    def set_initial_condition(
        self, 
        initial_values: Dict, 
        initial_derivatives: Optional[Dict] = None
    ):
        """
        Set the initial condition for the diffusion process.
        
        Parameters:
        -----------
        initial_values : Dict
            Dictionary mapping nodes to initial values
        initial_derivatives : Dict, optional
            Dictionary mapping nodes to initial time derivatives (du/dt at t=0)
            If None, all derivatives are set to 0
        """
        self.u = np.zeros(len(self.nodes))
        
        # Set initial values
        for node, value in initial_values.items():
            if node in self.node_indices:
                self.u[self.node_indices[node]] = value
        
        # Store initial condition
        self.solutions[0.0] = self.get_node_values()
        
        # Calculate u_prev using initial derivatives if provided
        self.u_prev = np.copy(self.u)
        
        if initial_derivatives is not None:
            # Backward Euler approximation for first step
            # u_prev = u - dt * (du/dt)
            for node, deriv in initial_derivatives.items():
                if node in self.node_indices:
                    self.u_prev[self.node_indices[node]] -= self.dt * deriv
        
    def step(self):
        """
        Perform one time step of the simulation.
        
        For finite-velocity diffusion, we use the following discretization:
        (u(t+dt) - 2u(t) + u(t-dt))/dt² * τ + (u(t+dt) - u(t-dt))/(2dt) = D * ∇²ᵍ u(t)
        
        Simplified to:
        u(t+dt) = [2u(t) - u(t-dt) + (D*dt²/τ)*∇²ᵍ u(t)] / [1 + dt/(2τ)]
        """
        # Calculate the Laplacian term
        laplacian_term = -self.D * np.dot(self.laplacian, self.u)
        
        # Calculate the coefficient for the update
        coef_denominator = 1 + self.dt / (2 * self.tau)
        
        # Calculate the new solution
        u_next = (2 * self.u - self.u_prev + (self.dt ** 2 / self.tau) * laplacian_term) / coef_denominator
        
        # Update solution
        self.u_prev = np.copy(self.u)
        self.u = u_next
        
        # Update time and store solution
        current_time = self.time_points[-1] + self.dt
        self.time_points.append(current_time)
        self.solutions[current_time] = self.get_node_values()
        
        return current_time
    
    def run(self, num_steps: int, store_interval: int = 1):
        """
        Run the simulation for a specified number of time steps.
        
        Parameters:
        -----------
        num_steps : int
            Number of time steps to run
        store_interval : int
            Interval at which to store solutions (1 = every step)
            
        Returns:
        --------
        Dict
            Dictionary of solutions at each stored time point
        """
        for i in range(num_steps):
            current_time = self.step()
            
            # Remove solution if not at store interval
            if (i + 1) % store_interval != 0 and current_time != self.time_points[-1]:
                del self.solutions[current_time]
                
        return self.solutions
    
    def get_node_values(self) -> Dict:
        """
        Get current values at each node.
        
        Returns:
        --------
        Dict
            Dictionary mapping nodes to their current values
        """
        return {node: self.u[i] for node, i in self.node_indices.items()}
    
    def get_solutions(self) -> Dict:
        """
        Get all stored solutions.
        
        Returns:
        --------
        Dict
            Dictionary mapping time points to solutions
        """
        return self.solutions


class GraphDiffusionWithPotential:
    """
    Implements diffusion with potential term on graphs.
    
    The equation on graphs is:
    ∂u/∂t = D ∇²ᵍ u - V(x)u
    
    where ∇²ᵍ is the graph Laplacian and V(x) is the potential at each node.
    """
    
    def __init__(
        self,
        graph: nx.Graph,
        diffusion_coefficient: float = 1.0,
        dt: float = 0.01,
        normalize_laplacian: bool = False,
        potential: Optional[Dict] = None
    ):
        """
        Initialize the graph diffusion solver with potential.
        
        Parameters:
        -----------
        graph : nx.Graph
            The graph on which to solve the diffusion equation
        diffusion_coefficient : float
            Diffusion coefficient (D)
        dt : float
            Time step size
        normalize_laplacian : bool
            Whether to use the normalized Laplacian instead of the standard Laplacian
        potential : Dict, optional
            Dictionary mapping nodes to potential values
            If None, all potentials are set to 0
        """
        self.graph = graph
        self.D = diffusion_coefficient
        self.dt = dt
        self.normalize_laplacian = normalize_laplacian
        
        # Calculate the graph Laplacian matrix
        if normalize_laplacian:
            self.laplacian = nx.normalized_laplacian_matrix(graph).todense()
        else:
            self.laplacian = nx.laplacian_matrix(graph).todense()
        
        # Initialize solution
        self.nodes = list(graph.nodes())
        self.node_indices = {node: i for i, node in enumerate(self.nodes)}
        self.u = np.zeros(len(self.nodes))
        
        # Initialize potential
        self.potential = np.zeros(len(self.nodes))
        if potential is not None:
            for node, value in potential.items():
                if node in self.node_indices:
                    self.potential[self.node_indices[node]] = value
        
        # Storage for time evolution
        self.time_points = [0.0]
        self.solutions = {}
        
    def set_initial_condition(self, initial_values: Dict):
        """
        Set the initial condition for the diffusion process.
        
        Parameters:
        -----------
        initial_values : Dict
            Dictionary mapping nodes to initial values
        """
        self.u = np.zeros(len(self.nodes))
        for node, value in initial_values.items():
            if node in self.node_indices:
                self.u[self.node_indices[node]] = value
        
        # Store initial condition
        self.solutions[0.0] = self.get_node_values()
        
    def set_potential(self, potential: Dict):
        """
        Set the potential values at each node.
        
        Parameters:
        -----------
        potential : Dict
            Dictionary mapping nodes to potential values
        """
        self.potential = np.zeros(len(self.nodes))
        for node, value in potential.items():
            if node in self.node_indices:
                self.potential[self.node_indices[node]] = value
        
    def step(self):
        """
        Perform one time step of the simulation.
        
        For diffusion with potential, we use the explicit Euler method:
        u(t+dt) = u(t) + dt * [D * ∇²ᵍ u(t) - V(x)u(t)]
        """
        # Calculate the Laplacian term
        laplacian_term = -self.D * np.dot(self.laplacian, self.u)
        
        # Calculate the potential term
        potential_term = -self.potential * self.u
        
        # Update solution
        self.u = self.u + self.dt * (laplacian_term + potential_term)
        
        # Update time and store solution
        current_time = self.time_points[-1] + self.dt
        self.time_points.append(current_time)
        self.solutions[current_time] = self.get_node_values()
        
        return current_time
    
    def run(self, num_steps: int, store_interval: int = 1):
        """
        Run the simulation for a specified number of time steps.
        
        Parameters:
        -----------
        num_steps : int
            Number of time steps to run
        store_interval : int
            Interval at which to store solutions (1 = every step)
            
        Returns:
        --------
        Dict
            Dictionary of solutions at each stored time point
        """
        for i in range(num_steps):
            current_time = self.step()
            
            # Remove solution if not at store interval
            if (i + 1) % store_interval != 0 and current_time != self.time_points[-1]:
                del self.solutions[current_time]
                
        return self.solutions
    
    def get_node_values(self) -> Dict:
        """
        Get current values at each node.
        
        Returns:
        --------
        Dict
            Dictionary mapping nodes to their current values
        """
        return {node: self.u[i] for node, i in self.node_indices.items()}
    
    def get_solutions(self) -> Dict:
        """
        Get all stored solutions.
        
        Returns:
        --------
        Dict
            Dictionary mapping time points to solutions
        """
        return self.solutions


# Utility functions for working with graph diffusion

def create_gaussian_initial_condition(
    graph: nx.Graph,
    center_node,
    sigma: float = 1.0,
    amplitude: float = 1.0,
    pos: Optional[Dict] = None
) -> Dict:
    """
    Create a Gaussian initial condition centered at a specified node.
    
    Parameters:
    -----------
    graph : nx.Graph
        The graph on which to create the initial condition
    center_node
        The node at which to center the Gaussian
    sigma : float
        Width of the Gaussian
    amplitude : float
        Amplitude of the Gaussian
    pos : Dict, optional
        Dictionary of node positions for distance calculation
        If None, shortest path distance is used
        
    Returns:
    --------
    Dict
        Dictionary mapping nodes to initial values
    """
    initial_values = {}
    
    # If positions are provided, use Euclidean distance
    if pos is not None:
        center_pos = pos[center_node]
        for node in graph.nodes():
            node_pos = pos[node]
            distance = np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(center_pos, node_pos)))
            initial_values[node] = amplitude * np.exp(-(distance ** 2) / (2 * sigma ** 2))
    
    # Otherwise, use shortest path distance
    else:
        # Compute shortest path distances
        try:
            path_lengths = dict(nx.single_source_shortest_path_length(graph, center_node))
            for node in graph.nodes():
                if node in path_lengths:
                    distance = path_lengths[node]
                    initial_values[node] = amplitude * np.exp(-(distance ** 2) / (2 * sigma ** 2))
                else:
                    # Node not reachable from center_node
                    initial_values[node] = 0.0
        except:
            # Fallback for disconnected graphs
            for node in graph.nodes():
                try:
                    distance = nx.shortest_path_length(graph, center_node, node)
                    initial_values[node] = amplitude * np.exp(-(distance ** 2) / (2 * sigma ** 2))
                except:
                    initial_values[node] = 0.0
    
    return initial_values


def create_delta_initial_condition(
    graph: nx.Graph,
    center_node,
    amplitude: float = 1.0
) -> Dict:
    """
    Create a delta function initial condition at a specified node.
    
    Parameters:
    -----------
    graph : nx.Graph
        The graph on which to create the initial condition
    center_node
        The node at which to place the delta function
    amplitude : float
        Amplitude of the delta function
        
    Returns:
    --------
    Dict
        Dictionary mapping nodes to initial values
    """
    initial_values = {node: 0.0 for node in graph.nodes()}
    initial_values[center_node] = amplitude
    return initial_values
