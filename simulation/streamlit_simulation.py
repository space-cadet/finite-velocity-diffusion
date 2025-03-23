"""
Simulation logic for the Streamlit finite velocity diffusion app.
"""

import numpy as np
from solver.one_dimensional import FiniteVelocityDiffusion1D
from solver.two_dimensional import FiniteVelocityDiffusion2D
from utils import (
    gaussian_initial_condition,
    classical_diffusion_solution,
    DEFAULT_PARAMETERS
)

def run_1d_simulation(params, progress_callback=None):
    """
    Run a 1D finite velocity diffusion simulation based on the provided parameters.
    
    Parameters:
    -----------
    params : dict
        Dictionary of simulation parameters
    progress_callback : callable, optional
        Callback function to update progress (takes progress fraction and status message)
    
    Returns:
    --------
    dict
        Dictionary containing simulation results
    """
    # Create solver
    solver = FiniteVelocityDiffusion1D(
        nx=DEFAULT_PARAMETERS['nx'],
        dx=DEFAULT_PARAMETERS['dx'],
        dt=DEFAULT_PARAMETERS['dt'],
        D=params['D'],
        tau=params['tau'],
        x_min=DEFAULT_PARAMETERS['x_min'],
        x_max=DEFAULT_PARAMETERS['x_max']
    )
    
    # Set initial condition
    initial_condition = gaussian_initial_condition(
        solver.x,
        amplitude=params['amplitude'],
        center=(solver.x[-1] + solver.x[0]) / 2,  # Center of domain
        sigma=params['sigma']
    )
    solver.set_initial_condition(initial_condition)
    
    # Storage for time evolution
    x = solver.x
    times = []
    solutions = []
    
    if params['show_evolution']:
        # Calculate step size for evolution plots/animation
        step_size = max(1, params['num_steps'] // params['evolution_steps'])
        
        # Run solver and capture intermediate results
        for i in range(0, params['num_steps'] + 1, step_size):
            if i == 0:
                # Initial condition
                times.append(0)
                solutions.append(solver.u.copy())
            else:
                # Solve for steps
                solver.solve(step_size)
                times.append(i * DEFAULT_PARAMETERS['dt'])
                solutions.append(solver.u.copy())
                
                # Update progress
                if progress_callback:
                    progress = min(1.0, i / params['num_steps'])
                    progress_callback(progress, f"Computing solution: {int(progress * 100)}% complete")
        
        # Final result is the last solution
        u = solutions[-1]
    else:
        # Solve normally
        if progress_callback:
            progress_callback(0.0, "Computing solution...")
        x, u = solver.solve(params['num_steps'])
        if progress_callback:
            progress_callback(1.0, "Solution complete")
    
    # Compute classical diffusion solution for comparison
    t = params['num_steps'] * DEFAULT_PARAMETERS['dt']
    u_classical = classical_diffusion_solution(
        x,
        t,
        params['D'],
        initial_amplitude=params['amplitude'],
        initial_center=(solver.x[-1] + solver.x[0]) / 2,  # Center of domain
        initial_sigma=params['sigma']
    )
    
    # Return results
    results = {
        'x': x,
        'u': u,
        'u_classical': u_classical,
        'times': times,
        'solutions': solutions,
        'final_time': t
    }
    
    return results

def run_2d_simulation(params, progress_callback=None):
    """
    Run a 2D finite velocity diffusion simulation based on the provided parameters.
    
    Parameters:
    -----------
    params : dict
        Dictionary of simulation parameters
    progress_callback : callable, optional
        Callback function to update progress (takes progress fraction and status message)
    
    Returns:
    --------
    dict
        Dictionary containing simulation results
    """
    # Create solver
    solver = FiniteVelocityDiffusion2D(
        nx=DEFAULT_PARAMETERS['nx'],
        ny=DEFAULT_PARAMETERS['ny'],
        dx=DEFAULT_PARAMETERS['dx'],
        dy=DEFAULT_PARAMETERS['dy'],
        dt=DEFAULT_PARAMETERS['dt'],
        D=params['D'],
        tau=params['tau'],
        x_min=DEFAULT_PARAMETERS['x_min'],
        x_max=DEFAULT_PARAMETERS['x_max'],
        y_min=DEFAULT_PARAMETERS['y_min'],
        y_max=DEFAULT_PARAMETERS['y_max']
    )
    
    # Set initial condition
    initial_condition = gaussian_initial_condition(
        solver.x,
        solver.y,
        amplitude=params['amplitude'],
        center=((solver.x[-1] + solver.x[0]) / 2, (solver.y[-1] + solver.y[0]) / 2),  # Center of domain
        sigma=(params['sigma'], params['sigma'])
    )
    solver.set_initial_condition(initial_condition)
    
    # Storage for time evolution
    X, Y = solver.X, solver.Y
    times = []
    solutions = []
    
    if params['show_evolution']:
        # Calculate step size for evolution plots/animation
        step_size = max(1, params['num_steps'] // params['evolution_steps'])
        
        # Run solver and capture intermediate results
        for i in range(0, params['num_steps'] + 1, step_size):
            if i == 0:
                # Initial condition
                times.append(0)
                solutions.append(solver.u.copy())
            else:
                # Solve for steps
                solver.solve(step_size)
                times.append(i * DEFAULT_PARAMETERS['dt'])
                solutions.append(solver.u.copy())
                
                # Update progress
                if progress_callback:
                    progress = min(1.0, i / params['num_steps'])
                    progress_callback(progress, f"Computing solution: {int(progress * 100)}% complete")
        
        # Final result is the last solution
        u = solutions[-1]
    else:
        # Solve normally
        if progress_callback:
            progress_callback(0.0, "Computing solution...")
        X, Y, u = solver.solve(params['num_steps'])
        if progress_callback:
            progress_callback(1.0, "Solution complete")
    
    # Return results
    results = {
        'x': solver.x,
        'y': solver.y,
        'X': X,
        'Y': Y,
        'u': u,
        'times': times,
        'solutions': solutions,
        'final_time': params['num_steps'] * DEFAULT_PARAMETERS['dt']
    }
    
    return results
