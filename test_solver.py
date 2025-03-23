import numpy as np
import matplotlib.pyplot as plt
from solver import FiniteVelocityDiffusion1D
from utils import (
    gaussian_initial_condition,
    classical_diffusion_solution,
    calculate_propagation_speed
)
from visualization.plotting import plot_1d_comparison, plot_time_series

def test_1d_solver():
    """
    Test the 1D finite velocity diffusion solver with a Gaussian initial condition.
    
    This function will:
    1. Set up the solver with a Gaussian initial condition
    2. Run the simulation for multiple time steps
    3. Plot the results and compare with classical diffusion
    4. Show the time evolution of the solution
    """
    # Solver parameters
    D = 1.0  # Diffusion coefficient
    tau = 1.0  # Relaxation time
    nx = 200  # Number of spatial points
    dx = 0.1  # Spatial step size
    dt = 0.001  # Time step size
    x_min = 0.0  # Left boundary
    x_max = 20.0  # Right boundary
    
    # Initialize solver
    solver = FiniteVelocityDiffusion1D(
        nx=nx,
        dx=dx,
        dt=dt,
        D=D,
        tau=tau,
        x_min=x_min,
        x_max=x_max
    )
    
    # Set initial condition (Gaussian centered in the domain)
    center = (x_max + x_min) / 2
    sigma = 1.0
    amplitude = 1.0
    initial_condition = gaussian_initial_condition(
        solver.x,
        amplitude=amplitude,
        center=center,
        sigma=sigma
    )
    solver.set_initial_condition(initial_condition)
    
    # Calculate propagation speed
    c = calculate_propagation_speed(D, tau)
    print(f"Propagation speed: {c} units/time")
    
    # Store solutions at different time points
    num_steps = 500
    sample_points = 5
    step_size = num_steps // sample_points
    times = []
    solutions = []
    
    # Initial state
    times.append(0)
    solutions.append(solver.u.copy())
    
    # Run simulation and collect samples
    for i in range(1, sample_points + 1):
        steps = step_size
        _, u = solver.solve(steps)
        t = i * step_size * dt
        times.append(t)
        solutions.append(u)
    
    # Final state
    x = solver.x
    u_final = solutions[-1]
    
    # Compute classical diffusion solution for comparison
    t_final = times[-1]
    u_classical = classical_diffusion_solution(
        x,
        t_final,
        D,
        initial_amplitude=amplitude,
        initial_center=center,
        initial_sigma=sigma
    )
    
    # Plot comparison
    fig1, _ = plot_1d_comparison(
        x=x,
        u_finite=u_final,
        u_classical=u_classical,
        title=f'1D Finite Velocity vs Classical Diffusion (t = {t_final:.3f})'
    )
    
    # Plot time evolution
    fig2, _ = plot_time_series(
        times=times,
        solutions=solutions,
        x=x,
        title="Time Evolution of Finite Velocity Diffusion"
    )
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    test_1d_solver()
