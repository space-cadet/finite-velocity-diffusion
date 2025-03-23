import numpy as np
import matplotlib.pyplot as plt
from solver.one_dimensional import FiniteVelocityDiffusion1D
from utils import gaussian_initial_condition, classical_diffusion_solution

def main():
    """
    Minimal working example of the finite velocity diffusion solver.
    This example:
    1. Sets up a 1D finite velocity diffusion problem with a Gaussian initial condition
    2. Solves it for a specified number of time steps
    3. Compares the result with classical diffusion
    4. Displays the plot
    """
    # Parameters
    D = 1.0        # Diffusion coefficient
    tau = 1.0      # Relaxation time
    nx = 100       # Number of spatial points
    dx = 0.1       # Spatial step size
    dt = 0.001     # Time step size
    x_min = 0.0    # Left boundary
    x_max = 10.0   # Right boundary
    num_steps = 100  # Number of time steps to solve
    
    # Initialize solver
    solver = FiniteVelocityDiffusion1D(
        nx=nx, dx=dx, dt=dt, D=D, tau=tau, x_min=x_min, x_max=x_max
    )
    
    # Set initial condition (Gaussian centered in the domain)
    center = (x_max + x_min) / 2
    sigma = 1.0
    amplitude = 1.0
    initial_condition = gaussian_initial_condition(
        solver.x, amplitude=amplitude, center=center, sigma=sigma
    )
    solver.set_initial_condition(initial_condition)
    
    # Solve
    x, u = solver.solve(num_steps)
    
    # Compute classical diffusion solution for comparison
    t = num_steps * dt
    u_classical = classical_diffusion_solution(
        x, t, D, 
        initial_amplitude=amplitude,
        initial_center=center,
        initial_sigma=sigma
    )
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(x, u, 'b-', label='Finite Velocity')
    plt.plot(x, u_classical, 'r--', label='Classical')
    plt.xlabel('Position')
    plt.ylabel('Concentration')
    plt.title(f'1D Finite Velocity vs Classical Diffusion (t = {t:.3f})')
    plt.legend()
    plt.grid(True)
    
    # Calculate and display propagation speed
    c = np.sqrt(D / tau)
    plt.annotate(f"Propagation speed: {c:.3f}", xy=(0.02, 0.95), xycoords='axes fraction')
    
    # Show plot
    plt.tight_layout()
    plt.savefig('minimal_example_result.png')  # Save figure for reference
    plt.show()

if __name__ == "__main__":
    main()
