#!/usr/bin/env python3
"""
Command-line tool to run finite velocity diffusion simulations and save results.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from solver import FiniteVelocityDiffusion1D, FiniteVelocityDiffusion2D
from utils import (
    gaussian_initial_condition,
    classical_diffusion_solution,
    calculate_propagation_speed,
    DEFAULT_PARAMETERS
)
from visualization.plotting import (
    plot_1d_comparison,
    plot_2d_heatmap,
    plot_time_series
)
from visualization.animation import create_1d_animation, create_2d_animation


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run finite velocity diffusion simulations.'
    )
    
    parser.add_argument(
        '--dimension', '-d', type=int, choices=[1, 2], default=1,
        help='Dimension of the simulation (1 or 2)'
    )
    
    parser.add_argument(
        '--diffusion-coef', '-D', type=float, default=DEFAULT_PARAMETERS['D'],
        help='Diffusion coefficient'
    )
    
    parser.add_argument(
        '--relaxation-time', '-tau', type=float, default=DEFAULT_PARAMETERS['tau'],
        help='Relaxation time'
    )
    
    parser.add_argument(
        '--num-steps', '-n', type=int, default=100,
        help='Number of time steps'
    )
    
    parser.add_argument(
        '--amplitude', '-a', type=float, default=1.0,
        help='Initial amplitude'
    )
    
    parser.add_argument(
        '--sigma', '-s', type=float, default=1.0,
        help='Initial standard deviation'
    )
    
    parser.add_argument(
        '--output-dir', '-o', type=str, default='results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--animate', action='store_true',
        help='Create animation of the solution'
    )
    
    parser.add_argument(
        '--evolution-points', '-e', type=int, default=10,
        help='Number of points to capture for time evolution'
    )
    
    return parser.parse_args()


def run_1d_simulation(args):
    """Run 1D simulation with the given parameters."""
    # Initialize solver
    solver = FiniteVelocityDiffusion1D(
        nx=DEFAULT_PARAMETERS['nx'],
        dx=DEFAULT_PARAMETERS['dx'],
        dt=DEFAULT_PARAMETERS['dt'],
        D=args.diffusion_coef,
        tau=args.relaxation_time,
        x_min=DEFAULT_PARAMETERS['x_min'],
        x_max=DEFAULT_PARAMETERS['x_max']
    )
    
    # Set initial condition
    center = (solver.x.max() + solver.x.min()) / 2
    initial_condition = gaussian_initial_condition(
        solver.x,
        amplitude=args.amplitude,
        center=center,
        sigma=args.sigma
    )
    solver.set_initial_condition(initial_condition)
    
    # Calculate propagation speed
    c = calculate_propagation_speed(args.diffusion_coef, args.relaxation_time)
    print(f"Propagation speed: {c:.3f} units/time")
    
    # Create output directory if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"1d_sim_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run simulation with time evolution if animation is requested
    if args.animate:
        # Store solutions at different time points
        step_size = max(1, args.num_steps // args.evolution_points)
        times = []
        solutions = []
        
        # Initial state
        times.append(0)
        solutions.append(solver.u.copy())
        
        # Run simulation and collect samples
        for i in range(1, args.evolution_points + 1):
            steps = step_size
            solver.solve(steps)
            t = i * step_size * DEFAULT_PARAMETERS['dt']
            times.append(t)
            solutions.append(solver.u.copy())
        
        # Final state
        x = solver.x
        u_final = solutions[-1]
        
        # Create animation
        ani = create_1d_animation(
            x=x,
            solutions=solutions,
            times=times,
            title=f'1D Finite Velocity Diffusion (D={args.diffusion_coef}, τ={args.relaxation_time})',
            save_path=os.path.join(output_dir, 'animation.mp4')
        )
        
        # Create time evolution plot
        fig, _ = plot_time_series(
            times=times,
            solutions=solutions,
            x=x,
            title="Time Evolution of Finite Velocity Diffusion"
        )
        fig.savefig(os.path.join(output_dir, 'time_evolution.png'), dpi=300)
        plt.close(fig)
    else:
        # Just run the simulation
        x, u_final = solver.solve(args.num_steps)
    
    # Compute classical diffusion solution for comparison
    t_final = args.num_steps * DEFAULT_PARAMETERS['dt']
    u_classical = classical_diffusion_solution(
        x,
        t_final,
        args.diffusion_coef,
        initial_amplitude=args.amplitude,
        initial_center=center,
        initial_sigma=args.sigma
    )
    
    # Create comparison plot
    fig, _ = plot_1d_comparison(
        x=x,
        u_finite=u_final,
        u_classical=u_classical,
        title=f'1D Finite Velocity vs Classical Diffusion (t = {t_final:.3f})'
    )
    fig.savefig(os.path.join(output_dir, 'comparison.png'), dpi=300)
    plt.close(fig)
    
    # Save parameters
    with open(os.path.join(output_dir, 'parameters.txt'), 'w') as f:
        f.write(f"Dimension: 1D\n")
        f.write(f"Diffusion coefficient (D): {args.diffusion_coef}\n")
        f.write(f"Relaxation time (τ): {args.relaxation_time}\n")
        f.write(f"Propagation speed: {c:.3f}\n")
        f.write(f"Number of steps: {args.num_steps}\n")
        f.write(f"Final time: {t_final:.3f}\n")
        f.write(f"Initial amplitude: {args.amplitude}\n")
        f.write(f"Initial sigma: {args.sigma}\n")
        f.write(f"dx: {DEFAULT_PARAMETERS['dx']}\n")
        f.write(f"dt: {DEFAULT_PARAMETERS['dt']}\n")
    
    # Save numerical data
    np.savetxt(
        os.path.join(output_dir, 'solution_data.csv'),
        np.column_stack((x, u_final, u_classical)),
        delimiter=',',
        header='x,finite_velocity,classical',
        comments=''
    )
    
    print(f"Results saved to {output_dir}")


def run_2d_simulation(args):
    """Run 2D simulation with the given parameters."""
    # Initialize solver
    solver = FiniteVelocityDiffusion2D(
        nx=DEFAULT_PARAMETERS['nx'],
        ny=DEFAULT_PARAMETERS['ny'],
        dx=DEFAULT_PARAMETERS['dx'],
        dy=DEFAULT_PARAMETERS['dy'],
        dt=DEFAULT_PARAMETERS['dt'],
        D=args.diffusion_coef,
        tau=args.relaxation_time,
        x_min=DEFAULT_PARAMETERS['x_min'],
        x_max=DEFAULT_PARAMETERS['x_max'],
        y_min=DEFAULT_PARAMETERS['y_min'],
        y_max=DEFAULT_PARAMETERS['y_max']
    )
    
    # Set initial condition
    center_x = (solver.x.max() + solver.x.min()) / 2
    center_y = (solver.y.max() + solver.y.min()) / 2
    initial_condition = gaussian_initial_condition(
        solver.x,
        solver.y,
        amplitude=args.amplitude,
        center=(center_x, center_y),
        sigma=(args.sigma, args.sigma)
    )
    solver.set_initial_condition(initial_condition)
    
    # Calculate propagation speed
    c = calculate_propagation_speed(args.diffusion_coef, args.relaxation_time)
    print(f"Propagation speed: {c:.3f} units/time")
    
    # Create output directory if it doesn't exist
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"2d_sim_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run simulation with time evolution if animation is requested
    if args.animate:
        # Store solutions at different time points
        step_size = max(1, args.num_steps // args.evolution_points)
        times = []
        solutions = []
        
        # Initial state
        times.append(0)
        solutions.append(solver.u.copy())
        
        # Run simulation and collect samples
        for i in range(1, args.evolution_points + 1):
            steps = step_size
            solver.solve(steps)
            t = i * step_size * DEFAULT_PARAMETERS['dt']
            times.append(t)
            solutions.append(solver.u.copy())
        
        # Final state
        X, Y, u_final = solver.X, solver.Y, solutions[-1]
        
        # Create animation
        ani = create_2d_animation(
            X=X,
            Y=Y,
            solutions=solutions,
            times=times,
            title=f'2D Finite Velocity Diffusion (D={args.diffusion_coef}, τ={args.relaxation_time})',
            save_path=os.path.join(output_dir, 'animation.mp4')
        )
        
        # Save individual time frames
        for i, (t, sol) in enumerate(zip(times, solutions)):
            fig, _ = plot_2d_heatmap(
                X=X,
                Y=Y,
                u=sol,
                title=f'Time = {t:.3f}'
            )
            fig.savefig(os.path.join(output_dir, f'frame_{i:03d}.png'), dpi=300)
            plt.close(fig)
    else:
        # Just run the simulation
        X, Y, u_final = solver.solve(args.num_steps)
    
    # Create final state plot
    fig, _ = plot_2d_heatmap(
        X=X,
        Y=Y,
        u=u_final,
        title=f'2D Finite Velocity Diffusion (t = {args.num_steps * DEFAULT_PARAMETERS["dt"]:.3f})'
    )
    fig.savefig(os.path.join(output_dir, 'final_state.png'), dpi=300)
    plt.close(fig)
    
    # Save parameters
    with open(os.path.join(output_dir, 'parameters.txt'), 'w') as f:
        f.write(f"Dimension: 2D\n")
        f.write(f"Diffusion coefficient (D): {args.diffusion_coef}\n")
        f.write(f"Relaxation time (τ): {args.relaxation_time}\n")
        f.write(f"Propagation speed: {c:.3f}\n")
        f.write(f"Number of steps: {args.num_steps}\n")
        f.write(f"Final time: {args.num_steps * DEFAULT_PARAMETERS['dt']:.3f}\n")
        f.write(f"Initial amplitude: {args.amplitude}\n")
        f.write(f"Initial sigma: {args.sigma}\n")
        f.write(f"dx: {DEFAULT_PARAMETERS['dx']}\n")
        f.write(f"dy: {DEFAULT_PARAMETERS['dy']}\n")
        f.write(f"dt: {DEFAULT_PARAMETERS['dt']}\n")
    
    # Save center slice data
    center_idx = solver.ny // 2
    np.savetxt(
        os.path.join(output_dir, 'center_slice_data.csv'),
        np.column_stack((solver.x, u_final[center_idx, :])),
        delimiter=',',
        header='x,u',
        comments=''
    )
    
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dimension == 1:
        run_1d_simulation(args)
    else:
        run_2d_simulation(args)
