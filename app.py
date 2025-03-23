import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from solver import FiniteVelocityDiffusion1D, FiniteVelocityDiffusion2D
from visualization.plotting import (
    plot_1d_comparison,
    plot_2d_heatmap,
    plot_time_series
)
from utils import (
    gaussian_initial_condition,
    classical_diffusion_solution,
    calculate_propagation_speed,
    DEFAULT_PARAMETERS
)

st.set_page_config(page_title="Finite Velocity Diffusion Solver", layout="wide")

st.title("Finite Velocity Diffusion Solver")
st.markdown("""
This app solves the finite velocity diffusion equation (telegrapher's equation):
τ∂²u/∂t² + ∂u/∂t = D∇²u

The solution is compared with the classical diffusion equation for a Gaussian initial condition.
""")

# Sidebar controls
st.sidebar.header("Simulation Parameters")

# Dimension selection
dimension = st.sidebar.radio("Select Dimension", ["1D", "2D"])

# Physical parameters
D = st.sidebar.slider(
    "Diffusion Coefficient (D)",
    min_value=0.1,
    max_value=5.0,
    value=DEFAULT_PARAMETERS['D'],
    step=0.1
)

tau = st.sidebar.slider(
    "Relaxation Time (τ)",
    min_value=0.1,
    max_value=5.0,
    value=DEFAULT_PARAMETERS['tau'],
    step=0.1
)

# Initial condition parameters
amplitude = st.sidebar.slider(
    "Initial Amplitude",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1
)

sigma = st.sidebar.slider(
    "Initial Standard Deviation",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1
)

# Simulation parameters
num_steps = st.sidebar.slider(
    "Number of Time Steps",
    min_value=10,
    max_value=1000,
    value=100,
    step=10
)

# Show time evolution
show_evolution = st.sidebar.checkbox("Show Time Evolution", value=False)
if show_evolution:
    evolution_steps = st.sidebar.slider(
        "Number of Time Points to Show",
        min_value=2,
        max_value=10,
        value=5
    )

# Create solver
if dimension == "1D":
    solver = FiniteVelocityDiffusion1D(
        nx=DEFAULT_PARAMETERS['nx'],
        dx=DEFAULT_PARAMETERS['dx'],
        dt=DEFAULT_PARAMETERS['dt'],
        D=D,
        tau=tau,
        x_min=DEFAULT_PARAMETERS['x_min'],
        x_max=DEFAULT_PARAMETERS['x_max']
    )
    
    # Set initial condition
    initial_condition = gaussian_initial_condition(
        solver.x,
        amplitude=amplitude,
        center=(solver.x_max + solver.x_min) / 2,
        sigma=sigma
    )
    solver.set_initial_condition(initial_condition)
    
    # Show time evolution if requested
    if show_evolution:
        # Store solutions at different time points
        times = []
        solutions = []
        
        # Calculate step size for evolution plots
        step_size = max(1, num_steps // evolution_steps)
        
        # Run solver and capture intermediate results
        for i in range(0, num_steps + 1, step_size):
            if i == 0:
                # Initial condition
                times.append(0)
                solutions.append(solver.u.copy())
            else:
                # Solve for steps
                solver.solve(step_size)
                times.append(i * DEFAULT_PARAMETERS['dt'])
                solutions.append(solver.u.copy())
        
        # Final result
        x = solver.x
        u = solutions[-1]
    else:
        # Solve normally
        x, u = solver.solve(num_steps)
    
    # Compute classical diffusion solution for comparison
    t = num_steps * DEFAULT_PARAMETERS['dt']
    u_classical = classical_diffusion_solution(
        x,
        t,
        D,
        initial_amplitude=amplitude,
        initial_center=(solver.x_max + solver.x_min) / 2,
        initial_sigma=sigma
    )
    
    # Plot results using visualization module
    fig, _ = plot_1d_comparison(
        x=x,
        u_finite=u,
        u_classical=u_classical,
        title='1D Finite Velocity vs Classical Diffusion'
    )
    st.pyplot(fig)
    
    # Plot time evolution if requested
    if show_evolution:
        fig, _ = plot_time_series(
            times=times,
            solutions=solutions,
            x=x,
            title="Time Evolution of Finite Velocity Diffusion"
        )
        st.pyplot(fig)
    
else:  # 2D case
    solver = FiniteVelocityDiffusion2D(
        nx=DEFAULT_PARAMETERS['nx'],
        ny=DEFAULT_PARAMETERS['ny'],
        dx=DEFAULT_PARAMETERS['dx'],
        dy=DEFAULT_PARAMETERS['dy'],
        dt=DEFAULT_PARAMETERS['dt'],
        D=D,
        tau=tau,
        x_min=DEFAULT_PARAMETERS['x_min'],
        x_max=DEFAULT_PARAMETERS['x_max'],
        y_min=DEFAULT_PARAMETERS['y_min'],
        y_max=DEFAULT_PARAMETERS['y_max']
    )
    
    # Set initial condition
    initial_condition = gaussian_initial_condition(
        solver.x,
        solver.y,
        amplitude=amplitude,
        center=((solver.x_max + solver.x_min) / 2, (solver.y_max + solver.y_min) / 2),
        sigma=(sigma, sigma)
    )
    solver.set_initial_condition(initial_condition)
    
    # Show time evolution if requested
    if show_evolution:
        # Store solutions at different time points
        times = []
        solutions = []
        
        # Calculate step size for evolution plots
        step_size = max(1, num_steps // evolution_steps)
        
        # Run solver and capture intermediate results
        for i in range(0, num_steps + 1, step_size):
            if i == 0:
                # Initial condition
                times.append(0)
                solutions.append(solver.u.copy())
            else:
                # Solve for steps
                solver.solve(step_size)
                times.append(i * DEFAULT_PARAMETERS['dt'])
                solutions.append(solver.u.copy())
        
        # Final result
        X, Y, u = solver.X, solver.Y, solutions[-1]
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Final Result", "Time Evolution"])
        
        with tab1:
            # Plot final results
            fig, _ = plot_2d_heatmap(
                X=X,
                Y=Y,
                u=u,
                title='2D Finite Velocity Diffusion (Final State)'
            )
            st.pyplot(fig)
        
        with tab2:
            # Plot time evolution as multiple heatmaps
            st.subheader("Time Evolution of 2D Diffusion")
            for i, (t, sol) in enumerate(zip(times, solutions)):
                st.write(f"Time: {t:.3f}")
                fig, _ = plot_2d_heatmap(
                    X=X,
                    Y=Y,
                    u=sol,
                    title=f'Time = {t:.3f}'
                )
                st.pyplot(fig)
    else:
        # Solve normally
        X, Y, u = solver.solve(num_steps)
        
        # Plot results using visualization module
        fig, _ = plot_2d_heatmap(
            X=X,
            Y=Y,
            u=u,
            title='2D Finite Velocity Diffusion'
        )
        st.pyplot(fig)



# Add information about the simulation
st.sidebar.markdown("---")
st.sidebar.markdown("### Simulation Info")
st.sidebar.markdown(f"Time: {num_steps * DEFAULT_PARAMETERS['dt']:.3f}")
st.sidebar.markdown(f"Courant Number: {np.sqrt(D * DEFAULT_PARAMETERS['dt'] / (tau * DEFAULT_PARAMETERS['dx']**2)):.3f}")
st.sidebar.markdown(f"Propagation Speed: {calculate_propagation_speed(D, tau):.3f}") 