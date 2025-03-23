import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from solver.one_dimensional import FiniteVelocityDiffusion1D
from utils import gaussian_initial_condition, classical_diffusion_solution

# Set page configuration
st.set_page_config(page_title="Finite Velocity Diffusion", layout="wide")

# Title and description
st.title("Finite Velocity Diffusion Solver")
st.markdown("""
This app demonstrates the finite velocity diffusion equation (telegrapher's equation):
τ∂²u/∂t² + ∂u/∂t = D∇²u

The solution is compared with the classical diffusion equation for a Gaussian initial condition.
""")

# Sidebar controls for parameters
st.sidebar.header("Simulation Parameters")

# Physical parameters
D = st.sidebar.slider(
    "Diffusion Coefficient (D)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1
)

tau = st.sidebar.slider(
    "Relaxation Time (τ)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
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

# Run the simulation
if st.sidebar.button("Run Simulation"):
    # Progress bar
    progress_bar = st.progress(0)
    
    # Parameters
    nx = 100
    dx = 0.1
    dt = 0.001
    x_min = 0.0
    x_max = 10.0
    
    # Initialize solver
    solver = FiniteVelocityDiffusion1D(
        nx=nx, dx=dx, dt=dt, D=D, tau=tau, x_min=x_min, x_max=x_max
    )
    
    # Set initial condition
    center = (x_max + x_min) / 2
    initial_condition = gaussian_initial_condition(
        solver.x, amplitude=amplitude, center=center, sigma=sigma
    )
    solver.set_initial_condition(initial_condition)
    
    # Create placeholder for initial condition plot
    st.subheader("Initial Condition")
    fig_initial, ax_initial = plt.subplots(figsize=(10, 4))
    ax_initial.plot(solver.x, initial_condition)
    ax_initial.set_xlabel("Position")
    ax_initial.set_ylabel("Concentration")
    ax_initial.grid(True)
    initial_plot = st.pyplot(fig_initial)
    
    # Run simulation with progress updates
    step_size = max(1, num_steps // 10)  # Update progress bar in 10 steps
    
    for i in range(0, num_steps + 1, step_size):
        if i > 0:
            solver.solve(step_size)
        progress_bar.progress(min(i / num_steps, 1.0))
    
    # Final solution
    x, u = solver.x, solver.u
    
    # Compute classical diffusion solution for comparison
    t = num_steps * dt
    u_classical = classical_diffusion_solution(
        x, t, D, 
        initial_amplitude=amplitude,
        initial_center=center,
        initial_sigma=sigma
    )
    
    # Plot results
    st.subheader("Simulation Results")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, u, 'b-', label='Finite Velocity')
    ax.plot(x, u_classical, 'r--', label='Classical')
    ax.set_xlabel('Position')
    ax.set_ylabel('Concentration')
    ax.set_title(f'1D Finite Velocity vs Classical Diffusion (t = {t:.3f})')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    # Calculate and display propagation speed and other info
    c = np.sqrt(D / tau)
    courant = np.sqrt(D * dt / (tau * dx**2))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Simulation Info")
    st.sidebar.markdown(f"Time: {t:.3f}")
    st.sidebar.markdown(f"Propagation Speed: {c:.3f}")
    st.sidebar.markdown(f"Courant Number: {courant:.3f}")
    
    if courant > 1:
        st.sidebar.warning(f"Courant number {courant:.3f} > 1. Simulation may be unstable.")
    
    # Show equations and explanation
    st.markdown("---")
    st.subheader("About the Equations")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Finite Velocity Diffusion")
        st.latex(r"\tau \frac{\partial^2 u}{\partial t^2} + \frac{\partial u}{\partial t} = D \nabla^2 u")
        st.markdown("""
        The finite velocity diffusion equation (also known as the telegrapher's equation) combines
        wave-like and diffusion-like behavior. It has a finite propagation speed of $c = \sqrt{D/\tau}$.
        """)
    
    with col2:
        st.markdown("### Classical Diffusion")
        st.latex(r"\frac{\partial u}{\partial t} = D \nabla^2 u")
        st.markdown("""
        The classical diffusion equation has infinite propagation speed, meaning that
        a disturbance at one point instantaneously affects all other points, which is physically unrealistic.
        """)

else:
    # Information to display when the app starts
    st.info("Adjust the parameters on the sidebar and click 'Run Simulation' to start.")
    
    # Show equations and explanation
    st.markdown("---")
    st.subheader("About the Equations")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Finite Velocity Diffusion")
        st.latex(r"\tau \frac{\partial^2 u}{\partial t^2} + \frac{\partial u}{\partial t} = D \nabla^2 u")
        st.markdown("""
        The finite velocity diffusion equation (also known as the telegrapher's equation) combines
        wave-like and diffusion-like behavior. It has a finite propagation speed of $c = \sqrt{D/\tau}$.
        """)
    
    with col2:
        st.markdown("### Classical Diffusion")
        st.latex(r"\frac{\partial u}{\partial t} = D \nabla^2 u")
        st.markdown("""
        The classical diffusion equation has infinite propagation speed, meaning that
        a disturbance at one point instantaneously affects all other points, which is physically unrealistic.
        """)
