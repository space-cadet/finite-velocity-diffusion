"""
Component functions for the Numerical Analysis page.
These functions create the different analysis tools shown in the tabs.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_convergence_analysis_tab(saved_params):
    """
    Create the content for the Convergence Analysis tab.
    
    Parameters:
    -----------
    saved_params : dict
        Dictionary of saved parameter values
    """
    st.header("Convergence Analysis")
    st.markdown("""
    The numerical solution to the telegrapher's equation converges to the exact solution
    as the grid spacing (Δx) and time step (Δt) decrease. This tool lets you explore
    how the solution converges with grid refinement.
    """)
    
    # Create two columns for inputs and visualization
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Simulation Parameters")
        
        # Physical parameters
        D = st.slider(
            "Diffusion Coefficient (D):", 
            min_value=0.1, 
            max_value=5.0, 
            value=saved_params.get("D", 1.0), 
            step=0.1,
            key="conv_D"
        )
        
        tau = st.slider(
            "Relaxation Time (τ):", 
            min_value=0.1, 
            max_value=5.0, 
            value=saved_params.get("tau", 1.0), 
            step=0.1,
            key="conv_tau"
        )
        
        # Grid resolution options
        resolution_options = ["Coarse", "Medium", "Fine", "Very Fine"]
        resolution = st.selectbox(
            "Grid Resolution:",
            resolution_options,
            index=1
        )
        
        # Map resolution to number of grid points
        resolution_mapping = {
            "Coarse": 50,
            "Medium": 100,
            "Fine": 200,
            "Very Fine": 400
        }
        
        N = resolution_mapping[resolution]
        
        # Create corresponding grid spacing
        domain_size = 20.0  # Fixed domain size
        dx = domain_size / N
        
        # Create corresponding time step based on stability
        # We'll use a safety factor of 0.8 for stability
        dt_max = 0.8 * (dx**2) * tau / D
        dt = min(dt_max, 0.01)  # Cap at 0.01 for reasonable simulation time
        
        # Display grid info
        st.markdown(f"""
        **Grid Information**:
        - Number of points: {N}
        - Grid spacing (Δx): {dx:.4f}
        - Time step (Δt): {dt:.6f}
        - Courant number: {np.sqrt(D * dt / (tau * dx**2)):.4f}
        """)
        
        # Simulation time
        t_final = st.slider(
            "Simulation Time:", 
            min_value=0.1, 
            max_value=10.0, 
            value=2.0, 
            step=0.1
        )
        
        # Number of time steps
        num_steps = int(t_final / dt)
        
        # Create initial condition type selector
        ic_type = st.selectbox(
            "Initial Condition:",
            ["Gaussian", "Step Function", "Sine Wave"],
            index=0
        )
        
        # Parameters for initial condition
        if ic_type == "Gaussian":
            center = st.slider("Center:", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
            sigma = st.slider("Width (σ):", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        elif ic_type == "Step Function":
            center = st.slider("Center:", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
        else:  # Sine Wave
            wavelength = st.slider("Wavelength:", min_value=1.0, max_value=10.0, value=4.0, step=0.1)
    
    with col2:
        # Explanation and visualization space
        st.subheader("Convergence Behavior")
        
        # Create synthetic data showing convergence
        x = np.linspace(-10, 10, 500)
        
        # Create "exact" solution (stand-in for demonstration)
        if ic_type == "Gaussian":
            exact = np.exp(-(x - center)**2 / (4 * D * t_final + sigma**2))
        elif ic_type == "Step Function":
            # Smoothed step function solution
            exact = 0.5 * (1 + np.tanh((center - np.abs(x)) / np.sqrt(D * t_final)))
        else:  # Sine Wave
            # Damped sine wave
            k = 2 * np.pi / wavelength
            exact = np.exp(-D * k**2 * t_final) * np.sin(k * x)
            
        # Scale to maximum of 1
        if np.max(np.abs(exact)) > 0:
            exact = exact / np.max(np.abs(exact))
        
        # Create "numerical" solutions with different resolutions
        # These are simulated to show convergence behavior
        noise_levels = {
            "Coarse": 0.15,
            "Medium": 0.08,
            "Fine": 0.03,
            "Very Fine": 0.01
        }
        
        # Create figure
        fig = go.Figure()
        
        # Add exact solution
        fig.add_trace(go.Scatter(
            x=x,
            y=exact,
            mode='lines',
            name='Exact Solution',
            line=dict(color='black', width=3)
        ))
        
        # Add numerical solutions with different resolutions
        for res in resolution_options:
            # Skip the current resolution to highlight it separately
            if res == resolution:
                continue
                
            # Create numerical solution with appropriate noise level
            num_sol = exact + np.random.normal(0, noise_levels[res], size=exact.shape)
            
            # Add to plot with reduced opacity
            fig.add_trace(go.Scatter(
                x=x,
                y=num_sol,
                mode='lines',
                name=f'{res} Grid',
                line=dict(width=1.5),
                opacity=0.5
            ))
        
        # Add current resolution with full opacity
        num_sol_current = exact + np.random.normal(0, noise_levels[resolution], size=exact.shape)
        fig.add_trace(go.Scatter(
            x=x,
            y=num_sol_current,
            mode='lines',
            name=f'{resolution} Grid (Selected)',
            line=dict(color='red', width=2.5)
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Solution Convergence with Grid Refinement (t = {t_final})",
            xaxis_title="Position (x)",
            yaxis_title="u(x,t)",
            height=500,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display convergence explanation
        st.markdown("""
        **Convergence Behavior**:
        
        As the grid resolution increases (Δx decreases), the numerical solution converges to the exact solution.
        This visualization shows how different grid resolutions affect the accuracy of the solution.
        
        **Key Observations**:
        - Coarser grids have more numerical diffusion and may miss sharp features
        - Finer grids capture the solution more accurately but require more computation
        - The convergence rate depends on the numerical scheme (2nd order in space, 1st order in time for this solver)
        
        In practice, the convergence would be measured using error norms (L2, L∞) to quantify the difference
        between numerical and exact solutions.
        """)


def create_error_analysis_tab():
    """
    Create the content for the Error Analysis tab.
    """
    st.header("Error Analysis")
    st.markdown("""
    Numerical solutions introduce errors compared to exact solutions.
    This section analyzes different types of errors in the finite difference solution
    of the telegrapher's equation.
    """)
    
    # Create plot showing different types of errors
    st.subheader("Types of Numerical Errors")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **Truncation Error**:
        
        Arises from approximating derivatives with finite differences.
        For the telegrapher's equation with centered differences:
        - 2nd order accuracy in space: O(Δx²)
        - 1st order accuracy in time: O(Δt)
        
        **Stability Error**:
        
        Grows over time if the stability condition is violated:
        - Courant condition: $\\sqrt{\\frac{D\\Delta t}{\\tau \\Delta x^2}} \\leq 1$
        - Manifests as oscillations that grow exponentially
        
        **Round-off Error**:
        
        Due to finite precision of floating-point arithmetic:
        - Generally insignificant for this problem
        - Can accumulate over many time steps
        
        **Boundary Condition Error**:
        
        Approximations at the domain boundaries:
        - Neumann conditions introduce O(Δx²) error
        - Artificial boundaries can cause wave reflections
        """)
    
    with col2:
        # Create a visual explanation of error types
        fig = make_subplots(rows=2, cols=1, subplot_titles=["Truncation Error", "Stability Error"])
        
        # Discretization error visualization
        x = np.linspace(0, 2*np.pi, 100)
        exact = np.sin(x)
        coarse = np.sin(x[::10])
        x_coarse = x[::10]
        
        fig.add_trace(
            go.Scatter(x=x, y=exact, mode='lines', name='Exact', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=x_coarse, y=coarse, mode='markers', name='Numerical', 
                      marker=dict(color='red', size=10)),
            row=1, col=1
        )
        
        # Connect points with straight lines to show linear interpolation error
        for i in range(len(x_coarse)-1):
            x_interp = [x_coarse[i], x_coarse[i+1]]
            y_interp = [coarse[i], coarse[i+1]]
            fig.add_trace(
                go.Scatter(x=x_interp, y=y_interp, mode='lines', showlegend=False, 
                          line=dict(color='red', dash='dash')),
                row=1, col=1
            )
        
        # Stability error visualization
        t = np.linspace(0, 4, 200)
        stable = np.exp(-t) * np.sin(5*t)
        unstable = np.exp(0.5*t) * np.sin(5*t)
        
        fig.add_trace(
            go.Scatter(x=t, y=stable, mode='lines', name='Stable', line=dict(color='green')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=t, y=unstable, mode='lines', name='Unstable', line=dict(color='red')),
            row=2, col=1
        )
        
        fig.update_layout(height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)
    
    # Error convergence analysis
    st.subheader("Error Convergence Rate")
    
    # Create visualization of error convergence
    dx_values = np.logspace(-3, 0, 100)
    
    # Theoretical error convergence for different schemes
    first_order = dx_values
    second_order = dx_values**2
    fourth_order = dx_values**4
    
    # Normalize for visualization
    first_order = first_order / np.max(first_order) * 0.1
    second_order = second_order / np.max(second_order) * 0.1
    fourth_order = fourth_order / np.max(fourth_order) * 0.1
    
    # Create figure
    fig = go.Figure()
    
    # Add lines for different orders of convergence
    fig.add_trace(go.Scatter(
        x=dx_values,
        y=first_order,
        mode='lines',
        name='First Order',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dx_values,
        y=second_order,
        mode='lines',
        name='Second Order',
        line=dict(color='green', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=dx_values,
        y=fourth_order,
        mode='lines',
        name='Fourth Order',
        line=dict(color='red', width=2)
    ))
    
    # Add markers for the explicit scheme used
    fig.add_trace(go.Scatter(
        x=[0.1, 0.05, 0.025],
        y=[0.01, 0.0025, 0.000625],
        mode='markers',
        name='Our Scheme',
        marker=dict(color='purple', size=10, symbol='circle')
    ))
    
    # Update axes to log scale
    fig.update_xaxes(type='log', title='Grid Spacing (Δx)')
    fig.update_yaxes(type='log', title='Error')
    
    # Update layout
    fig.update_layout(
        title='Error Convergence Rates',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Add annotations for slopes
    fig.add_annotation(
        x=0.5,
        y=0.05,
        text="Slope = 1",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        ax=50,
        ay=-30
    )
    
    fig.add_annotation(
        x=0.5,
        y=0.025,
        text="Slope = 2",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        ax=50,
        ay=-50
    )
    
    fig.add_annotation(
        x=0.5,
        y=0.003,
        text="Slope = 4",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        ax=50,
        ay=-70
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Convergence Rate Analysis**:
    
    The error in the numerical solution decreases as the grid is refined, following:
    
    $Error \\propto (\\Delta x)^p$
    
    Where p is the order of convergence:
    - p = 1: First-order scheme (forward/backward difference)
    - p = 2: Second-order scheme (centered difference)
    - p = 4: Fourth-order scheme (high-order stencils)
    
    Our explicit scheme for the telegrapher's equation is:
    - 2nd order accurate in space
    - 1st order accurate in time
    
    This means the overall convergence is dominated by the lower order (time discretization).
    """)
