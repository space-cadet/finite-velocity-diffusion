"""
Component functions for the Parameter Analysis page.
These functions create the different analysis tools shown in the tabs.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import calculate_propagation_speed


def create_propagation_speed_tab(saved_params):
    """
    Create the content for the Propagation Speed analysis tab.
    
    Parameters:
    -----------
    saved_params : dict
        Dictionary of saved parameter values
    """
    st.header("Propagation Speed Analysis")
    st.markdown("""
    The finite velocity diffusion equation has a finite propagation speed given by:
    
    $c = \sqrt{\\frac{D}{\\tau}}$
    
    where D is the diffusion coefficient and τ is the relaxation time.
    """)
    
    # Create two columns for inputs and visualization
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Parameter inputs
        st.subheader("Parameters")
        fixed_param = st.radio("Fixed Parameter:", ["D (Diffusion Coefficient)", "τ (Relaxation Time)", "c (Propagation Speed)"])
        
        if fixed_param == "D (Diffusion Coefficient)":
            fixed_D = st.slider("D value:", min_value=0.1, max_value=10.0, value=saved_params.get("D", 1.0), step=0.1)
            tau_range = st.slider("τ range:", min_value=0.1, max_value=10.0, value=(0.1, 5.0), step=0.1)
            
            # Calculate speeds
            tau_values = np.linspace(tau_range[0], tau_range[1], 100)
            speed_values = [calculate_propagation_speed(fixed_D, tau) for tau in tau_values]
            
            # Create plot data
            plot_x = tau_values
            plot_y = speed_values
            x_label = "Relaxation Time (τ)"
            y_label = "Propagation Speed (c)"
            title = f"Propagation Speed vs. Relaxation Time (D = {fixed_D})"
            
        elif fixed_param == "τ (Relaxation Time)":
            fixed_tau = st.slider("τ value:", min_value=0.1, max_value=10.0, value=saved_params.get("tau", 1.0), step=0.1)
            D_range = st.slider("D range:", min_value=0.1, max_value=10.0, value=(0.1, 5.0), step=0.1)
            
            # Calculate speeds
            D_values = np.linspace(D_range[0], D_range[1], 100)
            speed_values = [calculate_propagation_speed(D, fixed_tau) for D in D_values]
            
            # Create plot data
            plot_x = D_values
            plot_y = speed_values
            x_label = "Diffusion Coefficient (D)"
            y_label = "Propagation Speed (c)"
            title = f"Propagation Speed vs. Diffusion Coefficient (τ = {fixed_tau})"
            
        else:  # fixed_param == "c (Propagation Speed)"
            fixed_c = st.slider("c value:", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
            D_range = st.slider("D range:", min_value=0.1, max_value=10.0, value=(0.1, 5.0), step=0.1)
            
            # Calculate tau values based on c^2 = D/tau -> tau = D/c^2
            D_values = np.linspace(D_range[0], D_range[1], 100)
            tau_values = [D / (fixed_c**2) for D in D_values]
            
            # Create plot data
            plot_x = D_values
            plot_y = tau_values
            x_label = "Diffusion Coefficient (D)"
            y_label = "Relaxation Time (τ)"
            title = f"Parameter Relationship for Constant Speed (c = {fixed_c})"
    
    with col2:
        # Visualization
        st.subheader("Visualization")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_x,
            y=plot_y,
            mode='lines',
            line=dict(width=2, color='blue'),
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            height=500,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display equation
        if fixed_param == "c (Propagation Speed)":
            st.latex(f"\\tau = \\frac{{D}}{{{fixed_c}^2}}")
        else:
            st.latex("c = \\sqrt{\\frac{D}{\\tau}}")


def create_wave_diffusion_spectrum_tab(saved_params):
    """
    Create the content for the Wave-Diffusion Spectrum analysis tab.
    
    Parameters:
    -----------
    saved_params : dict
        Dictionary of saved parameter values
    """
    st.header("Wave-Diffusion Spectrum Analysis")
    st.markdown("""
    The finite velocity diffusion equation bridges the gap between pure wave behavior and pure diffusion behavior.
    The equation can be rewritten as:
    
    $\\tau \\frac{\\partial^2 u}{\\partial t^2} + \\frac{\\partial u}{\\partial t} = D \\nabla^2 u$
    
    When τ approaches 0, the behavior approaches classical diffusion.
    As τ increases, the behavior becomes more wave-like.
    """)
    
    # Create sliders for wave-diffusion spectrum analysis
    D_value = st.slider(
        "Diffusion Coefficient (D):", 
        min_value=0.1, 
        max_value=5.0, 
        value=saved_params.get("D", 1.0), 
        step=0.1,
        key="wave_diff_D"
    )
    
    # Create multiple values of tau to show the spectrum
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Wave-Diffusion Balance")
        tau_min = st.number_input("Minimum τ value:", min_value=0.01, max_value=0.5, value=0.01, step=0.01)
        tau_max = st.number_input("Maximum τ value:", min_value=0.5, max_value=10.0, value=5.0, step=0.1)
        num_curves = st.slider("Number of curves:", min_value=3, max_value=10, value=5)
        
        # Calculate tau values
        tau_values = np.logspace(np.log10(tau_min), np.log10(tau_max), num_curves)
        tau_values = np.round(tau_values, 3)  # Round for better display
        
        st.markdown("### τ Values")
        st.write(", ".join([str(tau) for tau in tau_values]))
        
        # Display legend explanation
        st.markdown("### Legend")
        for i, tau in enumerate(tau_values):
            st.markdown(f"* **τ = {tau}**: {'More diffusion-like' if i == 0 else 'More wave-like' if i == len(tau_values)-1 else 'Intermediate behavior'}")
    
    with col2:
        # Create a "dummy" 1D diffusion problem to illustrate the spectrum
        # We'll show impulse responses for different tau values
        
        # Create spatial domain
        x = np.linspace(-10, 10, 200)
        
        # Create time points
        t_values = [0.5, 1.0, 2.0, 4.0]
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=[f"t = {t}" for t in t_values],
            shared_xaxes=True,
            shared_yaxes=True,
        )
        
        # For each time value, show the impulse response for different tau values
        for i, t in enumerate(t_values):
            row = i // 2 + 1
            col = i % 2 + 1
            
            for j, tau in enumerate(tau_values):
                # Calculate analytical solution for impulse response
                # For illustration purposes, we're using a simplified analytical form
                # that approximates the behavior
                
                # Calculate wave speed
                c = np.sqrt(D_value / tau)
                
                # For small tau (diffusion-like): Gaussian with width ~ sqrt(D*t)
                # For large tau (wave-like): Moving front at x = ±c*t
                
                if tau < 0.1:
                    # More diffusion-like
                    u = np.exp(-x**2 / (4 * D_value * t)) / np.sqrt(4 * np.pi * D_value * t)
                else:
                    # Blend of wave and diffusion
                    # Approximation of telegrapher's equation Green's function
                    u = np.zeros_like(x)
                    
                    # Inside the light cone (|x| < c*t)
                    mask = np.abs(x) < c * t
                    if np.any(mask):
                        wave_factor = np.exp(-t / (2 * tau))
                        u[mask] = wave_factor * np.exp(-(t - np.abs(x[mask]) / c)**2 / (D_value * t * tau)) / np.sqrt(D_value * t * tau)
                
                # Normalize for visualization
                if np.max(u) > 0:
                    u = u / np.max(u)
                
                # Add trace to subplot
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=u,
                        mode='lines',
                        name=f"τ = {tau}",
                        line=dict(
                            width=2,
                            color=f'rgb({int(255*(1-j/(len(tau_values)-1)))}, {int(255*j/(len(tau_values)-1))}, {int(255*0.5)})'
                        ),
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=row,
                    col=col
                )
        
        fig.update_layout(
            height=600,
            title="Wave-Diffusion Behavior Spectrum (Impulse Response)",
            legend_title="Relaxation Time (τ)",
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Position (x)")
        fig.update_yaxes(title_text="Normalized Response")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Note**: This visualization shows approximate impulse responses for different relaxation times (τ).
        * Small τ values produce diffusion-like behavior (Gaussian spreading)
        * Large τ values produce wave-like behavior (propagating fronts)
        """)


def create_stability_analysis_tab(default_params):
    """
    Create the content for the Stability Analysis tab.
    
    Parameters:
    -----------
    default_params : dict
        Dictionary of default parameter values
    """
    st.header("Stability Analysis")
    st.markdown("""
    The explicit finite difference scheme for solving the telegrapher's equation has a stability condition
    known as the Courant condition:
    
    $\\sqrt{\\frac{D \\cdot \\Delta t}{\\tau \\cdot \\Delta x^2}} \\leq 1$
    
    This tool helps you explore stable vs. unstable parameter regions for numerical solutions.
    """)
    
    # Create inputs for stability analysis
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Simulation Parameters")
        
        # Grid spacing
        dx = st.slider("Grid Spacing (Δx):", min_value=0.01, max_value=1.0, value=default_params.get("dx", 0.1), step=0.01)
        
        # Time step
        dt = st.slider("Time Step (Δt):", min_value=0.001, max_value=0.1, value=default_params.get("dt", 0.01), step=0.001)
        
        # Parameter ranges for analysis
        D_range = st.slider("D range:", min_value=0.1, max_value=10.0, value=(0.1, 5.0), step=0.1)
        tau_range = st.slider("τ range:", min_value=0.01, max_value=10.0, value=(0.1, 5.0), step=0.1)
        
        # Create parameter grids
        D_values = np.linspace(D_range[0], D_range[1], 100)
        tau_values = np.linspace(tau_range[0], tau_range[1], 100)
        
        # Create meshgrid for parameter space
        D_grid, tau_grid = np.meshgrid(D_values, tau_values)
        
        # Calculate Courant number
        courant_grid = np.sqrt(D_grid * dt / (tau_grid * dx**2))
        
        # Create stability mask
        stability_mask = courant_grid <= 1.0
    
    with col2:
        st.subheader("Stability Region")
        
        # Create stability plot
        fig = go.Figure()
        
        # Add contour for stability boundary
        fig.add_trace(go.Contour(
            z=courant_grid,
            x=D_values,
            y=tau_values,
            contours=dict(
                start=0,
                end=2,
                size=0.1,
                showlabels=True,
            ),
            colorscale='Viridis',
            colorbar=dict(
                title=dict(
                    text="Courant Number",
                    side="right"
                )
            ),
            hovertemplate="D: %{x:.2f}<br>τ: %{y:.2f}<br>Courant: %{z:.3f}<extra></extra>",
        ))
        
        # Add line for stability boundary (Courant = 1)
        boundary_contour = go.Contour(
            z=courant_grid,
            x=D_values,
            y=tau_values,
            contours=dict(
                start=1,
                end=1,
                coloring='lines',
                showlabels=False,
                labelfont=dict(size=12, color='white'),
            ),
            line=dict(width=3, color='red'),
            showscale=False,
            hoverinfo='none',
            name="Stability Boundary"
        )
        fig.add_trace(boundary_contour)
        
        # Add annotation for stable/unstable regions
        fig.add_annotation(
            x=D_range[0] + (D_range[1] - D_range[0]) * 0.2,
            y=tau_range[1] - (tau_range[1] - tau_range[0]) * 0.2,
            text="STABLE",
            font=dict(size=20, color="white"),
            showarrow=False,
        )
        
        fig.add_annotation(
            x=D_range[1] - (D_range[1] - D_range[0]) * 0.2,
            y=tau_range[0] + (tau_range[1] - tau_range[0]) * 0.2,
            text="UNSTABLE",
            font=dict(size=20, color="white"),
            showarrow=False,
        )
        
        # Update layout
        fig.update_layout(
            title="Stability Region for Explicit Scheme",
            xaxis_title="Diffusion Coefficient (D)",
            yaxis_title="Relaxation Time (τ)",
            height=600,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display key information
        st.markdown(f"""
        **Stability Condition**: $\\sqrt{{\\frac{{D \\cdot \\Delta t}}{{\\tau \\cdot \\Delta x^2}}}} \\leq 1$
        
        For the current parameters:
        - Grid spacing (Δx): {dx}
        - Time step (Δt): {dt}
        
        **Rule of thumb**: Higher D or smaller τ requires smaller time steps for stability.
        """)
