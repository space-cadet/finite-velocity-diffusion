"""
Sidebar UI components for the Streamlit finite velocity diffusion app.
"""

import streamlit as st
import numpy as np
from utils import DEFAULT_PARAMETERS, calculate_propagation_speed

def create_sidebar_controls():
    """
    Create and render the sidebar controls for parameter adjustment.
    
    Returns:
    --------
    dict
        Dictionary containing the user-selected parameters
    """
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
    evolution_steps = None
    viz_type = None
    animation_speed = None
    
    if show_evolution:
        evolution_steps = st.sidebar.slider(
            "Number of Time Points to Show",
            min_value=5,
            max_value=50,
            value=20
        )
        
        # Visualization options
        st.sidebar.markdown("### Visualization Options")
        viz_type = st.sidebar.radio(
            "Visualization Type",
            ["Static Plots", "Interactive Slider", "Plotly Animation"],
            index=0
        )
        
        if viz_type == "Plotly Animation":
            animation_speed = st.sidebar.slider(
                "Animation Speed (frames per second)",
                min_value=1,
                max_value=30,
                value=10,
                step=1
            )

    # Return all parameters as a dictionary
    return {
        "dimension": dimension,
        "D": D,
        "tau": tau,
        "amplitude": amplitude,
        "sigma": sigma,
        "num_steps": num_steps,
        "show_evolution": show_evolution,
        "evolution_steps": evolution_steps,
        "viz_type": viz_type,
        "animation_speed": animation_speed,
    }

def display_simulation_info(params):
    """
    Display simulation information in the sidebar.
    
    Parameters:
    -----------
    params : dict
        Dictionary containing simulation parameters
    """
    # Add information about the simulation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Simulation Info")
    st.sidebar.markdown(f"Time: {params['num_steps'] * DEFAULT_PARAMETERS['dt']:.3f}")
    
    # Calculate and display Courant number
    if params["dimension"] == "1D":
        courant = np.sqrt(params["D"] * DEFAULT_PARAMETERS['dt'] / 
                         (params["tau"] * DEFAULT_PARAMETERS['dx']**2))
        st.sidebar.markdown(f"Courant Number: {courant:.3f}")
    else:
        courant_x = np.sqrt(params["D"] * DEFAULT_PARAMETERS['dt'] / 
                           (params["tau"] * DEFAULT_PARAMETERS['dx']**2))
        courant_y = np.sqrt(params["D"] * DEFAULT_PARAMETERS['dt'] / 
                           (params["tau"] * DEFAULT_PARAMETERS['dy']**2))
        st.sidebar.markdown(f"Courant Number (x): {courant_x:.3f}")
        st.sidebar.markdown(f"Courant Number (y): {courant_y:.3f}")
    
    # Display propagation speed
    propagation_speed = calculate_propagation_speed(params["D"], params["tau"])
    st.sidebar.markdown(f"Propagation Speed: {propagation_speed:.3f}")
