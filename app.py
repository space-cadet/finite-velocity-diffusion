"""
Home page for the finite velocity diffusion solver multipage app.

This app allows users to explore solutions to the finite velocity diffusion
equation (telegrapher's equation) in both 1D and 2D, and compare them with
classical diffusion solutions.

Parameter settings are persisted between sessions using file-based storage.
"""

import streamlit as st

# Local imports
from streamlit_ui.sidebar import create_sidebar_controls, display_simulation_info
from streamlit_ui.state_management import initialize_session_state
from streamlit_ui.main import create_main_content, create_progress_indicators, display_equations_info
from simulation.streamlit_simulation import run_1d_simulation, run_2d_simulation
from visualization.streamlit_visualization import (
    display_1d_results, 
    display_2d_results,
    display_1d_time_evolution_static,
    display_1d_time_evolution_slider,
    display_1d_time_evolution_animation,
    display_2d_time_evolution_static,
    display_2d_time_evolution_slider,
    display_2d_time_evolution_animation
)

# Configure the Streamlit page
st.set_page_config(
    page_title="Finite Velocity Diffusion Solver",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create the main UI components
create_main_content()
st.markdown("---")
st.info("This is a multipage app. Use the navigation menu in the sidebar to explore additional features.")

# Get parameters from sidebar
params = create_sidebar_controls()

# Create progress indicators
progress_bar, status_text = create_progress_indicators()

# Create a progress callback function
def update_progress(progress, message):
    progress_bar.progress(progress)
    status_text.text(message)

# Run the simulation and display results
if params['dimension'] == "1D":
    # Run 1D simulation
    results = run_1d_simulation(params, progress_callback=update_progress)
    
    # Display 1D results
    display_1d_results(
        x=results['x'],
        u_finite=results['u'],
        u_classical=results['u_classical'],
        title='1D Finite Velocity vs Classical Diffusion'
    )
    
    # Display time evolution if requested
    if params['show_evolution']:
        if params['viz_type'] == "Static Plots":
            # Display static plots
            display_1d_time_evolution_static(
                times=results['times'],
                solutions=results['solutions'],
                x=results['x']
            )
            
        elif params['viz_type'] == "Interactive Slider":
            # Display interactive slider
            display_1d_time_evolution_slider(
                times=results['times'],
                solutions=results['solutions'],
                x=results['x']
            )
            
        elif params['viz_type'] == "Plotly Animation":
            # Display Plotly animation
            display_1d_time_evolution_animation(
                times=results['times'],
                solutions=results['solutions'],
                x=results['x'],
                animation_speed=params.get('animation_speed', 10)
            )
else:  # 2D case
    # Run 2D simulation
    results = run_2d_simulation(params, progress_callback=update_progress)
    
    # Display 2D results if not showing evolution
    if not params['show_evolution']:
        display_2d_results(
            X=results['X'],
            Y=results['Y'],
            u=results['u'],
            title='2D Finite Velocity Diffusion'
        )
    
    # Display time evolution if requested
    if params['show_evolution']:
        if params['viz_type'] == "Static Plots":
            # Display static plots
            display_2d_time_evolution_static(
                times=results['times'],
                solutions=results['solutions'],
                X=results['X'],
                Y=results['Y']
            )
            
        elif params['viz_type'] == "Interactive Slider":
            # Display interactive slider
            display_2d_time_evolution_slider(
                times=results['times'],
                solutions=results['solutions'],
                X=results['X'],
                Y=results['Y']
            )
            
        elif params['viz_type'] == "Plotly Animation":
            # Display Plotly animation
            display_2d_time_evolution_animation(
                times=results['times'],
                solutions=results['solutions'],
                x=results['x'],
                y=results['y'],
                animation_speed=params.get('animation_speed', 10)
            )

# Display simulation information in sidebar
display_simulation_info(params)

# Display equations and information at the bottom
display_equations_info()
