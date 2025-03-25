"""
Parameter Analysis page for the finite velocity diffusion solver.

This page allows users to explore how different parameters affect the
properties of finite velocity diffusion, such as propagation speed,
wave vs. diffusion behavior, and stability conditions.
"""

import streamlit as st
from utils import DEFAULT_PARAMETERS
from streamlit_ui.state_management import initialize_session_state
from pages.components.parameter_analysis import (
    create_propagation_speed_tab,
    create_wave_diffusion_spectrum_tab,
    create_stability_analysis_tab
)

# Configure the page
st.set_page_config(
    page_title="Parameter Analysis - Finite Velocity Diffusion",
    page_icon="📊",
    layout="wide"
)

# Page Title
st.title("Parameter Analysis: Finite Velocity Diffusion")
st.markdown("""
This page provides tools to analyze how different parameters affect the behavior of finite velocity diffusion.
Explore the relationships between diffusion coefficient (D), relaxation time (τ), and the resulting physical properties.
""")

# Initialize session state with saved parameters
saved_params = initialize_session_state()

# Create tabs for different analysis tools
tab1, tab2, tab3 = st.tabs(["Propagation Speed", "Wave-Diffusion Spectrum", "Stability Analysis"])

# Initialize session state with saved parameters
saved_params = initialize_session_state()

with tab1:
    create_propagation_speed_tab(saved_params)

with tab2:
    create_wave_diffusion_spectrum_tab(saved_params)

with tab3:
    create_stability_analysis_tab(DEFAULT_PARAMETERS)
# Footer
st.markdown("---")
st.markdown("""
**Note**: This page focuses on the theoretical aspects of finite velocity diffusion and 
helps build intuition about the parameters. Return to the main page to run simulations.
""")

# Footer
st.markdown("---")
st.markdown("""
**Note**: This page focuses on the theoretical aspects of finite velocity diffusion and 
helps build intuition about the parameters. Return to the main page to run simulations.
""")
