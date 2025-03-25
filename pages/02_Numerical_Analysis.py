"""
Numerical Analysis page for the finite velocity diffusion solver.

This page focuses on comparing numerical solutions with analytical solutions
where available, and analyzing convergence and accuracy of the numerical methods.
"""

import streamlit as st
from utils import DEFAULT_PARAMETERS
from streamlit_ui.state_management import initialize_session_state

# Configure the page
st.set_page_config(
    page_title="Numerical Analysis - Finite Velocity Diffusion",
    page_icon="🧮",
    layout="wide"
)

# Page Title
st.title("Numerical Analysis: Finite Velocity Diffusion")
st.markdown("""
This page focuses on the numerical aspects of solving the finite velocity diffusion equation.
Explore convergence, accuracy, and compare numerical solutions with analytical solutions where available.
""")

# Import the component functions
from pages.components.numerical_analysis import create_convergence_analysis_tab, create_error_analysis_tab

# Initialize session state with saved parameters
saved_params = initialize_session_state()

# Create tabs for different analysis tools
tab1, tab2 = st.tabs(["Convergence Analysis", "Error Analysis"])

with tab1:
    create_convergence_analysis_tab(saved_params)

with tab2:
    create_error_analysis_tab()

# Footer
st.markdown("---")
st.markdown("""
**Note**: This page focuses on numerical properties of the finite difference solver.
For actual simulations, return to the main page.
""")
