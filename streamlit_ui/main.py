"""
Main content UI components for the Streamlit finite velocity diffusion app.
"""

import streamlit as st

def create_main_content():
    """
    Create and render the main content area with title and descriptions.
    """
    st.title("Finite Velocity Diffusion Solver")
    st.markdown("""
    This app solves the finite velocity diffusion equation (telegrapher's equation):
    τ∂²u/∂t² + ∂u/∂t = D∇²u

    The solution is compared with the classical diffusion equation for a Gaussian initial condition.
    """)

def display_equations_info():
    """
    Display information about the equations and their comparison.
    """
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

def create_progress_indicators():
    """
    Create progress indicators for the simulation.
    
    Returns:
    --------
    tuple
        Tuple containing (progress_bar, status_text)
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    return progress_bar, status_text
