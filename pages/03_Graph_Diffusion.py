"""
Graph Diffusion page for the finite velocity diffusion solver.

This page allows users to explore diffusion processes on graph structures,
including both ordinary diffusion and finite-velocity diffusion, with
visual representations of how these processes evolve on different graph types.
"""

import streamlit as st
from utils import DEFAULT_PARAMETERS
from streamlit_ui.state_management import initialize_session_state
from pages.components.graph_diffusion import (
    create_graph_construction_tab,
    create_graph_visualization_tab,
    display_graph_explanation
)

# Configure the page
st.set_page_config(
    page_title="Graph Diffusion - Finite Velocity Diffusion",
    page_icon="🔄",
    layout="wide"
)

# Initialize session state with saved parameters
saved_params = initialize_session_state()

# Page Title
st.title("Graph Diffusion: Exploring Diffusion on Networks")
st.markdown("""
This page allows you to explore diffusion processes on graph structures.
You can create different types of graphs, visualize them, and explore how
diffusion processes evolve on these structures.
""")

# Sidebar for graph construction
with st.sidebar:
    st.header("Graph Construction")
    create_graph_construction_tab()

# Main area for visualization
create_graph_visualization_tab()

# Display educational information about graph diffusion
st.markdown("---")
display_graph_explanation()

# Footer
st.markdown("---")
st.markdown("""
**Note**: This page focuses on exploring diffusion processes on graph structures. 
You can create different types of graphs, visualize them, and see how diffusion 
processes behave on these structures.
""")
