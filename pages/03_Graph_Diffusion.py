"""
Graph Diffusion page for the finite velocity diffusion solver.

This page allows users to explore diffusion processes on graph structures,
including both ordinary diffusion and finite-velocity diffusion, with
visual representations of how these processes evolve on different graph types.
"""

import streamlit as st
from utils import DEFAULT_PARAMETERS
from streamlit_ui.state_management import initialize_session_state
from pages.components.graph_construction import create_graph_construction_ui
from pages.components.diffusion_simulation import create_diffusion_simulation_ui
from pages.components.diffusion_visualization import create_diffusion_visualization_ui

def display_graph_explanation():
    """Display educational information about graphs and diffusion on graphs."""
    with st.expander("Learn about Graphs and Diffusion", expanded=False):
        st.markdown("""
        ## Graphs and Diffusion Processes
        
        ### What are Graphs?
        
        Graphs (or networks) are mathematical structures consisting of nodes (vertices) connected by edges. They are used to model pairwise relations between objects. In our context, graphs provide a discrete domain on which diffusion processes can occur.
        
        ### Types of Graphs
        
        - **Grid (Lattice)**: Regular arrangement of nodes, like a mesh or grid
        - **Triangular Lattice**: Each face is a triangle
        - **Hexagonal Lattice**: Each face is a hexagon
        - **Erdős–Rényi**: Random graph where each possible edge has equal probability of being created
        - **Barabási–Albert**: Scale-free network following preferential attachment (leads to power-law degree distribution)
        
        ### Diffusion on Graphs
        
        Diffusion on graphs is analogous to diffusion in continuous space, but occurs along the edges of the graph. The diffusion equation on a graph uses the graph Laplacian operator instead of the continuous Laplacian.
        
        #### Ordinary Diffusion (Heat Equation)
        
        The ordinary diffusion equation on graphs is:
        
        ```
        ∂u/∂t = D ∇²ᵍ u
        ```
        
        where ∇²ᵍ is the graph Laplacian.
        
        #### Finite-Velocity Diffusion (Telegrapher's Equation)
        
        The finite-velocity diffusion equation on graphs is:
        
        ```
        τ ∂²u/∂t² + ∂u/∂t = D ∇²ᵍ u
        ```
        
        This equation ensures that diffusion occurs with a finite propagation speed on the graph.
        
        ### Graph Laplacian
        
        The graph Laplacian is defined as:
        
        ```
        L = D - A
        ```
        
        where D is the degree matrix (diagonal matrix with node degrees) and A is the adjacency matrix of the graph.
        
        For weighted graphs, the weighted Laplacian uses the weighted adjacency matrix.
        """)

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

# Sidebar for graph construction and diffusion simulation
with st.sidebar:
    create_graph_construction_ui()
    st.markdown("---")
    create_diffusion_simulation_ui()

# Main area for visualization
create_diffusion_visualization_ui()

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
