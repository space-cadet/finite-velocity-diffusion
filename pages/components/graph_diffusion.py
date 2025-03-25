"""
Components for the Graph Diffusion page.

This module provides the UI components for the Graph Diffusion page,
including graph construction, visualization, and diffusion simulations.
"""

import streamlit as st
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List, Union

def initialize_node_values(G: nx.Graph):
    """
    Initialize node values for diffusion visualization.
    
    Creates a default node value dictionary with a single peak value
    at approximately the center of the graph.
    
    Parameters:
    -----------
    G : nx.Graph
        The graph for which to initialize node values
    """
    # Initialize all nodes to zero
    st.session_state.node_values = {node: 0.0 for node in G.nodes()}
    
    # Also initialize random node colors for visualization consistency
    st.session_state.random_node_colors = {node: np.random.random() for node in G.nodes()}
    
    # Get list of nodes to find center node (or something close to center)
    node_list = list(G.nodes())
    if node_list:
        center_idx = len(node_list) // 2
        center_node = node_list[center_idx]
        st.session_state.node_values[center_node] = 1.0

# Define graph types
GRAPH_TYPES = {
    "Grid (2D Lattice)": "grid",
    "Triangular Lattice": "triangular",
    "Hexagonal Lattice": "hexagonal",
    "Random (Erdős–Rényi)": "erdos_renyi",
    "Scale-Free (Barabási–Albert)": "barabasi_albert",
    "Custom": "custom"
}

def create_graph_construction_tab():
    """
    Create the graph construction tab UI components.
    
    This function creates UI controls for selecting and configuring graph types.
    Designed to be displayed in the sidebar.
    """
    
    # Graph type selection
    graph_type = st.selectbox(
        "Graph Type",
        list(GRAPH_TYPES.keys()),
        index=0,
        key="graph_type"
    )
    
    # Initialize session state with default graph if needed
    if "graph" not in st.session_state:
        # Initialize with a small grid graph by default
        rows, cols = 5, 5
        st.session_state.graph = nx.grid_2d_graph(rows, cols)
        st.session_state.graph_pos = {(i, j): [i, j] for i in range(rows) for j in range(cols)}
        
        # Show this as the currently selected type
        if "graph_type" not in st.session_state:
            st.session_state.graph_type = list(GRAPH_TYPES.keys())[0]  # Grid (2D Lattice)
        
        # Initialize node values for visualization
        initialize_node_values(st.session_state.graph)
        
        st.info(f"Initialized with a default {rows}x{cols} grid graph.")
    
    # Parameters for different graph types
    if GRAPH_TYPES[graph_type] == "grid":
        with st.expander("Grid Parameters", expanded=False):
            rows = st.slider("Rows", min_value=2, max_value=20, value=5, key="grid_rows")
            cols = st.slider("Columns", min_value=2, max_value=20, value=5, key="grid_cols")
            
            if st.button("Generate Grid Graph"):
                st.session_state.graph = nx.grid_2d_graph(rows, cols)
                st.session_state.graph_pos = {(i, j): [i, j] for i in range(rows) for j in range(cols)}
                
                # Initialize node values for the new graph
                initialize_node_values(st.session_state.graph)
                
                st.success(f"Generated a {rows}x{cols} grid graph with {rows * cols} nodes and {len(st.session_state.graph.edges)} edges")
    
    elif GRAPH_TYPES[graph_type] == "triangular":
        with st.expander("Triangular Lattice Parameters", expanded=False):
            size = st.slider("Size", min_value=2, max_value=15, value=5, key="triangular_size")
            
            if st.button("Generate Triangular Lattice"):
                # Create a triangular lattice
                st.session_state.graph = nx.triangular_lattice_graph(size, size)
                
                # Use spring layout for positioning
                st.session_state.graph_pos = nx.spring_layout(st.session_state.graph, seed=42)
                
                # Initialize node values for the new graph
                initialize_node_values(st.session_state.graph)
                
                st.success(f"Generated a triangular lattice with {len(st.session_state.graph.nodes)} nodes and {len(st.session_state.graph.edges)} edges")
    
    elif GRAPH_TYPES[graph_type] == "hexagonal":
        with st.expander("Hexagonal Lattice Parameters", expanded=False):
            size = st.slider("Size", min_value=2, max_value=10, value=3, key="hexagonal_size")
            
            if st.button("Generate Hexagonal Lattice"):
                # Create a hexagonal lattice
                st.session_state.graph = nx.hexagonal_lattice_graph(size, size)
                
                # Use spring layout for positioning
                st.session_state.graph_pos = nx.spring_layout(st.session_state.graph, seed=42)
                
                st.success(f"Generated a hexagonal lattice with {len(st.session_state.graph.nodes)} nodes and {len(st.session_state.graph.edges)} edges")
    
    elif GRAPH_TYPES[graph_type] == "erdos_renyi":
        with st.expander("Erdős–Rényi Parameters", expanded=False):
            n = st.slider("Number of Nodes", min_value=5, max_value=100, value=30, key="er_nodes")
            p = st.slider("Connection Probability", min_value=0.0, max_value=1.0, value=0.15, step=0.05, key="er_prob")
            
            if st.button("Generate Erdős–Rényi Graph"):
                st.session_state.graph = nx.erdos_renyi_graph(n, p, seed=42)
                st.session_state.graph_pos = nx.spring_layout(st.session_state.graph, seed=42)
                
                # Initialize node values for the new graph
                initialize_node_values(st.session_state.graph)
                
                st.success(f"Generated an Erdős–Rényi graph with {n} nodes and {len(st.session_state.graph.edges)} edges")
    
    elif GRAPH_TYPES[graph_type] == "barabasi_albert":
        with st.expander("Barabási–Albert Parameters", expanded=False):
            n = st.slider("Number of Nodes", min_value=5, max_value=100, value=30, key="ba_nodes")
            m = st.slider("Number of Edges to Attach from New Node", min_value=1, max_value=10, value=2, key="ba_edges")
            
            if st.button("Generate Barabási–Albert Graph"):
                st.session_state.graph = nx.barabasi_albert_graph(n, m, seed=42)
                st.session_state.graph_pos = nx.spring_layout(st.session_state.graph, seed=42)
                
                # Initialize node values for the new graph
                initialize_node_values(st.session_state.graph)
                
                st.success(f"Generated a Barabási–Albert graph with {n} nodes and {len(st.session_state.graph.edges)} edges")
    
    elif GRAPH_TYPES[graph_type] == "custom":
        with st.expander("Custom Graph", expanded=False):
            st.markdown("""
            Create a custom graph by specifying nodes and edges:
            1. Enter node names (comma-separated)
            2. Enter edges as pairs of nodes (comma-separated)
            
            Example:
            - Nodes: A, B, C, D
            - Edges: A-B, B-C, C-D, D-A
            """)
            
            nodes_input = st.text_input("Nodes (comma-separated)", "A, B, C, D, E", key="custom_nodes")
            edges_input = st.text_input("Edges (format: A-B, B-C, ...)", "A-B, B-C, C-D, D-E, E-A", key="custom_edges")
            
            if st.button("Generate Custom Graph"):
                try:
                    # Parse nodes
                    nodes = [node.strip() for node in nodes_input.split(",")]
                    
                    # Parse edges
                    edge_pairs = []
                    for edge in edges_input.split(","):
                        if "-" in edge:
                            source, target = edge.strip().split("-")
                            edge_pairs.append((source.strip(), target.strip()))
                    
                    # Create graph
                    G = nx.Graph()
                    G.add_nodes_from(nodes)
                    G.add_edges_from(edge_pairs)
                    
                    st.session_state.graph = G
                    st.session_state.graph_pos = nx.spring_layout(G, seed=42)
                    
                    # Initialize node values for the new graph
                    initialize_node_values(st.session_state.graph)
                    
                    st.success(f"Generated a custom graph with {len(nodes)} nodes and {len(edge_pairs)} edges")
                except Exception as e:
                    st.error(f"Error creating custom graph: {str(e)}")
    
    # Display graph info
    if "graph" in st.session_state:
        st.markdown("### Graph Information")
        st.write(f"**Nodes:** {st.session_state.graph.number_of_nodes()}")
        st.write(f"**Edges:** {st.session_state.graph.number_of_edges()}")
        st.write(f"**Avg. Degree:** {np.mean([d for n, d in st.session_state.graph.degree()]):.2f}")

def create_graph_visualization_tab():
    """
    Create the graph visualization UI components.
    
    This function creates UI controls for visualizing graphs and
    selecting visualization options.
    """
    st.header("Graph Visualization")
    
    if "graph" not in st.session_state:
        st.warning("Please create a graph using the controls in the sidebar first.")
        return
    
    # Import visualization functions
    from visualization.graph_visualization import create_3d_graph_visualization
    
    # Visualization options
    viz_type = st.radio(
        "Visualization Type",
        ["2D Static", "2D Interactive", "3D Interactive"],
        key="viz_type"
    )
    
    # Node color options
    color_by = st.selectbox(
        "Color Nodes By",
        ["Degree", "Random", "Centrality", "Constant"],
        key="color_by"
    )
    
    color_scale = st.selectbox(
        "Color Scale",
        ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Blues", "Reds", "Greens"],
        index=0,
        key="color_scale"
    )
    
    # Calculate node colors
    if color_by == "Degree":
        node_degrees = dict(st.session_state.graph.degree())
        node_colors = {node: node_degrees[node] for node in st.session_state.graph.nodes()}
        color_title = "Node Degree"
    elif color_by == "Random":
        # Use session state to keep colors consistent
        if "random_node_colors" not in st.session_state:
            st.session_state.random_node_colors = {node: np.random.random() for node in st.session_state.graph.nodes()}
        node_colors = st.session_state.random_node_colors
        color_title = "Random Value"
    elif color_by == "Centrality":
        # Calculate betweenness centrality
        centrality = nx.betweenness_centrality(st.session_state.graph)
        node_colors = centrality
        color_title = "Centrality"
    else:  # Constant
        node_colors = {node: 0.5 for node in st.session_state.graph.nodes()}
        color_title = "Constant Value"
    
    # Visualization settings
    with st.expander("Visualization Settings", expanded=False):
        node_size = st.slider("Node Size", min_value=5, max_value=30, value=10, key="node_size")
        edge_width = st.slider("Edge Width", min_value=1, max_value=5, value=1, key="edge_width")
        
        if viz_type == "3D Interactive":
            height = st.slider("Chart Height", min_value=400, max_value=1000, value=600, key="chart_height")
        else:
            height = None
    
    # Visualize the graph
    if viz_type == "2D Static":
        # Plot with Matplotlib
        fig = visualize_graph_matplotlib(st.session_state.graph, st.session_state.graph_pos, node_colors)
        st.pyplot(fig)
    elif viz_type == "2D Interactive":
        # Plot with Plotly
        fig = visualize_graph_plotly(st.session_state.graph, st.session_state.graph_pos, node_colors, color_scale, color_title)
        st.plotly_chart(fig, use_container_width=True, height=height or 600)
    else:  # 3D Interactive
        # 3D plot with Plotly
        fig = create_3d_graph_visualization(st.session_state.graph, st.session_state.graph_pos, node_colors, color_scale, "3D Graph Visualization")
        st.plotly_chart(fig, use_container_width=True, height=height or 700)
    
    # Add graph metrics
    with st.expander("Graph Metrics", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Nodes", st.session_state.graph.number_of_nodes())
            st.metric("Connected Components", nx.number_connected_components(st.session_state.graph))
            
            # Check if graph is connected before calculating diameter
            if nx.is_connected(st.session_state.graph):
                st.metric("Diameter", nx.diameter(st.session_state.graph))
            else:
                st.metric("Diameter", "N/A (Disconnected)")
                
        with col2:
            st.metric("Edges", st.session_state.graph.number_of_edges())
            st.metric("Average Degree", f"{np.mean([d for n, d in st.session_state.graph.degree()]):.2f}")
            st.metric("Density", f"{nx.density(st.session_state.graph):.4f}")

def visualize_graph_plotly(
    G: nx.Graph,
    pos: Dict,
    node_colors: Dict,
    color_scale: str = "Viridis",
    color_title: str = "Value"
) -> go.Figure:
    """
    Visualize a graph using Plotly.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to visualize
    pos : Dict
        Dictionary of node positions
    node_colors : Dict
        Dictionary of node color values
    color_scale : str
        Plotly color scale name
    color_title : str
        Title for the color scale
    
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Create edge trace
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Add the coordinates of each edge point
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_x = []
    node_y = []
    node_color = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(node_colors[node])
        
        # Add node information for hover
        if isinstance(node, tuple):
            node_label = f"({node[0]}, {node[1]})"
        else:
            node_label = str(node)
        
        node_text.append(f"Node: {node_label}<br>Degree: {G.degree(node)}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale=color_scale,
            color=node_color,
            size=10,
            colorbar=dict(
                thickness=15,
                title=color_title,
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)
        )
    )
    
    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f'Graph Visualization',
            titlefont=dict(size=16),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white"
        )
    )
    
    return fig

def visualize_graph_matplotlib(
    G: nx.Graph,
    pos: Dict,
    node_colors: Dict
) -> plt.Figure:
    """
    Visualize a graph using Matplotlib.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph to visualize
    pos : Dict
        Dictionary of node positions
    node_colors : Dict
        Dictionary of node color values
    
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get color values as list
    color_values = [node_colors[node] for node in G.nodes()]
    
    # Draw the graph
    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=True,
        node_color=color_values,
        cmap=plt.cm.viridis,
        node_size=300,
        edge_color='#888888',
        width=1,
        alpha=0.7,
        ax=ax
    )
    
    # Customize plot
    plt.title("Graph Visualization")
    plt.axis('off')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(min(color_values), max(color_values)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Node Value")
    
    return fig

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
