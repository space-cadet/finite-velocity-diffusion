"""
Graph construction UI components.

This module provides UI components for creating and configuring
different types of graph structures.
"""

import streamlit as st
import networkx as nx
import numpy as np
from typing import Dict, Any, Optional, Tuple

# Define graph types
GRAPH_TYPES = {
    "Grid (2D Lattice)": "grid",
    "Triangular Lattice": "triangular",
    "Hexagonal Lattice": "hexagonal",
    "Random (Erdős–Rényi)": "erdos_renyi",
    "Scale-Free (Barabási–Albert)": "barabasi_albert",
    "Custom": "custom"
}

def initialize_node_values(G: nx.Graph, initialization_type="center_peak"):
    """
    Initialize node values for visualization.
    
    Creates node value dictionary based on the specified initialization type.
    
    Parameters:
    -----------
    G : nx.Graph
        The graph for which to initialize node values
    initialization_type : str
        Type of initialization to use
    """
    # Import here to avoid circular imports
    from solver.graph_solver import create_gaussian_initial_condition
    
    # Initialize all nodes to zero
    # Use consistent string representation for node keys to ensure compatibility
    st.session_state.node_values = {node: 0.0 for node in G.nodes()}
    
    # Also initialize random node colors for visualization consistency
    st.session_state.random_node_colors = {node: np.random.random() for node in G.nodes()}
    
    # Get list of nodes to find center node (or something close to center)
    node_list = list(G.nodes())
    if node_list:
        center_idx = len(node_list) // 2
        center_node = node_list[center_idx]
        
        if initialization_type == "center_peak":
            # Single peak at center
            st.session_state.node_values[center_node] = 1.0
        elif initialization_type == "gaussian":
            # Gaussian distribution centered at center node
            if "graph_pos" in st.session_state:
                st.session_state.node_values = create_gaussian_initial_condition(
                    G, center_node, sigma=2.0, amplitude=1.0, pos=st.session_state.graph_pos
                )
            else:
                st.session_state.node_values = create_gaussian_initial_condition(
                    G, center_node, sigma=2.0, amplitude=1.0
                )
        elif initialization_type == "random":
            # Random values
            st.session_state.node_values = {node: np.random.random() for node in G.nodes()}
    
    # Store center node for future reference (useful for diffusion simulation)
    st.session_state.center_node = center_node

def create_graph_construction_ui():
    """
    Create the graph construction UI components.
    
    This function creates UI controls for selecting and configuring graph types.
    Designed to be displayed in the sidebar.
    """
    st.header("Graph Construction")
    
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
                
                # Initialize node values for the new graph
                initialize_node_values(st.session_state.graph)
                
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
