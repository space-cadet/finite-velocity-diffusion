"""
Diffusion visualization UI components.

This module provides UI components for visualizing and interacting with
diffusion results on graph structures.
"""

import streamlit as st
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, Any, Optional, Tuple

def create_diffusion_visualization_ui():
    """
    Create the UI components for visualizing diffusion results.
    
    This function creates UI controls for visualizing diffusion on graphs
    and interacting with time evolution data.
    """
    st.header("Graph Visualization")
    
    if "graph" not in st.session_state:
        st.warning("Please create a graph using the controls in the sidebar first.")
        return
    
    # Import necessary visualization functions
    from visualization.graph_diffusion_visualization import (
        visualize_diffusion_2d,
        visualize_diffusion_3d
    )
    
    # Check if we're visualizing diffusion results
    if "show_diffusion" in st.session_state and st.session_state.show_diffusion:
        _create_diffusion_result_ui()
    else:
        _create_graph_structure_ui()
        
def _create_diffusion_result_ui():
    """
    Create UI components for displaying diffusion simulation results.
    """
    st.subheader("Diffusion Simulation Results")
    
    # Initialize animation-related session state variables
    _initialize_animation_state()
    
    # Time point selection - only if we have multiple time points
    if "diffusion_time_points" in st.session_state and st.session_state.diffusion_time_points:
        # Check if we have more than one time point
        if len(st.session_state.diffusion_time_points) > 1:
            time_index = st.slider(
                "Time Step",
                min_value=0,
                max_value=len(st.session_state.diffusion_time_points) - 1,
                value=st.session_state.current_time_index if "current_time_index" in st.session_state else 0,
                key="time_slider"
            )
            
            st.session_state.current_time_index = time_index
            current_time = st.session_state.diffusion_time_points[time_index]
            
            # Update node values for visualization
            st.session_state.node_values = st.session_state.diffusion_solutions[current_time]
            
            st.write(f"Displaying solution at t = {current_time:.3f}")
        else:
            # Only one time point (initial condition)
            st.info("Displaying initial condition (t = 0.0)")
            st.session_state.current_time_index = 0
            current_time = st.session_state.diffusion_time_points[0]
        
        # Animation controls - only show if we have multiple time points
        if len(st.session_state.diffusion_time_points) > 1:
            # Create placeholders for animation components
            st.session_state.animation_status = st.empty()
            st.session_state.animation_progress = st.empty()
            st.session_state.animation_container = st.container()
            
            # Add animation speed control
            animation_speed = st.slider(
                "Animation Speed", 
                min_value=0.05, 
                max_value=1.0, 
                value=st.session_state.animation_speed,
                step=0.05,
                format="%.2f s",
                help="Time delay between frames in seconds (smaller = faster)"
            )
            st.session_state.animation_speed = animation_speed
            
            # Create animation control buttons
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                start_button = st.button("▶️ Start")
            with col2:
                pause_button = st.button("⏸️ Pause")
            with col3:
                stop_button = st.button("⏹️ Stop")
            with col4:
                reset_button = st.button("🔄 Reset")
            
            # Handle button clicks
            if start_button:
                # Store current visualization type to preserve it during animation
                if "viz_type" in st.session_state:
                    st.session_state.animation_viz_type = st.session_state.viz_type
                st.session_state.animation_running = True
                
            if pause_button:
                st.session_state.animation_running = False
                
            if stop_button:
                st.session_state.animation_running = False
                st.session_state.animation_frame = len(st.session_state.diffusion_time_points) - 1
                # Update to the final frame
                if st.session_state.diffusion_time_points:
                    t = st.session_state.diffusion_time_points[st.session_state.animation_frame]
                    st.session_state.current_time_index = st.session_state.animation_frame
                    st.session_state.node_values = st.session_state.diffusion_solutions[t]
                    # Trigger an update of the display
                    st.rerun()
                
            if reset_button:
                st.session_state.animation_running = False
                st.session_state.animation_frame = 0
                # Update to the first frame
                if st.session_state.diffusion_time_points:
                    t = st.session_state.diffusion_time_points[0]
                    st.session_state.current_time_index = 0
                    st.session_state.node_values = st.session_state.diffusion_solutions[t]
                    # Trigger an update of the display
                    st.rerun()
            
            # Define parameters for animation
            viz_type = st.session_state.get("animation_viz_type", st.session_state.get("viz_type", "2D Interactive"))
            color_scale = st.session_state.get("color_scale", "Viridis")
            node_size = st.session_state.get("node_size", 10)
            edge_width = st.session_state.get("edge_width", 1)
            height = st.session_state.get("chart_height", 600)
            
            # Display animation status
            total_frames = len(st.session_state.diffusion_time_points)
            current_frame = st.session_state.animation_frame
            progress = current_frame / (total_frames - 1) if total_frames > 1 else 1.0
            
            # Update progress indicators
            if current_frame < total_frames:
                t = st.session_state.diffusion_time_points[current_frame]
                st.session_state.animation_status.text(f"Time: t = {t:.3f} (Frame {current_frame+1}/{total_frames})")
                st.session_state.animation_progress.progress(progress)
            
            # Prepare container for the visualization
            animation_container = st.session_state.animation_container
            
            # Function to display current frame
            def display_current_frame():
                if current_frame < total_frames:
                    t = st.session_state.diffusion_time_points[current_frame]
                    
                    # Update node values for the current frame
                    current_values = st.session_state.diffusion_solutions[t]
                    
                    # Display the visualization for this frame
                    with animation_container:
                        if viz_type == "2D Static":
                            # Plot with Matplotlib
                            fig = _visualize_graph_matplotlib(
                                st.session_state.graph, 
                                st.session_state.graph_pos, 
                                current_values
                            )
                            st.pyplot(fig)
                        elif viz_type == "2D Interactive":
                            # Plot with Plotly
                            from visualization.graph_diffusion_visualization import visualize_diffusion_2d
                            fig = visualize_diffusion_2d(
                                st.session_state.graph, 
                                st.session_state.graph_pos, 
                                current_values, 
                                color_scale, 
                                f"Diffusion on Graph (t = {t:.3f})", 
                                node_size,
                                edge_width
                            )
                            st.plotly_chart(fig, use_container_width=True, height=height)
                        else:  # 3D Interactive
                            # Plot with Plotly
                            from visualization.graph_diffusion_visualization import visualize_diffusion_3d
                            fig = visualize_diffusion_3d(
                                st.session_state.graph, 
                                st.session_state.graph_pos, 
                                current_values, 
                                color_scale, 
                                f"Diffusion on Graph (t = {t:.3f})", 
                                node_size,
                                edge_width
                            )
                            st.plotly_chart(fig, use_container_width=True, height=height)
            
            # Display the current frame
            display_current_frame()
            
            # Update animation frame if running
            if st.session_state.animation_running and current_frame < total_frames - 1:
                # Advance to next frame
                st.session_state.animation_frame += 1
                
                # Update current time index and node values
                next_frame = st.session_state.animation_frame
                if next_frame < total_frames:
                    t = st.session_state.diffusion_time_points[next_frame]
                    st.session_state.current_time_index = next_frame
                    st.session_state.node_values = st.session_state.diffusion_solutions[t]
                
                # Rerun to update the visualization with controlled speed
                time.sleep(st.session_state.animation_speed)  # Use the animation speed control
                st.rerun()
            elif st.session_state.animation_running and current_frame >= total_frames - 1:
                # Animation has finished
                st.session_state.animation_running = False
                st.session_state.animation_status.text("Animation completed.")
        
        # Option to reset visualization to structure view
        if st.button("Reset to Graph Structure View"):
            st.session_state.show_diffusion = False
            st.rerun()
    
    # Visualization options
    _create_visualization_options(is_diffusion=True)
    
def _initialize_animation_state():
    """
    Initialize or reset animation-related session state variables.
    """
    # Core animation state
    if "animation_running" not in st.session_state:
        st.session_state.animation_running = False
    if "animation_frame" not in st.session_state:
        st.session_state.animation_frame = 0
    
    # Animation UI components - these will be populated during rendering
    if "animation_container" not in st.session_state:
        st.session_state.animation_container = None
    if "animation_status" not in st.session_state:
        st.session_state.animation_status = None
    if "animation_progress" not in st.session_state:
        st.session_state.animation_progress = None
    
    # Animation settings
    if "animation_speed" not in st.session_state:
        st.session_state.animation_speed = 0.2  # default delay between frames

def _create_graph_structure_ui():
    """
    Create UI components for displaying graph structure visualization.
    """
    # Visualization options
    _create_visualization_options(is_diffusion=False)
    
def _create_visualization_options(is_diffusion: bool):
    """
    Create common visualization options.
    
    Parameters:
    -----------
    is_diffusion : bool
        Whether we're visualizing diffusion results (True) or just structure (False)
    """
    # Import visualization functions
    from visualization.graph_diffusion_visualization import visualize_diffusion_2d, visualize_diffusion_3d
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Get the default visualization type from various session state sources
        # Priority: saved_viz_type (from simulation) > animation_viz_type > previous viz_type > default
        if "saved_viz_type" in st.session_state:
            default_viz_type = st.session_state.saved_viz_type
            # Clear after using it
            del st.session_state.saved_viz_type
        elif "animation_viz_type" in st.session_state:
            default_viz_type = st.session_state.animation_viz_type
            # Clear after using it
            del st.session_state.animation_viz_type
        else:
            default_viz_type = "2D Interactive"
        
        # Ensure the default_viz_type is valid
        if default_viz_type not in ["2D Static", "2D Interactive", "3D Interactive"]:
            default_viz_type = "2D Interactive"
        
        # Create the radio selector with the appropriate default
        viz_type = st.radio(
            "Visualization Type",
            ["2D Static", "2D Interactive", "3D Interactive"],
            key="viz_type",
            index=["2D Static", "2D Interactive", "3D Interactive"].index(default_viz_type)
        )
    
    with col2:
        # Node color options
        if is_diffusion:
            # When viewing diffusion results, we always color by node values
            color_by = "Node Values"
            st.info("Nodes colored by diffusion values")
        else:
            color_by = st.selectbox(
                "Color Nodes By",
                ["Degree", "Random", "Centrality", "Constant"],
                key="color_by"
            )
    
    # Color scale selection
    color_scale = st.selectbox(
        "Color Scale",
        ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Blues", "Reds", "Greens"],
        index=0,
        key="color_scale"
    )
    
    # Calculate node colors
    if is_diffusion:
        # Use diffusion values for colors
        node_colors = st.session_state.node_values
        color_title = "Diffusion Value"
    else:
        # Use graph properties for colors
        if color_by == "Degree":
            node_degrees = dict(st.session_state.graph.degree())
            node_colors = {node: node_degrees[node] for node in st.session_state.graph.nodes()}
            color_title = "Node Degree"
        elif color_by == "Random":
            # Use session state to keep colors consistent
            if "random_node_colors" not in st.session_state:
                st.session_state.random_node_colors = {node: np.random.random() 
                                                    for node in st.session_state.graph.nodes()}
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
    
    # Display the appropriate visualization based on type
    _display_visualization(viz_type, node_colors, color_scale, color_title, node_size, edge_width, height)
    
    # Add graph metrics
    _display_graph_metrics()

def _display_visualization(
    viz_type: str,
    node_colors: Dict,
    color_scale: str,
    color_title: str,
    node_size: int,
    edge_width: int,
    height: Optional[int]
):
    """
    Display the graph visualization with the specified options.
    
    Parameters:
    -----------
    viz_type : str
        Type of visualization (2D Static, 2D Interactive, 3D Interactive)
    node_colors : Dict
        Dictionary of node color values
    color_scale : str
        Color scale name
    color_title : str
        Title for the color scale
    node_size : int
        Size of nodes in the visualization
    edge_width : int
        Width of edges in the visualization
    height : Optional[int]
        Height of the visualization (for interactive plots)
    """
    from visualization.graph_diffusion_visualization import visualize_diffusion_2d, visualize_diffusion_3d
    
    is_diffusion = "show_diffusion" in st.session_state and st.session_state.show_diffusion
    is_animating = "animate" in st.session_state and st.session_state.animate
    
    # Set title based on whether we're showing diffusion or structure
    if is_diffusion:
        diffusion_type = st.session_state.get('diffusion_type', 'Diffusion')
        title = f"Diffusion on Graph ({diffusion_type})"
    else:
        title = "Graph Structure"
    
    # Add time info to title if showing diffusion results
    if is_diffusion and "current_time_index" in st.session_state and "diffusion_time_points" in st.session_state:
        time_point = st.session_state.diffusion_time_points[st.session_state.current_time_index]
        title += f" - t = {time_point:.3f}"
    
    # For 2D Static view
    if viz_type == "2D Static":
        # Plot with Matplotlib
        fig = _visualize_graph_matplotlib(st.session_state.graph, st.session_state.graph_pos, node_colors)
        st.pyplot(fig)
    
    # For 2D Interactive view
    elif viz_type == "2D Interactive":
        # Plot with Plotly
        fig = visualize_diffusion_2d(
            st.session_state.graph, 
            st.session_state.graph_pos, 
            node_colors, 
            color_scale, 
            title,
            node_size,
            edge_width
        )
        st.plotly_chart(fig, use_container_width=True, height=height or 600)
    
    # For 3D Interactive view
    else:  # 3D Interactive
        # 3D plot with Plotly
        fig = visualize_diffusion_3d(
            st.session_state.graph, 
            st.session_state.graph_pos, 
            node_colors, 
            color_scale, 
            title,
            node_size,
            edge_width
        )
        st.plotly_chart(fig, use_container_width=True, height=height or 700)

def _display_graph_metrics():
    """
    Display graph metrics in an expandable section.
    """
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

def _visualize_graph_matplotlib(
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
