"""
Diffusion simulation UI components.

This module provides UI components for configuring and running
diffusion simulations on graph structures.
"""

import streamlit as st
import networkx as nx
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple

from solver.graph_solver import (
    OrdinaryGraphDiffusion,
    FiniteVelocityGraphDiffusion,
    GraphDiffusionWithPotential,
    create_gaussian_initial_condition,
    create_delta_initial_condition
)

def create_diffusion_simulation_ui():
    """
    Create the diffusion simulation UI components.
    
    This function creates UI controls for configuring and running diffusion simulations.
    Designed to be displayed in the sidebar.
    """
    if "graph" not in st.session_state:
        st.warning("Please create a graph first.")
        return
    
    st.header("Diffusion Simulation")
    
    # Diffusion type selection
    diffusion_type = st.selectbox(
        "Diffusion Type",
        ["Ordinary Diffusion", "Finite-Velocity Diffusion", "Diffusion with Potential"],
        key="diffusion_type"
    )
    
    # Common parameters
    with st.expander("Diffusion Parameters", expanded=True):
        # Common parameters for all diffusion types
        diffusion_coefficient = st.slider(
            "Diffusion Coefficient (D)",
            min_value=0.01,
            max_value=2.0,
            value=1.0,
            step=0.01,
            key="diffusion_coefficient"
        )
        
        dt = st.slider(
            "Time Step (dt)",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            key="dt"
        )
        
        # Parameters specific to finite-velocity diffusion
        if diffusion_type == "Finite-Velocity Diffusion":
            relaxation_time = st.slider(
                "Relaxation Time (τ)",
                min_value=0.01,
                max_value=1.0,
                value=0.1,
                step=0.01,
                key="relaxation_time"
            )
        
        # Parameters specific to diffusion with potential
        if diffusion_type == "Diffusion with Potential":
            potential_type = st.selectbox(
                "Potential Type",
                ["Constant", "Node Degree", "Random", "Center Attractor"],
                key="potential_type"
            )
            
            potential_strength = st.slider(
                "Potential Strength",
                min_value=0.0,
                max_value=2.0,
                value=0.5,
                step=0.1,
                key="potential_strength"
            )
    
    # Initial condition options
    with st.expander("Initial Condition", expanded=True):
        ic_type = st.selectbox(
            "Initial Condition Type",
            ["Delta Function", "Gaussian", "Random"],
            key="ic_type"
        )
        
        if ic_type in ["Delta Function", "Gaussian"]:
            # Show node selection if the graph is small enough
            if len(st.session_state.graph.nodes()) <= 30:
                nodes = list(st.session_state.graph.nodes())
                center_node = st.selectbox(
                    "Center Node",
                    nodes,
                    index=min(len(nodes)//2, len(nodes)-1),
                    format_func=lambda x: str(x),
                    key="center_node"
                )
            else:
                st.info("Graph is too large for manual node selection. Using approximate center node.")
                center_node = list(st.session_state.graph.nodes())[len(st.session_state.graph.nodes())//2]
        
        if ic_type == "Gaussian":
            sigma = st.slider(
                "Gaussian Width (σ)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1,
                key="gaussian_sigma"
            )
        
        amplitude = st.slider(
            "Amplitude",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            key="amplitude"
        )
        
        # Apply button for initial condition
        # This allows users to visualize the initial condition without running a full simulation
        if st.button("Apply Initial Condition", key="apply_ic"):
            apply_initial_condition()
    
    # Simulation control
    with st.expander("Simulation Control", expanded=True):
        num_steps = st.slider(
            "Number of Steps",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            key="num_steps"
        )
        
        store_interval = st.slider(
            "Store Interval",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="store_interval",
            help="Store solution every N steps to save memory"
        )
        
        normalize_laplacian = st.checkbox(
            "Normalize Laplacian",
            value=False,
            key="normalize_laplacian",
            help="Use normalized Laplacian for diffusion"
        )
    
    # Run simulation button
    if st.button("Run Simulation", key="run_simulation"):
        run_diffusion_simulation()

def apply_initial_condition():
    """
    Apply the selected initial condition to the graph without running a simulation.
    
    This function creates the initial condition based on user settings and displays it,
    allowing users to visualize the starting point of a diffusion process.
    """
    ic_type = st.session_state.ic_type
    amplitude = st.session_state.amplitude
    
    # Create initial values based on the selected type
    if ic_type == "Delta Function":
        # Use center_node from session state if available
        if "center_node" in st.session_state and st.session_state.center_node in st.session_state.graph.nodes():
            center = st.session_state.center_node
        else:
            # Use first node as fallback
            center = list(st.session_state.graph.nodes())[0]
        
        initial_values = create_delta_initial_condition(
            st.session_state.graph, 
            center, 
            amplitude=amplitude
        )
    elif ic_type == "Gaussian":
        # Use center_node from session state if available
        if "center_node" in st.session_state and st.session_state.center_node in st.session_state.graph.nodes():
            center = st.session_state.center_node
        else:
            # Use first node as fallback
            center = list(st.session_state.graph.nodes())[0]
        
        sigma = st.session_state.get('gaussian_sigma', 1.0)  # Default to 1.0 if not set
        initial_values = create_gaussian_initial_condition(
            st.session_state.graph, 
            center, 
            sigma=sigma, 
            amplitude=amplitude, 
            pos=st.session_state.graph_pos if "graph_pos" in st.session_state else None
        )
    else:  # Random
        initial_values = {node: np.random.random() * amplitude for node in st.session_state.graph.nodes()}
    
    # Store the initial values in session state for visualization
    st.session_state.node_values = initial_values
    
    # Enable diffusion visualization mode
    st.session_state.show_diffusion = True
    
    # Initialize solutions dictionary with just the initial values
    st.session_state.diffusion_solutions = {0.0: initial_values}
    st.session_state.diffusion_time_points = [0.0]
    st.session_state.current_time_index = 0
    
    # Set diffusion type if not already set
    if "diffusion_type" not in st.session_state:
        st.session_state.diffusion_type = "Initial Condition"
    
    # Store visualization type preference before rerunning
    if "viz_type" in st.session_state:
        st.session_state.saved_viz_type = st.session_state.viz_type
    
    # Notify the user
    st.success("Initial condition applied.")
    
    # Rerun to update the visualization
    st.rerun()

def run_diffusion_simulation():
    """
    Run the diffusion simulation with the current parameters from session state.
    
    This function creates the appropriate solver, sets the initial condition,
    runs the simulation, and stores the results in session state.
    """
    # Get parameters from session state
    diffusion_type = st.session_state.diffusion_type
    diffusion_coefficient = st.session_state.diffusion_coefficient
    dt = st.session_state.dt
    num_steps = st.session_state.num_steps
    store_interval = st.session_state.store_interval
    normalize_laplacian = st.session_state.normalize_laplacian
    ic_type = st.session_state.ic_type
    amplitude = st.session_state.amplitude
    
    # Create appropriate solver based on diffusion type
    if diffusion_type == "Ordinary Diffusion":
        solver = OrdinaryGraphDiffusion(
            st.session_state.graph,
            diffusion_coefficient=diffusion_coefficient,
            dt=dt,
            normalize_laplacian=normalize_laplacian
        )
    elif diffusion_type == "Finite-Velocity Diffusion":
        relaxation_time = st.session_state.relaxation_time
        solver = FiniteVelocityGraphDiffusion(
            st.session_state.graph,
            diffusion_coefficient=diffusion_coefficient,
            relaxation_time=relaxation_time,
            dt=dt,
            normalize_laplacian=normalize_laplacian
        )
    else:  # Diffusion with Potential
        potential_type = st.session_state.potential_type
        potential_strength = st.session_state.potential_strength
        
        # Create potential based on selected type
        potential = {}
        if potential_type == "Constant":
            potential = {node: potential_strength for node in st.session_state.graph.nodes()}
        elif potential_type == "Node Degree":
            degree_dict = dict(st.session_state.graph.degree())
            max_degree = max(degree_dict.values()) if degree_dict else 1
            potential = {node: (degree/max_degree) * potential_strength 
                        for node, degree in degree_dict.items()}
        elif potential_type == "Random":
            potential = {node: np.random.random() * potential_strength 
                        for node in st.session_state.graph.nodes()}
        elif potential_type == "Center Attractor":
            # Create attractive potential that pulls toward the center
            if "graph_pos" in st.session_state:
                center_pos = np.mean([pos for pos in st.session_state.graph_pos.values()], axis=0)
                potential = {}
                for node in st.session_state.graph.nodes():
                    node_pos = st.session_state.graph_pos[node]
                    distance = np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(node_pos, center_pos)))
                    # Attraction increases with distance from center
                    potential[node] = distance * potential_strength
            else:
                # Fallback for when positions aren't available
                potential = {node: potential_strength for node in st.session_state.graph.nodes()}
        
        solver = GraphDiffusionWithPotential(
            st.session_state.graph,
            diffusion_coefficient=diffusion_coefficient,
            dt=dt,
            normalize_laplacian=normalize_laplacian,
            potential=potential
        )
    
    # Set initial condition based on selected type
    if ic_type == "Delta Function":
        # Try to use center_node from session state
        if "center_node" in st.session_state and st.session_state.center_node in st.session_state.graph.nodes():
            center = st.session_state.center_node
        else:
            # Use first node as fallback
            center = list(st.session_state.graph.nodes())[0]
        
        initial_values = create_delta_initial_condition(
            st.session_state.graph, 
            center, 
            amplitude=amplitude
        )
    elif ic_type == "Gaussian":
        # Try to use center_node from session state
        if "center_node" in st.session_state and st.session_state.center_node in st.session_state.graph.nodes():
            center = st.session_state.center_node
        else:
            # Use first node as fallback
            center = list(st.session_state.graph.nodes())[0]
        
        sigma = st.session_state.get('gaussian_sigma', 1.0)  # Default to 1.0 if not set
        initial_values = create_gaussian_initial_condition(
            st.session_state.graph, 
            center, 
            sigma=sigma, 
            amplitude=amplitude, 
            pos=st.session_state.graph_pos if "graph_pos" in st.session_state else None
        )
    else:  # Random
        initial_values = {node: np.random.random() * amplitude for node in st.session_state.graph.nodes()}
    
    # Set initial condition
    solver.set_initial_condition(initial_values)
    
    # Store the initial node values for visualization
    st.session_state.node_values = solver.get_node_values()
    
    # Run simulation with progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Running simulation...")
    
    # Run simulation
    start_time = time.time()
    
    # Run in smaller chunks to update progress bar
    chunk_size = min(20, num_steps)
    steps_completed = 0
    
    while steps_completed < num_steps:
        steps_to_run = min(chunk_size, num_steps - steps_completed)
        solver.run(steps_to_run, store_interval=store_interval)
        steps_completed += steps_to_run
        
        # Update progress
        progress = steps_completed / num_steps
        progress_bar.progress(progress)
        status_text.text(f"Running simulation... {steps_completed}/{num_steps} steps completed")
    
    end_time = time.time()
    
    # Store results for visualization
    st.session_state.diffusion_solutions = solver.get_solutions()
    st.session_state.diffusion_time_points = sorted(st.session_state.diffusion_solutions.keys())
    st.session_state.current_time_index = 0
    
    # Set current values to initial condition
    if st.session_state.diffusion_time_points:
        initial_time = st.session_state.diffusion_time_points[0]
        st.session_state.node_values = st.session_state.diffusion_solutions[initial_time]
    
    # Report completion
    status_text.text(f"Simulation completed in {end_time - start_time:.2f} seconds.")
    
    # Store visualization type preference before rerunning
    if "viz_type" in st.session_state:
        st.session_state.saved_viz_type = st.session_state.viz_type
    
    # Enable diffusion visualization
    st.session_state.show_diffusion = True
    
    # Rerun to show results
    st.rerun()
