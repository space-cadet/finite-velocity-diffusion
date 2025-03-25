"""
Visualization functions for diffusion on graphs.

This module provides specialized visualization functions for displaying
diffusion processes on graph structures.
"""

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List, Union

def visualize_diffusion_2d(
    G: nx.Graph,
    pos: Dict,
    node_values: Dict,
    color_scale: str = "Viridis",
    title: str = "Diffusion on Graph",
    node_size: int = 10,
    edge_width: int = 1
) -> go.Figure:
    """
    Visualize diffusion values on a graph using Plotly.
    
    Parameters:
    -----------
    G : nx.Graph
        The graph to visualize
    pos : Dict
        Dictionary of node positions
    node_values : Dict
        Dictionary of node values from the diffusion process
    color_scale : str
        Plotly color scale name
    title : str
        Title for the visualization
    node_size : int
        Size of the nodes in the visualization
    edge_width : int
        Width of the edges in the visualization
    
    Returns:
    --------
    go.Figure
        Plotly figure with graph visualization
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
        line=dict(width=edge_width, color='#888'),
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
        
        # Handle potential key mismatch (integer vs string/tuple nodes)
        if node in node_values:
            value = node_values[node]
        else:
            # Try string conversion as fallback
            try:
                str_node = str(node)
                value = node_values.get(str_node, 0.0)
            except:
                value = 0.0
                
        node_color.append(value)
        
        # Add node information for hover
        if isinstance(node, tuple):
            node_label = f"({node[0]}, {node[1]})"
        else:
            node_label = str(node)
        
        node_text.append(f"Node: {node_label}<br>Value: {value:.4f}<br>Degree: {G.degree(node)}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale=color_scale,
            color=node_color,
            size=node_size,
            colorbar=dict(
                thickness=15,
                title="Diffusion Value"
            ),
            line=dict(width=2)
        )
    )
    
    # Create the figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(size=16)),
            showlegend=False,
            hovermode='closest',
            margin=dict(t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white"
        )
    )
    
    return fig

def visualize_diffusion_3d(
    G: nx.Graph,
    pos: Dict,
    node_values: Dict,
    color_scale: str = "Viridis",
    title: str = "3D Diffusion Visualization",
    node_size: int = 8,
    edge_width: int = 2
) -> go.Figure:
    """
    Create a 3D visualization of diffusion on a graph using Plotly.
    
    Parameters:
    -----------
    G : nx.Graph
        The graph to visualize
    pos : Dict
        Dictionary of node positions (2D layout)
    node_values : Dict
        Values for each node from the diffusion process
    color_scale : str
        Color scale to use for node colors
    title : str
        Title for the visualization
    node_size : int
        Size of the nodes in the visualization
    edge_width : int
        Width of the edges in the visualization
    
    Returns:
    --------
    go.Figure
        Plotly figure with 3D graph visualization
    """
    # Convert 2D positions to 3D, using node value as Z-coordinate
    pos_3d = {}
    for node, position in pos.items():
        x, y = position
        
        # Handle potential key mismatch (integer vs string/tuple nodes)
        if node in node_values:
            z = node_values[node]  # Use the diffusion value as z-coordinate
        else:
            # Try string conversion as fallback
            try:
                str_node = str(node)
                z = node_values.get(str_node, 0.0)
            except:
                z = 0.0
        
        pos_3d[node] = (x, y, z)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_z = []
    
    for edge in G.edges():
        x0, y0, z0 = pos_3d[edge[0]]
        x1, y1, z1 = pos_3d[edge[1]]
        
        # Add edge line segments
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(color='#888', width=edge_width),
        hoverinfo='none'
    )
    
    # Create node trace
    node_x = []
    node_y = []
    node_z = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y, z = pos_3d[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        
        if isinstance(node, tuple):
            node_label = f"({node[0]}, {node[1]})"
        else:
            node_label = str(node)
        
        # Handle potential key mismatch (integer vs string/tuple nodes)
        if node in node_values:
            value = node_values[node]
        else:
            # Try string conversion as fallback
            try:
                str_node = str(node)
                value = node_values.get(str_node, 0.0)
            except:
                value = 0.0
        
        node_text.append(f"Node: {node_label}<br>Value: {value:.4f}<br>Degree: {G.degree(node)}")
        node_color.append(value)
    
    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode='markers',
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale=color_scale,
            colorbar=dict(title="Diffusion Value"),
            line=dict(color='#000', width=0.5)
        ),
        text=node_text,
        hoverinfo='text'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    # Update layout
    fig.update_layout(
        title=dict(text=title),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title="Value"),
            aspectmode='data'
        ),
        margin=dict(t=40),
        template="plotly_white"
    )
    
    return fig

def create_animated_diffusion(
    G: nx.Graph,
    pos: Dict,
    solutions: Dict[float, Dict],
    color_scale: str = "Viridis",
    base_title: str = "Diffusion on Graph",
    node_size: int = 10,
    edge_width: int = 1
) -> go.Figure:
    """
    Create an animated visualization of diffusion on a graph over time.
    
    Parameters:
    -----------
    G : nx.Graph
        The graph to visualize
    pos : Dict
        Dictionary of node positions
    solutions : Dict[float, Dict]
        Dictionary mapping time points to node values dictionaries
    color_scale : str
        Plotly color scale name
    base_title : str
        Base title for the visualization
    node_size : int
        Size of the nodes in the visualization
    edge_width : int
        Width of the edges in the visualization
    
    Returns:
    --------
    go.Figure
        Plotly figure with animation capabilities
    """
    # Get all time points sorted
    time_points = sorted(solutions.keys())
    
    if not time_points:
        raise ValueError("No time points provided in solutions")
    
    # Create edge trace (static across all frames)
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
        line=dict(width=edge_width, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Define a function to create a frame for a given time point
    def create_frame_data(t):
        node_values = solutions[t]
        node_x = []
        node_y = []
        node_color = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Handle potential key mismatch (integer vs string/tuple nodes)
            if node in node_values:
                value = node_values[node]
            else:
                # Try string conversion as fallback
                try:
                    str_node = str(node)
                    value = node_values.get(str_node, 0.0)
                except:
                    value = 0.0
            
            node_color.append(value)
            
            # Add node information for hover
            if isinstance(node, tuple):
                node_label = f"({node[0]}, {node[1]})"
            else:
                node_label = str(node)
            
            node_text.append(f"Node: {node_label}<br>Value: {value:.4f}<br>Degree: {G.degree(node)}")
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale=color_scale,
                color=node_color,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title="Diffusion Value"
                ),
                line=dict(width=2)
            )
        )
        
        return [edge_trace, node_trace]
    
    # Create the base figure with the first frame
    fig = go.Figure(
        data=create_frame_data(time_points[0]),
        layout=go.Layout(
            title=dict(text=f"{base_title} (t = {time_points[0]:.3f})", font=dict(size=16)),
            showlegend=False,
            hovermode='closest',
            margin=dict(t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white"
        )
    )
    
    # Add frames for animation
    frames = []
    for t in time_points:
        frame = go.Frame(
            data=create_frame_data(t),
            layout=go.Layout(
                title=dict(text=f"{base_title} (t = {t:.3f})")
            ),
            name=f"t={t:.3f}"
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Add animation controls
    sliders = [
        dict(
            active=0,
            yanchor="top",
            xanchor="left",
            currentvalue=dict(
                font=dict(size=12),
                prefix="Time: ",
                visible=True,
                xanchor="right"
            ),
            transition=dict(duration=150, easing="cubic-in-out"),
            pad=dict(b=10, t=50),
            len=0.9,
            x=0.1,
            y=0,
            steps=[
                dict(
                    method="animate",
                    args=[
                        [f"t={t:.3f}"],
                        dict(
                            frame=dict(duration=200, redraw=True),
                            mode="immediate",
                            transition=dict(duration=150, easing="cubic-in-out")
                        )
                    ],
                    label=f"{t:.2f}"
                )
                for t in time_points
            ]
        )
    ]
    
    # Add play and pause buttons
    updatemenus = [
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=200, redraw=True),
                            fromcurrent=True,
                            mode="immediate",
                            transition=dict(duration=150, easing="cubic-in-out")
                        )
                    ]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame=dict(duration=0, redraw=True),
                            mode="immediate",
                            transition=dict(duration=0)
                        )
                    ]
                )
            ],
            direction="left",
            pad=dict(r=10, t=87),
            x=0.1,
            y=0,
            xanchor="right",
            yanchor="top"
        )
    ]
    
    # Update layout with animation controls
    fig.update_layout(
        updatemenus=updatemenus,
        sliders=sliders
    )
    
    return fig

def visualize_diffusion_comparison(
    G: nx.Graph,
    pos: Dict,
    values1: Dict,
    values2: Dict,
    color_scale: str = "Viridis",
    title1: str = "Ordinary Diffusion",
    title2: str = "Finite-Velocity Diffusion"
) -> go.Figure:
    """
    Create a side-by-side comparison of two diffusion processes on the same graph.
    
    Parameters:
    -----------
    G : nx.Graph
        The graph to visualize
    pos : Dict
        Dictionary of node positions
    values1 : Dict
        Node values for the first diffusion process
    values2 : Dict
        Node values for the second diffusion process
    color_scale : str
        Color scale to use for node colors
    title1 : str
        Title for the first subplot
    title2 : str
        Title for the second subplot
    
    Returns:
    --------
    go.Figure
        Plotly figure with side-by-side comparison
    """
    # Create subplots
    fig = go.Figure()
    
    # Add edges to both subplots (constant)
    edge_x1 = []
    edge_y1 = []
    edge_x2 = []
    edge_y2 = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_x1.extend([x0, x1, None])
        edge_y1.extend([y0, y1, None])
        edge_x2.extend([x0 + 2.0, x1 + 2.0, None])  # Offset for second subplot
        edge_y2.extend([y0, y1, None])
    
    # Add edge traces
    fig.add_trace(go.Scatter(
        x=edge_x1,
        y=edge_y1,
        mode='lines',
        line=dict(color='#888', width=1),
        hoverinfo='none',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=edge_x2,
        y=edge_y2,
        mode='lines',
        line=dict(color='#888', width=1),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add node traces
    node_x1 = []
    node_y1 = []
    node_text1 = []
    node_color1 = []
    
    node_x2 = []
    node_y2 = []
    node_text2 = []
    node_color2 = []
    
    for node in G.nodes():
        x, y = pos[node]
        
        # First subplot
        node_x1.append(x)
        node_y1.append(y)
        
        # Second subplot (offset x-coordinate)
        node_x2.append(x + 2.0)
        node_y2.append(y)
        
        if isinstance(node, tuple):
            node_label = f"({node[0]}, {node[1]})"
        else:
            node_label = str(node)
        
        # Handle potential key mismatch (integer vs string/tuple nodes)
        if node in values1:
            value1 = values1[node]
        else:
            # Try string conversion as fallback
            try:
                str_node = str(node)
                value1 = values1.get(str_node, 0.0)
            except:
                value1 = 0.0
                
        if node in values2:
            value2 = values2[node]
        else:
            # Try string conversion as fallback
            try:
                str_node = str(node)
                value2 = values2.get(str_node, 0.0)
            except:
                value2 = 0.0
        
        node_text1.append(f"Node: {node_label}<br>Value: {value1:.4f}")
        node_text2.append(f"Node: {node_label}<br>Value: {value2:.4f}")
        
        node_color1.append(value1)
        node_color2.append(value2)
    
    # Add node traces
    fig.add_trace(go.Scatter(
        x=node_x1,
        y=node_y1,
        mode='markers',
        marker=dict(
            size=10,
            color=node_color1,
            colorscale=color_scale,
            showscale=True,
            colorbar=dict(
                title="Value"
            )
        ),
        text=node_text1,
        hoverinfo='text',
        name=title1
    ))
    
    fig.add_trace(go.Scatter(
        x=node_x2,
        y=node_y2,
        mode='markers',
        marker=dict(
            size=10,
            color=node_color2,
            colorscale=color_scale,
            showscale=True,
            colorbar=dict(
                title="Value"
            )
        ),
        text=node_text2,
        hoverinfo='text',
        name=title2
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(text="Diffusion Comparison"),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1, 3]  # Adjust range to fit both subplots
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        hovermode='closest',
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=40),
        annotations=[
            dict(
                x=0,
                y=1.05,
                xref="paper",
                yref="paper",
                text=title1,
                showarrow=False,
                font=dict(size=14)
            ),
            dict(
                x=1,
                y=1.05,
                xref="paper",
                yref="paper",
                text=title2,
                showarrow=False,
                font=dict(size=14)
            )
        ]
    )
    
    return fig
