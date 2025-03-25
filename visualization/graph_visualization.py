"""
Graph visualization functions for the graph diffusion solver.

This module provides functions for visualizing graphs and diffusion on graphs,
using Plotly and NetworkX.
"""

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List, Union

def create_3d_graph_visualization(
    G: nx.Graph,
    pos: Dict,
    node_values: Dict,
    color_scale: str = "Viridis",
    title: str = "3D Graph Visualization"
) -> go.Figure:
    """
    Create a 3D visualization of a graph using Plotly.
    
    Parameters:
    -----------
    G : nx.Graph
        The graph to visualize
    pos : Dict
        Dictionary of node positions (2D layout)
    node_values : Dict
        Values for each node, used for coloring
    color_scale : str
        Color scale to use for node colors
    title : str
        Title for the visualization
    
    Returns:
    --------
    go.Figure
        Plotly figure with 3D graph visualization
    """
    # Convert 2D positions to 3D
    pos_3d = {}
    for node, position in pos.items():
        # Add a z-coordinate as a function of the node value
        x, y = position
        z = node_values[node] * 0.5  # Scale the z value for better visualization
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
        line=dict(color='#888', width=2),
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
        
        node_text.append(f"Node: {node_label}<br>Value: {node_values[node]:.3f}<br>Degree: {G.degree(node)}")
        node_color.append(node_values[node])
    
    node_trace = go.Scatter3d(
        x=node_x,
        y=node_y,
        z=node_z,
        mode='markers',
        marker=dict(
            size=8,
            color=node_color,
            colorscale=color_scale,
            colorbar=dict(title="Node Value"),
            line=dict(color='#000', width=0.5)
        ),
        text=node_text,
        hoverinfo='text'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        template="plotly_white"
    )
    
    return fig

def create_time_evolution_visualization(
    G: nx.Graph,
    pos: Dict,
    node_values_over_time: Dict[float, Dict],
    color_scale: str = "Viridis",
    title: str = "Diffusion on Graph"
) -> go.Figure:
    """
    Create a visualization of diffusion on a graph over time using Plotly.
    
    Parameters:
    -----------
    G : nx.Graph
        The graph to visualize
    pos : Dict
        Dictionary of node positions
    node_values_over_time : Dict[float, Dict]
        Dictionary mapping time points to node values
    color_scale : str
        Color scale to use for node colors
    title : str
        Title for the visualization
    
    Returns:
    --------
    go.Figure
        Plotly figure with time evolution visualization
    """
    # Create edge traces (constant over time)
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(color='#888', width=1),
        hoverinfo='none'
    )
    
    # Create frames for animation
    frames = []
    
    # Create slider steps
    steps = []
    
    # Sort time points
    times = sorted(node_values_over_time.keys())
    
    for i, t in enumerate(times):
        node_values = node_values_over_time[t]
        
        # Node trace for this time point
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            if isinstance(node, tuple):
                node_label = f"({node[0]}, {node[1]})"
            else:
                node_label = str(node)
            
            value = node_values[node]
            node_text.append(f"Node: {node_label}<br>Value: {value:.3f}<br>Time: {t:.2f}")
            node_color.append(value)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            marker=dict(
                size=10,
                color=node_color,
                colorscale=color_scale,
                showscale=True,
                colorbar=dict(title="Value")
            ),
            text=node_text,
            hoverinfo='text',
            name=f't={t:.2f}'
        )
        
        # Create frame for this time point
        frame = go.Frame(
            data=[edge_trace, node_trace],
            name=f'{t:.2f}'
        )
        frames.append(frame)
        
        # Create slider step
        step = {
            'args': [
                [f'{t:.2f}'],
                {
                    'frame': {'duration': 300, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 300}
                }
            ],
            'label': f'{t:.2f}',
            'method': 'animate'
        }
        steps.append(step)
    
    # Initial node trace (t=0)
    initial_node_values = node_values_over_time[times[0]]
    
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        if isinstance(node, tuple):
            node_label = f"({node[0]}, {node[1]})"
        else:
            node_label = str(node)
        
        value = initial_node_values[node]
        node_text.append(f"Node: {node_label}<br>Value: {value:.3f}<br>Time: {times[0]:.2f}")
        node_color.append(value)
    
    initial_node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        marker=dict(
            size=10,
            color=node_color,
            colorscale=color_scale,
            showscale=True,
            colorbar=dict(title="Value")
        ),
        text=node_text,
        hoverinfo='text',
        name=f't={times[0]:.2f}'
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, initial_node_trace],
        frames=frames,
        layout=go.Layout(
            title=title,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode='closest',
            updatemenus=[
                {
                    'buttons': [
                        {
                            'args': [
                                None,
                                {
                                    'frame': {'duration': 500, 'redraw': True},
                                    'fromcurrent': True,
                                    'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                                }
                            ],
                            'label': 'Play',
                            'method': 'animate'
                        },
                        {
                            'args': [
                                [None],
                                {
                                    'frame': {'duration': 0, 'redraw': True},
                                    'mode': 'immediate',
                                    'transition': {'duration': 0}
                                }
                            ],
                            'label': 'Pause',
                            'method': 'animate'
                        }
                    ],
                    'direction': 'left',
                    'pad': {'r': 10, 't': 10},
                    'showactive': False,
                    'type': 'buttons',
                    'x': 0.1,
                    'xanchor': 'right',
                    'y': 0,
                    'yanchor': 'top'
                }
            ],
            sliders=[
                {
                    'active': 0,
                    'yanchor': 'top',
                    'xanchor': 'left',
                    'currentvalue': {
                        'font': {'size': 16},
                        'prefix': 'Time: ',
                        'visible': True,
                        'xanchor': 'right'
                    },
                    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
                    'pad': {'b': 10, 't': 50},
                    'len': 0.9,
                    'x': 0.1,
                    'y': 0,
                    'steps': steps
                }
            ]
        )
    )
    
    return fig

def visualize_graph_comparison(
    G: nx.Graph,
    pos: Dict,
    values1: Dict,
    values2: Dict,
    title1: str = "Ordinary Diffusion",
    title2: str = "Finite-Velocity Diffusion",
    color_scale: str = "Viridis"
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
        Node values for the first process
    values2 : Dict
        Node values for the second process
    title1 : str
        Title for the first subplot
    title2 : str
        Title for the second subplot
    color_scale : str
        Color scale to use for node colors
    
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
        
        value1 = values1[node]
        value2 = values2[node]
        
        node_text1.append(f"Node: {node_label}<br>Value: {value1:.3f}")
        node_text2.append(f"Node: {node_label}<br>Value: {value2:.3f}")
        
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
                title="Value",
                x=-0.07
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
                title="Value",
                x=1.07
            )
        ),
        text=node_text2,
        hoverinfo='text',
        name=title2
    ))
    
    # Update layout
    fig.update_layout(
        title="Diffusion Comparison",
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
