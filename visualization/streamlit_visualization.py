"""
Streamlit-specific visualization functions for the finite velocity diffusion app.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from visualization.plotting import (
    plot_1d_comparison, 
    plot_2d_heatmap,
    plot_time_series
)

def display_1d_results(x, u_finite, u_classical, title='1D Finite Velocity vs Classical Diffusion'):
    """
    Display 1D comparison results using matplotlib.
    
    Parameters:
    -----------
    x : np.ndarray
        Spatial coordinates
    u_finite : np.ndarray
        Finite velocity diffusion solution
    u_classical : np.ndarray
        Classical diffusion solution
    title : str
        Plot title
    """
    fig, _ = plot_1d_comparison(
        x=x,
        u_finite=u_finite,
        u_classical=u_classical,
        title=title
    )
    st.pyplot(fig)

def display_2d_results(X, Y, u, title='2D Finite Velocity Diffusion'):
    """
    Display 2D results using matplotlib.
    
    Parameters:
    -----------
    X, Y : np.ndarray
        Meshgrid of spatial coordinates
    u : np.ndarray
        Solution array
    title : str
        Plot title
    """
    fig, _ = plot_2d_heatmap(
        X=X,
        Y=Y,
        u=u,
        title=title
    )
    st.pyplot(fig)

def display_1d_time_evolution_static(times, solutions, x, title="Time Evolution of Finite Velocity Diffusion"):
    """
    Display static plots of 1D time evolution.
    
    Parameters:
    -----------
    times : list
        List of time points
    solutions : list
        List of solution arrays at each time point
    x : np.ndarray
        Spatial coordinates
    title : str
        Plot title
    """
    fig, _ = plot_time_series(
        times=times,
        solutions=solutions,
        x=x,
        title=title
    )
    st.pyplot(fig)

def display_1d_time_evolution_slider(times, solutions, x):
    """
    Display 1D time evolution with an interactive slider.
    
    Parameters:
    -----------
    times : list
        List of time points
    solutions : list
        List of solution arrays at each time point
    x : np.ndarray
        Spatial coordinates
    """
    st.subheader("Interactive Time Evolution")
    
    # Create a slider for the time index
    time_idx = st.slider(
        "Time", 
        min_value=0, 
        max_value=len(times)-1, 
        value=0,
        format=f"t = %f"
    )
    
    # Get the solution at the selected time
    selected_time = times[time_idx]
    selected_solution = solutions[time_idx]
    
    # Plot the selected solution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, selected_solution, 'b-', linewidth=2)
    ax.set_xlabel('Position')
    ax.set_ylabel('Concentration')
    ax.set_title(f'Solution at Time = {selected_time:.4f}')
    ax.grid(True)
    st.pyplot(fig)

def display_1d_time_evolution_animation(times, solutions, x, animation_speed=10):
    """
    Display 1D time evolution with a Plotly animation.
    
    Parameters:
    -----------
    times : list
        List of time points
    solutions : list
        List of solution arrays at each time point
    x : np.ndarray
        Spatial coordinates
    animation_speed : int
        Animation speed in frames per second
    """
    st.subheader("Animated Time Evolution")
    
    # Create the Plotly figure for animation
    fig = go.Figure()
    
    # Add traces for each time step
    for i, (t, sol) in enumerate(zip(times, solutions)):
        visible = (i == 0)  # Only first frame is visible initially
        fig.add_trace(
            go.Scatter(
                x=x,
                y=sol,
                mode='lines',
                name=f'Time {t:.4f}',
                line=dict(width=2, color='blue'),
                visible=visible
            )
        )
    
    # Create frames for animation
    frames = []
    for i, t in enumerate(times):
        frames.append(
            go.Frame(
                data=[go.Scatter(x=x, y=solutions[i])],
                name=f"frame_{i}",
                layout=go.Layout(title_text=f"Time: {t:.4f}")
            )
        )
    
    fig.frames = frames
    
    # Add buttons for animation control
    fig.update_layout(
        title="Finite Velocity Diffusion Evolution",
        xaxis_title="Position",
        yaxis_title="Concentration",
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 1000 // animation_speed, "redraw": True},
                                   "fromcurrent": True,
                                   "transition": {"duration": 0}}]
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                     "mode": "immediate",
                                     "transition": {"duration": 0}}]
                }
            ],
            "direction": "left",
            "pad": {"l": 10, "t": 10},
            "showactive": False,
            "x": 0.1,
            "y": 0,
            "xanchor": "right",
            "yanchor": "top"
        }],
        sliders=[{
            "steps": [
                {
                    "label": f"{t:.3f}",
                    "method": "animate",
                    "args": [[f"frame_{i}"], {"frame": {"duration": 0, "redraw": True},
                                             "mode": "immediate",
                                             "transition": {"duration": 0}}]
                }
                for i, t in enumerate(times)
            ],
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "visible": True,
                "prefix": "Time: ",
                "xanchor": "right",
                "font": {"size": 16}
            },
            "transition": {"duration": 0},
            "pad": {"l": 50, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0
        }]
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

def display_2d_time_evolution_static(times, solutions, X, Y):
    """
    Display static plots of 2D time evolution.
    
    Parameters:
    -----------
    times : list
        List of time points
    solutions : list
        List of solution arrays at each time point
    X, Y : np.ndarray
        Meshgrid of spatial coordinates
    """
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Final Result", "Time Evolution"])
    
    with tab1:
        # Plot final results
        fig, _ = plot_2d_heatmap(
            X=X,
            Y=Y,
            u=solutions[-1],
            title='2D Finite Velocity Diffusion (Final State)'
        )
        st.pyplot(fig)
    
    with tab2:
        # Plot time evolution as multiple heatmaps
        st.subheader("Time Evolution of 2D Diffusion")
        for i, (t, sol) in enumerate(zip(times, solutions)):
            st.write(f"Time: {t:.3f}")
            fig, _ = plot_2d_heatmap(
                X=X,
                Y=Y,
                u=sol,
                title=f'Time = {t:.3f}'
            )
            st.pyplot(fig)

def display_2d_time_evolution_slider(times, solutions, X, Y):
    """
    Display 2D time evolution with an interactive slider.
    
    Parameters:
    -----------
    times : list
        List of time points
    solutions : list
        List of solution arrays at each time point
    X, Y : np.ndarray
        Meshgrid of spatial coordinates
    """
    st.subheader("Interactive Time Evolution")
    
    # Create a slider for the time index
    time_idx = st.slider(
        "Time", 
        min_value=0, 
        max_value=len(times)-1, 
        value=0,
        format=f"t = %f"
    )
    
    # Get the solution at the selected time
    selected_time = times[time_idx]
    selected_solution = solutions[time_idx]
    
    # Plot the selected solution
    fig, _ = plot_2d_heatmap(
        X=X,
        Y=Y,
        u=selected_solution,
        title=f'2D Solution at Time = {selected_time:.4f}'
    )
    st.pyplot(fig)

def display_2d_time_evolution_animation(times, solutions, x, y, animation_speed=10):
    """
    Display 2D time evolution with a Plotly animation.
    
    Parameters:
    -----------
    times : list
        List of time points
    solutions : list
        List of solution arrays at each time point
    x, y : np.ndarray
        Spatial coordinate arrays
    animation_speed : int
        Animation speed in frames per second
    """
    st.subheader("Animated Time Evolution")
    
    # Try a different approach for 2D animation using heatmap instead of contour
    fig = go.Figure()
    
    # Find global min and max for consistent color scale
    vmin = min(np.min(sol) for sol in solutions)
    vmax = max(np.max(sol) for sol in solutions)
    
    # Create heatmap for each time step (only first one visible initially)
    fig.add_trace(
        go.Heatmap(
            z=solutions[0],
            x=x,
            y=y,
            colorscale='viridis',
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(title="Concentration")
        )
    )
    
    # Create frames for animation using heatmap
    frames = []
    for i, t in enumerate(times):
        frames.append(
            go.Frame(
                data=[go.Heatmap(
                    z=solutions[i],
                    x=x,
                    y=y,
                    colorscale='viridis',
                    zmin=vmin,
                    zmax=vmax
                )],
                name=f"frame_{i}",
                layout=go.Layout(title_text=f"Time: {t:.4f}")
            )
        )
    
    fig.frames = frames
    
    # Add buttons for animation control
    fig.update_layout(
        title="2D Finite Velocity Diffusion Evolution",
        xaxis_title="X Position",
        yaxis_title="Y Position",
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 1000 // animation_speed, "redraw": True},
                                  "fromcurrent": True,
                                  "transition": {"duration": 0}}]
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                     "mode": "immediate",
                                     "transition": {"duration": 0}}]
                }
            ],
            "direction": "left",
            "pad": {"l": 10, "t": 10},
            "showactive": False,
            "x": 0.1,
            "y": 0,
            "xanchor": "right",
            "yanchor": "top"
        }],
        sliders=[{
            "steps": [
                {
                    "label": f"{t:.3f}",
                    "method": "animate",
                    "args": [[f"frame_{i}"], {"frame": {"duration": 0, "redraw": True},
                                             "mode": "immediate",
                                             "transition": {"duration": 0}}]
                }
                for i, t in enumerate(times)
            ],
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "visible": True,
                "prefix": "Time: ",
                "xanchor": "right",
                "font": {"size": 16}
            },
            "transition": {"duration": 0},
            "pad": {"l": 50, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0
        }]
    )
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True)
