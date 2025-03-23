import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from solver.one_dimensional import FiniteVelocityDiffusion1D
from solver.two_dimensional import FiniteVelocityDiffusion2D
from utils import (
    gaussian_initial_condition,
    classical_diffusion_solution,
    calculate_propagation_speed,
    DEFAULT_PARAMETERS
)
from visualization.plotting import (
    plot_1d_comparison,
    plot_2d_heatmap,
    plot_time_series
)

st.set_page_config(page_title="Finite Velocity Diffusion Solver", layout="wide")

st.title("Finite Velocity Diffusion Solver")
st.markdown("""
This app solves the finite velocity diffusion equation (telegrapher's equation):
τ∂²u/∂t² + ∂u/∂t = D∇²u

The solution is compared with the classical diffusion equation for a Gaussian initial condition.
""")

# Sidebar controls
st.sidebar.header("Simulation Parameters")

# Dimension selection
dimension = st.sidebar.radio("Select Dimension", ["1D", "2D"])

# Physical parameters
D = st.sidebar.slider(
    "Diffusion Coefficient (D)",
    min_value=0.1,
    max_value=5.0,
    value=DEFAULT_PARAMETERS['D'],
    step=0.1
)

tau = st.sidebar.slider(
    "Relaxation Time (τ)",
    min_value=0.1,
    max_value=5.0,
    value=DEFAULT_PARAMETERS['tau'],
    step=0.1
)

# Initial condition parameters
amplitude = st.sidebar.slider(
    "Initial Amplitude",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1
)

sigma = st.sidebar.slider(
    "Initial Standard Deviation",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1
)

# Simulation parameters
num_steps = st.sidebar.slider(
    "Number of Time Steps",
    min_value=10,
    max_value=1000,
    value=100,
    step=10
)

# Show time evolution
show_evolution = st.sidebar.checkbox("Show Time Evolution", value=False)
if show_evolution:
    evolution_steps = st.sidebar.slider(
        "Number of Time Points to Show",
        min_value=5,
        max_value=50,
        value=20
    )
    
    # Visualization options
    st.sidebar.markdown("### Visualization Options")
    viz_type = st.sidebar.radio(
        "Visualization Type",
        ["Static Plots", "Interactive Slider", "Plotly Animation"],
        index=0
    )
    
    if viz_type == "Plotly Animation":
        animation_speed = st.sidebar.slider(
            "Animation Speed (frames per second)",
            min_value=1,
            max_value=30,
            value=10,
            step=1
        )

# Create solver
if dimension == "1D":
    solver = FiniteVelocityDiffusion1D(
        nx=DEFAULT_PARAMETERS['nx'],
        dx=DEFAULT_PARAMETERS['dx'],
        dt=DEFAULT_PARAMETERS['dt'],
        D=D,
        tau=tau,
        x_min=DEFAULT_PARAMETERS['x_min'],
        x_max=DEFAULT_PARAMETERS['x_max']
    )
    
    # Set initial condition
    initial_condition = gaussian_initial_condition(
        solver.x,
        amplitude=amplitude,
        center=(solver.x[-1] + solver.x[0]) / 2,  # Center of domain
        sigma=sigma
    )
    solver.set_initial_condition(initial_condition)
    
    # Common computation code for all visualization types
    # This optimizes by computing solutions once and reusing them
    x = solver.x
    times = []
    solutions = []
    
    # Progress bar for simulation
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if show_evolution:
        # Calculate step size for evolution plots/animation
        step_size = max(1, num_steps // evolution_steps)
        
        # Run solver and capture intermediate results
        for i in range(0, num_steps + 1, step_size):
            if i == 0:
                # Initial condition
                times.append(0)
                solutions.append(solver.u.copy())
            else:
                # Solve for steps
                solver.solve(step_size)
                times.append(i * DEFAULT_PARAMETERS['dt'])
                solutions.append(solver.u.copy())
                
                # Update progress
                progress = min(1.0, i / num_steps)
                progress_bar.progress(progress)
                status_text.text(f"Computing solution: {int(progress * 100)}% complete")
        
        # Final result is the last solution
        u = solutions[-1]
        
        # Clear status
        status_text.empty()
    else:
        # Solve normally
        status_text.text("Computing solution...")
        x, u = solver.solve(num_steps)
        progress_bar.progress(1.0)
        status_text.empty()
    
    # Compute classical diffusion solution for comparison
    t = num_steps * DEFAULT_PARAMETERS['dt']
    u_classical = classical_diffusion_solution(
        x,
        t,
        D,
        initial_amplitude=amplitude,
        initial_center=(solver.x[-1] + solver.x[0]) / 2,  # Center of domain
        initial_sigma=sigma
    )
    
    # Plot results using visualization module
    fig, _ = plot_1d_comparison(
        x=x,
        u_finite=u,
        u_classical=u_classical,
        title='1D Finite Velocity vs Classical Diffusion'
    )
    st.pyplot(fig)
    
    # Plot time evolution if requested
    if show_evolution:
        if viz_type == "Static Plots":
            # Use the existing static plot
            fig, _ = plot_time_series(
                times=times,
                solutions=solutions,
                x=x,
                title="Time Evolution of Finite Velocity Diffusion"
            )
            st.pyplot(fig)
            
        elif viz_type == "Interactive Slider":
            # Create an interactive slider for time steps
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
            
        elif viz_type == "Plotly Animation":
            # Create Plotly animation
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
    
else:  # 2D case
    solver = FiniteVelocityDiffusion2D(
        nx=DEFAULT_PARAMETERS['nx'],
        ny=DEFAULT_PARAMETERS['ny'],
        dx=DEFAULT_PARAMETERS['dx'],
        dy=DEFAULT_PARAMETERS['dy'],
        dt=DEFAULT_PARAMETERS['dt'],
        D=D,
        tau=tau,
        x_min=DEFAULT_PARAMETERS['x_min'],
        x_max=DEFAULT_PARAMETERS['x_max'],
        y_min=DEFAULT_PARAMETERS['y_min'],
        y_max=DEFAULT_PARAMETERS['y_max']
    )
    
    # Set initial condition
    initial_condition = gaussian_initial_condition(
        solver.x,
        solver.y,
        amplitude=amplitude,
        center=((solver.x[-1] + solver.x[0]) / 2, (solver.y[-1] + solver.y[0]) / 2),  # Center of domain
        sigma=(sigma, sigma)
    )
    solver.set_initial_condition(initial_condition)
    
    # Common computation code for both static and dynamic visualizations
    X, Y = solver.X, solver.Y
    times = []
    solutions = []
    
    # Progress bar for simulation
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if show_evolution:
        # Calculate step size for evolution plots/animation
        step_size = max(1, num_steps // evolution_steps)
        
        # Run solver and capture intermediate results
        for i in range(0, num_steps + 1, step_size):
            if i == 0:
                # Initial condition
                times.append(0)
                solutions.append(solver.u.copy())
            else:
                # Solve for steps
                solver.solve(step_size)
                times.append(i * DEFAULT_PARAMETERS['dt'])
                solutions.append(solver.u.copy())
                
                # Update progress
                progress = min(1.0, i / num_steps)
                progress_bar.progress(progress)
                status_text.text(f"Computing solution: {int(progress * 100)}% complete")
        
        # Final result is the last solution
        u = solutions[-1]
        
        # Clear status
        status_text.empty()
        
        # Create visualizations based on selected type
        if viz_type == "Static Plots":
            # Create tabs for different visualizations
            tab1, tab2 = st.tabs(["Final Result", "Time Evolution"])
            
            with tab1:
                # Plot final results
                fig, _ = plot_2d_heatmap(
                    X=X,
                    Y=Y,
                    u=u,
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
        
        elif viz_type == "Interactive Slider":
            # Create an interactive slider for time steps
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
        
        elif viz_type == "Plotly Animation":
            # Create Plotly animation for 2D
            st.subheader("Animated Time Evolution")
            
            # Create a Plotly figure for animation
            fig = go.Figure()
            
            # Create contour plots for each time step
            for i, (t, sol) in enumerate(zip(times, solutions)):
                # Only first frame is visible initially
                visible = (i == 0)
                
                # Add a contour trace
                fig.add_trace(
                    go.Contour(
                        z=sol,
                        x=solver.x,  # 1D array for x-coordinates
                        y=solver.y,  # 1D array for y-coordinates
                        colorscale='viridis',
                        visible=visible,
                        contours=dict(
                            showlabels=True,
                            labelfont=dict(size=12, color='white')
                        ),
                        colorbar=dict(
                            title="Concentration",
                            titleside="right"
                        )
                    )
                )
            
            # Create frames for animation
            frames = []
            for i, t in enumerate(times):
                frames.append(
                    go.Frame(
                        data=[go.Contour(
                            z=solutions[i],
                            x=solver.x,
                            y=solver.y,
                            colorscale='viridis',
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
    
    else:
        # Solve normally
        status_text.text("Computing solution...")
        X, Y, u = solver.solve(num_steps)
        progress_bar.progress(1.0)
        status_text.empty()
        
        # Plot results using visualization module
        fig, _ = plot_2d_heatmap(
            X=X,
            Y=Y,
            u=u,
            title='2D Finite Velocity Diffusion'
        )
        st.pyplot(fig)



# Add information about the simulation
st.sidebar.markdown("---")
st.sidebar.markdown("### Simulation Info")
st.sidebar.markdown(f"Time: {num_steps * DEFAULT_PARAMETERS['dt']:.3f}")
st.sidebar.markdown(f"Courant Number: {np.sqrt(D * DEFAULT_PARAMETERS['dt'] / (tau * DEFAULT_PARAMETERS['dx']**2)):.3f}")
st.sidebar.markdown(f"Propagation Speed: {calculate_propagation_speed(D, tau):.3f}") 