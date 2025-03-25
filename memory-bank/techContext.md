# Technical Context: Finite Velocity Diffusion Solver

## Technology Stack

The project is built using the following key technologies:

1. **Python**: Core programming language for all components
2. **NumPy**: Foundation for numerical computations and array operations
3. **SciPy**: Used for specialized scientific computing functions
4. **Matplotlib**: Static plotting and visualization
5. **Plotly**: Interactive plots and animations
6. **Streamlit**: Web application framework for the interactive multipage UI
7. **NetworkX**: Graph creation, manipulation, and analysis for the graph diffusion features
8. **JSON**: File format for parameter persistence

## Development Environment

### Python Environment

- Python 3.8+ recommended
- Virtual environment (`.venv`) for dependency isolation
- Requirements specified in `requirements.txt`

### Core Dependencies

```
numpy<2.0.0
scipy>=1.10.0
matplotlib>=3.7.0
streamlit>=1.24.0
plotly>=5.13.0
networkx>=3.0.0
```

### Installation

```bash
# Clone the repository
git clone [repository-url]

# Navigate to project directory
cd finite-velocity-diffusion

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
finite-velocity-diffusion/
│
├── app.py                     # Main Streamlit application
├── pages/                     # Additional pages for the multipage app
│   ├── 01_Parameter_Analysis.py   # Parameter analysis tools
│   ├── 02_Numerical_Analysis.py   # Numerical analysis tools
│   ├── 03_Graph_Diffusion.py      # Graph diffusion tools (planned)
│   └── components/                # Reusable UI components for pages
│       ├── parameter_analysis.py  # Parameter analysis components
│       ├── numerical_analysis.py  # Numerical analysis components
│       └── graph_diffusion.py     # Graph diffusion components (planned)
│
├── saved_params.json          # Persistent storage for user parameters
├── minimal_app.py             # Simplified version of the app
├── minimal_streamlit_app.py   # Another minimal example
├── run_simulation.py          # Command-line simulation runner
├── test_solver.py             # Tests for the solver
├── utils.py                   # Utility functions
│
├── solver/                    # Core numerical solvers
│   ├── __init__.py
│   ├── one_dimensional.py     # 1D solver implementation
│   ├── two_dimensional.py     # 2D solver implementation
│   └── graph_solver.py        # Graph-based diffusion solvers (planned)
│
├── simulation/                # Simulation management
│   ├── __init__.py
│   ├── analysis.py            # Analysis tools
│   ├── config.py              # Configuration management
│   ├── logging.py             # Logging utilities
│   ├── streamlit_simulation.py # Streamlit-specific simulation logic
│   └── workflow.py            # Workflow management
│
├── streamlit_ui/              # UI components
│   ├── __init__.py
│   ├── main.py                # Main UI layout
│   ├── sidebar.py             # Sidebar controls
│   └── state_management.py    # User session state persistence
│
└── visualization/             # Visualization components
    ├── __init__.py
    ├── streamlit_visualization.py  # Visualization functions for Streamlit
    └── graph_visualization.py      # Graph visualization functions (planned)
```

## Technical Constraints

1. **Performance Considerations**:
   - Computational intensity increases with grid resolution, especially in 2D
   - Graph-based simulations can be resource-intensive for large graphs
   - Interactive performance requires careful balance of accuracy vs. speed
   - Animation rendering can be memory-intensive for long simulations
   - Parameter analysis tools use approximations for interactive responsiveness

2. **Numerical Stability**:
   - Solutions must satisfy the Courant condition (`sqrt(D*dt/(tau*dx^2)) <= 1`)
   - Parameter ranges must be restricted to ensure stable solutions
   - Explicit schemes require smaller time steps than implicit methods

3. **Browser Limitations**:
   - Large animations may cause browser performance issues
   - Interactive visualizations of 2D solutions require optimization for responsiveness
   - Multipage app navigation may reset some Streamlit states

4. **Deployment Considerations**:
   - Streamlit Cloud has memory and computational limitations
   - Docker containerization may be needed for more complex deployments
   - Parameter persistence requires write access to the file system

## Technical Design Decisions

1. **Solver Implementation**:
   - Explicit finite difference schemes chosen for educational clarity
   - Central difference for spatial derivatives (2nd order accuracy)
   - Object-oriented design for solvers to encapsulate state and behavior

2. **UI Framework**:
   - Streamlit chosen for rapid development and interactive features
   - Multipage app structure for organizing different types of functionality
   - Reactive programming model simplifies state management
   - Component-based design for UI modularity

3. **Visualization Strategy**:
   - Multiple visualization options to accommodate different needs:
     - Static plots for simple analysis and export
     - Interactive sliders for detailed inspection
     - Animations for dynamic behavior demonstration
   - Parameter analysis tools with interactive visualization

4. **State Management**:
   - Stateful solver objects to maintain solution across time steps
   - Intermediate solution storage for time evolution analysis
   - Progress callback mechanism for long-running computations
   - File-based parameter persistence for remembering settings between sessions

5. **Parameter Persistence**:
   - JSON file format for simple, human-readable storage
   - Automatic saving when parameters are changed
   - Fallback to defaults if file is not available or corrupted
   - Centralized parameter management in session state

## Development Workflows

1. **Running the Main Application**:
   ```bash
   streamlit run app.py
   ```

2. **Testing**:
   ```bash
   python test_solver.py
   ```

3. **Simulation from Command Line**:
   ```bash
   python run_simulation.py
   ```

4. **Parameter Management**:
   - Parameters are saved automatically when changed through the UI
   - Default parameters are defined in utils.py
   - Parameter persistence is managed through streamlit_ui/state_management.py
   - saved_params.json stores the latest user settings

This technical foundation provides a flexible and maintainable system for numerical simulation of the finite velocity diffusion equation, with special emphasis on educational value and interactive exploration.
