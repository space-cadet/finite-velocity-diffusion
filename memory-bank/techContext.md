# Technical Context: Finite Velocity Diffusion Solver

## Technology Stack

The project is built using the following key technologies:

1. **Python**: Core programming language for all components
2. **NumPy**: Foundation for numerical computations and array operations
3. **SciPy**: Used for specialized scientific computing functions
4. **Matplotlib**: Static plotting and visualization
5. **Plotly**: Interactive plots and animations
6. **Streamlit**: Web application framework for the interactive UI

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
├── minimal_app.py             # Simplified version of the app
├── minimal_streamlit_app.py   # Another minimal example
├── run_simulation.py          # Command-line simulation runner
├── test_solver.py             # Tests for the solver
├── utils.py                   # Utility functions
│
├── solver/                    # Core numerical solvers
│   ├── __init__.py
│   ├── one_dimensional.py     # 1D solver implementation
│   └── two_dimensional.py     # 2D solver implementation
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
│   └── sidebar.py             # Sidebar controls
│
└── visualization/             # Visualization components
    ├── __init__.py
    └── ...                    # Visualization modules
```

## Technical Constraints

1. **Performance Considerations**:
   - Computational intensity increases with grid resolution, especially in 2D
   - Interactive performance requires careful balance of accuracy vs. speed
   - Animation rendering can be memory-intensive for long simulations

2. **Numerical Stability**:
   - Solutions must satisfy the Courant condition (`sqrt(D*dt/(tau*dx^2)) <= 1`)
   - Parameter ranges must be restricted to ensure stable solutions
   - Explicit schemes require smaller time steps than implicit methods

3. **Browser Limitations**:
   - Large animations may cause browser performance issues
   - Interactive visualizations of 2D solutions require optimization for responsiveness

4. **Deployment Considerations**:
   - Streamlit Cloud has memory and computational limitations
   - Docker containerization may be needed for more complex deployments

## Technical Design Decisions

1. **Solver Implementation**:
   - Explicit finite difference schemes chosen for educational clarity
   - Central difference for spatial derivatives (2nd order accuracy)
   - Object-oriented design for solvers to encapsulate state and behavior

2. **UI Framework**:
   - Streamlit chosen for rapid development and interactive features
   - Reactive programming model simplifies state management
   - Component-based design for UI modularity

3. **Visualization Strategy**:
   - Multiple visualization options to accommodate different needs:
     - Static plots for simple analysis and export
     - Interactive sliders for detailed inspection
     - Animations for dynamic behavior demonstration

4. **State Management**:
   - Stateful solver objects to maintain solution across time steps
   - Intermediate solution storage for time evolution analysis
   - Progress callback mechanism for long-running computations

## Development Workflows

1. **Running the Application**:
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

This technical foundation provides a flexible and maintainable system for numerical simulation of the finite velocity diffusion equation.
