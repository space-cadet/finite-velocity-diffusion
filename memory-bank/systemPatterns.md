# System Patterns: Finite Velocity Diffusion Solver

## Architecture Overview

The system follows a modular architecture with clear separation of concerns:

```
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│   Streamlit   │      │  Simulation   │      │    Solver     │
│  Multipage UI │─────▶│    Logic      │─────▶│    Engines    │
└───────────────┘      └───────────────┘      └───────────────┘
        │                      ▲                      │
        │                      │                      │
        ▼                      │                      ▼
┌───────────────┐             │              ┌───────────────┐
│     User      │             │              │   Utils &     │
│   Controls    │             └──────────────│   Helpers     │
└───────────────┘                            └───────────────┘
```

## Key Components

### 1. Solver Engines

The core numerical solvers implement the finite difference methods for the telegrapher's equation:

- **FiniteVelocityDiffusion1D**: Solves the 1D telegrapher's equation using a finite difference scheme
- **FiniteVelocityDiffusion2D**: Extends the solution to 2D domains
- **GraphDiffusionSolver**: (Planned) Implements diffusion processes on graph structures

These classes encapsulate:
- Domain discretization
- Time stepping algorithms
- Boundary condition handling
- Stability checking
- Solution storage and retrieval

### 2. Simulation Logic

Bridges between the UI and solvers:

- Configures solvers with user-defined parameters
- Manages simulation runs and captures intermediate results
- Provides progress updates during computation
- Prepares comparison data (e.g., classical diffusion solutions)
- Organizes results for visualization

### 3. Streamlit Multipage UI

Organized into a multipage structure with modular components:

- **Main App (app.py)**: Primary simulation interface for running and visualizing simulations
- **Parameter Analysis Page**: Tools for exploring parameter relationships
- **Numerical Analysis Page**: Tools for analyzing numerical properties
- **Components**: Reusable UI components for each page

Each page contains:
- **Content Area**: Primary display area for visualizations and results
- **Sidebar Controls**: Parameter inputs and simulation options
- **Analysis Tools**: Specialized tools for the page's purpose

### 4. Utilities and Helpers

Common functionality used across the system:

- Initial condition generators (e.g., Gaussian profiles)
- Analytical solutions for comparison
- Default parameter configurations
- Helper functions for visualization
- State management for parameter persistence

## Design Patterns

1. **Factory Pattern**: Used to create appropriate solver instances based on dimension selection

2. **Strategy Pattern**: Different visualization strategies (static, slider, animation) with a common interface

3. **Observer Pattern**: Progress callback mechanism to update UI during lengthy computations

4. **Repository Pattern**: State management for solver configurations and results

5. **Component-Based UI**: Modular UI components for reusability and maintainability

## Data Flow

1. User selects a page from the navigation menu
2. User sets parameters through the Streamlit controls
3. UI invokes simulation logic or analysis tools with these parameters
4. Results flow back to visualization components
5. UI renders visualizations based on results and user preferences
6. Parameters are saved to persistent storage for future sessions

## Technical Decisions

1. **Multipage Structure**: Division of functionality into specialized pages for clarity and organization

2. **Explicit Time Stepping**: Used for simplicity and educational transparency

3. **Finite Difference Method**: Chosen for intuitive implementation and educational value

4. **Central Difference Scheme**: Second-order accuracy for spatial derivatives

5. **Stability Checking**: Runtime validation of Courant condition to prevent unstable solutions

6. **Neumann Boundary Conditions**: Used as default to simulate semi-infinite domains

7. **Componentized UI**: Modular UI design for maintainability and separation of concerns

8. **Reactive Visualization**: Immediate update of visuals when parameters change

9. **Parameter Persistence**: File-based storage for remembering user settings between sessions

## Page Organization

1. **Main App (app.py)**:
   - Primary simulation interface
   - Parameter controls for simulation configuration
   - Visualization of simulation results
   - Parameter persistence via file-based storage

2. **Parameter Analysis Page (01_Parameter_Analysis.py)**:
   - Propagation speed analysis tool
   - Wave-diffusion spectrum visualization
   - Stability region analysis
   - Educational information about parameter relationships

3. **Numerical Analysis Page (02_Numerical_Analysis.py)**:
   - Convergence analysis tools
   - Error analysis tools
   - Educational information about numerical properties

4. **Graph Diffusion Page (03_Graph_Diffusion.py)** (Planned):
   - Graph construction and visualization
   - Ordinary diffusion on graphs
   - Finite-velocity diffusion on graphs
   - Optional: Diffusion with potential term
   - Comparative analysis between different diffusion models on graphs

5. **Component Structure**:
   - Each page has a dedicated components module
   - Reusable UI components are organized by functionality
   - Clear separation between UI construction and logic

This architecture ensures a clean separation between numerical algorithms, simulation management, and user interface, making the system maintainable and extensible.
