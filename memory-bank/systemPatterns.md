# System Patterns: Finite Velocity Diffusion Solver

## Architecture Overview

The system follows a modular architecture with clear separation of concerns:

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Streamlit  │      │ Simulation  │      │   Solver    │
│     UI      │─────▶│   Logic     │─────▶│   Engines   │
└─────────────┘      └─────────────┘      └─────────────┘
       │                    ▲                    │
       │                    │                    │
       ▼                    │                    ▼
┌─────────────┐             │             ┌─────────────┐
│    User     │             │             │   Utils &   │
│  Controls   │             └─────────────│   Helpers   │
└─────────────┘                           └─────────────┘
```

## Key Components

### 1. Solver Engines

The core numerical solvers implement the finite difference methods for the telegrapher's equation:

- **FiniteVelocityDiffusion1D**: Solves the 1D telegrapher's equation using a finite difference scheme
- **FiniteVelocityDiffusion2D**: Extends the solution to 2D domains

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

### 3. Streamlit UI

Organized into modular components:

- **Main Content**: Primary display area for visualizations and results
- **Sidebar Controls**: Parameter inputs and simulation options
- **Visualization Components**: Various plot types and display options
- **State Management**: Persistence of user settings between sessions using file-based storage

### 4. Utilities and Helpers

Common functionality used across the system:

- Initial condition generators (e.g., Gaussian profiles)
- Analytical solutions for comparison
- Default parameter configurations
- Helper functions for visualization

## Design Patterns

1. **Factory Pattern**: Used to create appropriate solver instances based on dimension selection

2. **Strategy Pattern**: Different visualization strategies (static, slider, animation) with a common interface

3. **Observer Pattern**: Progress callback mechanism to update UI during lengthy computations

4. **Repository Pattern**: State management for solver configurations and results

## Data Flow

1. User sets parameters through the Streamlit sidebar
2. UI invokes simulation logic with these parameters
3. Simulation configures and runs the appropriate solver
4. Solver computes numerical solution with progress updates
5. Results flow back to visualization components
6. UI renders visualizations based on results and user preferences

## Technical Decisions

1. **Explicit Time Stepping**: Used for simplicity and educational transparency

2. **Finite Difference Method**: Chosen for intuitive implementation and educational value

3. **Central Difference Scheme**: Second-order accuracy for spatial derivatives

4. **Stability Checking**: Runtime validation of Courant condition to prevent unstable solutions

5. **Neumann Boundary Conditions**: Used as default to simulate semi-infinite domains

6. **Componentized UI**: Modular UI design for maintainability and separation of concerns

7. **Reactive Visualization**: Immediate update of visuals when parameters change

This architecture ensures a clean separation between numerical algorithms, simulation management, and user interface, making the system maintainable and extensible.
