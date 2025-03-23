# Project Brief: Finite Velocity Diffusion Solver

## Project Overview

The Finite Velocity Diffusion Solver is a numerical simulation tool that solves the telegrapher's equation (finite velocity diffusion equation) in one and two dimensions. Unlike classical diffusion equations where signals propagate at infinite speeds, the telegrapher's equation models physical phenomena with a finite propagation speed, making it more physically realistic for many applications.

## Core Equation

The finite velocity diffusion equation (telegrapher's equation) is given by:

```
τ∂²u/∂t² + ∂u/∂t = D∇²u
```

where:
- u is the concentration/density field
- τ is the relaxation time
- D is the diffusion coefficient
- ∇² is the Laplacian operator

## Core Requirements

1. **Numerical Solvers**: Implement finite difference solvers for both 1D and 2D cases of the telegrapher's equation.
2. **Interactive Visualization**: Provide an interactive web interface using Streamlit to visualize and explore solutions.
3. **Comparative Analysis**: Allow users to compare solutions with classical diffusion for insight into the differences.
4. **Parameter Exploration**: Support adjustment of key parameters (diffusion coefficient, relaxation time, etc.) to explore their effects.
5. **Time Evolution**: Visualize how solutions evolve over time through static plots, interactive sliders, or animations.

## Key Features

- 1D and 2D numerical solvers using finite difference methods
- Interactive parameter configuration via Streamlit
- Multiple visualization options (static, interactive slider, animation)
- Comparison with classical diffusion solutions
- Time evolution analysis
- Stability checking to ensure numerical validity

## Success Criteria

- Accurate solutions verified against analytical cases where available
- Stable numerical implementation with appropriate error handling
- Intuitive, responsive user interface
- Comprehensive visualization options
- Educational value for understanding wave-like diffusion phenomena

This project serves as both a scientific tool for exploring non-classical diffusion and an educational resource for understanding the behavior of hyperbolic partial differential equations.
