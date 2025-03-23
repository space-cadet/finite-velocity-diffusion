# Finite Velocity Diffusion Solver

This project implements a numerical solver for the finite velocity diffusion equation (telegrapher's equation) in one and two dimensions.

## Equation

The finite velocity diffusion equation is given by:

τ∂²u/∂t² + ∂u/∂t = D∇²u

where:
- u is the concentration/density
- τ is the relaxation time
- D is the diffusion coefficient
- ∇² is the Laplacian operator

## Features

- 1D and 2D numerical solutions using finite difference methods
- Interactive visualization using Streamlit
- Comparison with classical diffusion equation
- Parameter configuration and analysis tools

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

## Project Structure

- `solver/`: Core numerical solver implementation
  - `one_dimensional.py`: 1D finite velocity diffusion solver
  - `two_dimensional.py`: 2D finite velocity diffusion solver
- `visualization/`: Plotting and visualization utilities
- `app.py`: Streamlit web interface
- `utils.py`: Utility functions and constants 