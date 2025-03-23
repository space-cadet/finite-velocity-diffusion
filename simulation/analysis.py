"""
Analysis functions for finite velocity diffusion simulations.

This module provides functions for analyzing simulation results, including
error estimation, convergence testing, and physical interpretation.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Union
import matplotlib.pyplot as plt

from simulation.workflow import SimulationResult, SimulationConfig


def calculate_max_error(result: SimulationResult) -> float:
    """
    Calculate the maximum absolute error between finite velocity and classical diffusion.
    
    Parameters:
        result: SimulationResult containing both solutions
    
    Returns:
        Maximum absolute error value
        
    Raises:
        ValueError: If classical solution is not available
    """
    if result.u_classical is None:
        raise ValueError("Classical diffusion solution not available for error calculation")
    
    return np.max(np.abs(result.u - result.u_classical))


def calculate_rmse(result: SimulationResult) -> float:
    """
    Calculate the root mean square error between finite velocity and classical diffusion.
    
    Parameters:
        result: SimulationResult containing both solutions
    
    Returns:
        Root mean square error value
        
    Raises:
        ValueError: If classical solution is not available
    """
    if result.u_classical is None:
        raise ValueError("Classical diffusion solution not available for error calculation")
    
    return np.sqrt(np.mean((result.u - result.u_classical) ** 2))


def calculate_energy(result: SimulationResult) -> Union[float, np.ndarray]:
    """
    Calculate the energy of the solution (integral of u²).
    
    Parameters:
        result: SimulationResult containing the solution
        
    Returns:
        Total energy value or energy time series if time evolution is available
    """
    if result.dimension == "1D":
        # Calculate energy for 1D case
        dx = result.config.dx
        energy = np.sum(result.u ** 2) * dx
        
        # Calculate energy time series if available
        if result.solutions is not None:
            energy_series = np.array([np.sum(sol ** 2) * dx for sol in result.solutions])
            return energy_series
        
        return energy
    else:
        # Calculate energy for 2D case
        dx = result.config.dx
        dy = result.config.dy
        energy = np.sum(result.u ** 2) * dx * dy
        
        # Calculate energy time series if available
        if result.solutions is not None:
            energy_series = np.array([np.sum(sol ** 2) * dx * dy for sol in result.solutions])
            return energy_series
        
        return energy


def calculate_convergence_rate(
    base_config: SimulationConfig,
    dimension: str,
    refinements: List[int] = [1, 2, 4, 8]
) -> Dict[str, Union[List[float], List[int]]]:
    """
    Calculate the convergence rate by running simulations with different grid resolutions.
    
    Parameters:
        base_config: Base configuration to refine
        dimension: "1D" or "2D"
        refinements: List of refinement factors for the grid
    
    Returns:
        Dictionary containing resolution sizes and corresponding errors
    """
    from simulation.workflow import run_simulation
    
    # Storage for results
    dx_values = []
    errors = []
    
    # Loop through refinements
    for refine in refinements:
        # Create refined configuration
        refined_config = SimulationConfig(**base_config.__dict__)
        
        # Update grid parameters
        refined_config.nx *= refine
        refined_config.dx /= refine
        
        if dimension == "2D":
            refined_config.ny *= refine
            refined_config.dy /= refine
        
        # Run simulation with refined grid
        result = run_simulation(refined_config, dimension)
        
        # Calculate error
        if result.u_classical is not None:
            error = calculate_rmse(result)
            
            # Store results
            dx_values.append(refined_config.dx)
            errors.append(error)
    
    return {
        "dx_values": dx_values,
        "errors": errors,
        "refinements": refinements
    }


def analyze_wave_propagation(result: SimulationResult) -> Dict[str, float]:
    """
    Analyze wave propagation characteristics in the simulation.
    
    Parameters:
        result: SimulationResult, should include time evolution data
        
    Returns:
        Dictionary with wave propagation metrics
        
    Raises:
        ValueError: If time evolution data is not available
    """
    if result.times is None or result.solutions is None:
        raise ValueError("Time evolution data required for wave propagation analysis")
    
    # Calculate theoretical propagation speed
    theoretical_speed = result.propagation_speed
    
    # Track the peak position over time
    peak_positions = []
    for sol in result.solutions:
        if result.dimension == "1D":
            peak_idx = np.argmax(sol)
            peak_pos = result.x[peak_idx]
        else:
            # For 2D, find the maximum and get its coordinates
            max_idx = np.unravel_index(np.argmax(sol), sol.shape)
            peak_pos = (result.x[max_idx[1]], result.y[max_idx[0]])
        
        peak_positions.append(peak_pos)
    
    # Calculate observed wave speed
    if result.dimension == "1D" and len(result.times) >= 2:
        # For 1D, calculate speed from position differences
        speeds = []
        for i in range(1, len(result.times)):
            if result.times[i] == result.times[i-1]:
                continue  # Skip if time difference is zero
                
            dt = result.times[i] - result.times[i-1]
            dx = peak_positions[i] - peak_positions[i-1]
            if dt > 0:
                speeds.append(abs(dx) / dt)
        
        observed_speed = np.mean(speeds) if speeds else 0
    else:
        # For 2D or insufficient time points, just use theoretical
        observed_speed = theoretical_speed
    
    # Calculate additional metrics
    metrics = {
        "theoretical_speed": theoretical_speed,
        "observed_speed": observed_speed,
        "speed_ratio": observed_speed / theoretical_speed if theoretical_speed > 0 else 0,
        "final_time": result.final_time,
    }
    
    return metrics


def calculate_numerical_stability(result: SimulationResult) -> Dict[str, float]:
    """
    Calculate numerical stability metrics for the simulation.
    
    Parameters:
        result: SimulationResult to analyze
        
    Returns:
        Dictionary with stability metrics
    """
    # Extract configuration
    config = result.config
    
    # Calculate Courant numbers
    if result.dimension == "1D":
        courant = np.sqrt(config.D * config.dt / (config.tau * config.dx**2))
        courant_x, courant_y = courant, 0.0
    else:
        courant_x = np.sqrt(config.D * config.dt / (config.tau * config.dx**2))
        courant_y = np.sqrt(config.D * config.dt / (config.tau * config.dy**2))
    
    # Calculate maximum allowed dt for stability
    max_dt_x = config.tau * config.dx**2 / config.D
    if result.dimension == "2D":
        max_dt_y = config.tau * config.dy**2 / config.D
        max_dt = min(max_dt_x, max_dt_y)
    else:
        max_dt = max_dt_x
    
    # Check if time evolution data is available
    if result.solutions is not None:
        # Check for oscillations in the solution
        energy = calculate_energy(result)
        if isinstance(energy, np.ndarray) and len(energy) > 3:
            # Calculate energy differences
            energy_diff = np.diff(energy)
            # Count sign changes in energy differences (oscillations)
            sign_changes = np.sum(energy_diff[1:] * energy_diff[:-1] < 0)
            oscillation_metric = sign_changes / (len(energy_diff) - 1)
        else:
            oscillation_metric = 0.0
    else:
        oscillation_metric = 0.0
    
    # Compile stability metrics
    stability_metrics = {
        "courant_x": courant_x,
        "courant_y": courant_y if result.dimension == "2D" else 0.0,
        "max_allowed_dt": max_dt,
        "dt_ratio": config.dt / max_dt if max_dt > 0 else float('inf'),
        "is_stable": courant_x <= 1.0 and (result.dimension == "1D" or courant_y <= 1.0),
        "oscillation_metric": oscillation_metric
    }
    
    return stability_metrics


def calculate_solution_properties(result: SimulationResult) -> Dict[str, float]:
    """
    Calculate various properties of the solution.
    
    Parameters:
        result: SimulationResult to analyze
        
    Returns:
        Dictionary with solution properties
    """
    # Calculate total mass (integral of u)
    if result.dimension == "1D":
        dx = result.config.dx
        mass = np.sum(result.u) * dx
    else:
        dx = result.config.dx
        dy = result.config.dy
        mass = np.sum(result.u) * dx * dy
    
    # Calculate peak value and location
    peak_value = np.max(result.u)
    
    if result.dimension == "1D":
        peak_idx = np.argmax(result.u)
        peak_location = result.x[peak_idx]
        
        # Calculate width at half maximum
        half_max = peak_value / 2
        above_half_max = result.u >= half_max
        if np.any(above_half_max):
            indices = np.where(above_half_max)[0]
            width = (indices[-1] - indices[0]) * dx
        else:
            width = 0.0
    else:
        # For 2D, find the maximum and get its coordinates
        max_idx = np.unravel_index(np.argmax(result.u), result.u.shape)
        peak_location = (result.x[max_idx[1]], result.y[max_idx[0]])
        
        # Calculating width in 2D is more complex, use approximation
        threshold = peak_value / 2
        above_threshold = result.u >= threshold
        width = np.sqrt(np.sum(above_threshold) * dx * dy)  # Approximate area above half max
    
    # Calculate second moment (variance) for 1D case
    if result.dimension == "1D":
        # Calculate center of mass
        x_center = np.sum(result.u * result.x) / np.sum(result.u) if np.sum(result.u) > 0 else 0
        
        # Calculate variance
        variance = np.sum(result.u * (result.x - x_center)**2) / np.sum(result.u) if np.sum(result.u) > 0 else 0
        
        # Standard deviation (width measure)
        std_dev = np.sqrt(variance)
    else:
        # For 2D case, just use width approximation
        std_dev = width / 2.355  # Convert FWHM to sigma for Gaussian
    
    properties = {
        "total_mass": mass,
        "peak_value": peak_value,
        "width": width if result.dimension == "1D" else width,
        "standard_deviation": std_dev
    }
    
    # Add peak location as string or tuple
    if result.dimension == "1D":
        properties["peak_location"] = float(peak_location)
    else:
        properties["peak_location_x"] = float(peak_location[0])
        properties["peak_location_y"] = float(peak_location[1])
    
    return properties


def compare_with_analytical(result: SimulationResult) -> Dict[str, float]:
    """
    Compare the numerical solution with the analytical solution.
    
    For the finite velocity diffusion problem, analytical solutions are generally
    not available for arbitrary initial conditions. This function compares with
    classical diffusion and analyzes the differences.
    
    Parameters:
        result: SimulationResult to analyze
        
    Returns:
        Dictionary with comparison metrics
        
    Raises:
        ValueError: If classical solution is not available
    """
    if result.u_classical is None:
        raise ValueError("Classical diffusion solution not available for comparison")
    
    # Basic error metrics
    max_error = calculate_max_error(result)
    rmse = calculate_rmse(result)
    
    # Calculate relative error
    u_max = np.max(result.u_classical)
    if u_max > 0:
        relative_error = rmse / u_max
    else:
        relative_error = 0.0
    
    # Calculate correlation coefficient
    u_mean = np.mean(result.u)
    u_classical_mean = np.mean(result.u_classical)
    
    numerator = np.sum((result.u - u_mean) * (result.u_classical - u_classical_mean))
    denominator = np.sqrt(np.sum((result.u - u_mean)**2) * np.sum((result.u_classical - u_classical_mean)**2))
    
    if denominator > 0:
        correlation = numerator / denominator
    else:
        correlation = 0.0
    
    comparison = {
        "max_error": max_error,
        "rmse": rmse,
        "relative_error": relative_error,
        "correlation": correlation
    }
    
    return comparison


def analyze_result(result: SimulationResult) -> Dict[str, Dict[str, float]]:
    """
    Perform comprehensive analysis of simulation results.
    
    Parameters:
        result: SimulationResult to analyze
        
    Returns:
        Dictionary with various analysis metrics grouped by category
    """
    analysis = {}
    
    # Basic solution properties
    analysis["solution_properties"] = calculate_solution_properties(result)
    
    # Numerical stability analysis
    analysis["stability"] = calculate_numerical_stability(result)
    
    # Compare with classical diffusion if available
    if result.u_classical is not None:
        analysis["analytical_comparison"] = compare_with_analytical(result)
    
    # Wave propagation analysis if time evolution data is available
    if result.times is not None and result.solutions is not None:
        analysis["wave_propagation"] = analyze_wave_propagation(result)
    
    return analysis
