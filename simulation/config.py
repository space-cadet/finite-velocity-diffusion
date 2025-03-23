"""
Configuration utilities for finite velocity diffusion simulations.

This module provides classes and functions for managing simulation configurations,
including parameter validation, saving, and loading.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, Union, ClassVar
from pathlib import Path
import numpy as np

from utils import DEFAULT_PARAMETERS


@dataclass
class SimulationParameters:
    """
    Class for managing simulation parameters with validation.
    
    This class provides a structured way to store, validate, and manipulate
    simulation parameters for the finite velocity diffusion solver.
    """
    # Physical parameters
    D: float = DEFAULT_PARAMETERS['D']  # Diffusion coefficient
    tau: float = DEFAULT_PARAMETERS['tau']  # Relaxation time
    
    # Domain parameters
    nx: int = DEFAULT_PARAMETERS['nx']  # Number of x grid points
    ny: int = DEFAULT_PARAMETERS['ny']  # Number of y grid points
    dx: float = DEFAULT_PARAMETERS['dx']  # x step size
    dy: float = DEFAULT_PARAMETERS['dy']  # y step size
    x_min: float = DEFAULT_PARAMETERS['x_min']  # Minimum x value
    x_max: float = DEFAULT_PARAMETERS['x_max']  # Maximum x value
    y_min: float = DEFAULT_PARAMETERS['y_min']  # Minimum y value
    y_max: float = DEFAULT_PARAMETERS['y_max']  # Maximum y value
    
    # Time parameters
    dt: float = DEFAULT_PARAMETERS['dt']  # Time step size
    
    # Initial condition parameters
    amplitude: float = 1.0  # Initial amplitude
    sigma: float = 1.0  # Initial standard deviation (for Gaussian)
    
    # Simulation run parameters
    num_steps: int = 100  # Number of time steps
    
    # Visualization parameters
    show_evolution: bool = False  # Whether to show time evolution
    evolution_steps: int = 20  # Number of time points to show
    viz_type: str = "Static Plots"  # Visualization type
    animation_speed: int = 10  # Animation speed (frames per second)
    
    # Class constants
    VIZ_TYPES: ClassVar[list] = ["Static Plots", "Interactive Slider", "Plotly Animation"]
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert the parameters to a dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationParameters':
        """Create a SimulationParameters instance from a dictionary."""
        # Filter out any keys that aren't parameters in our class
        valid_keys = {field_name for field_name in cls.__dataclass_fields__}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)
    
    def save(self, filename: str) -> None:
        """
        Save the parameters to a JSON file.
        
        Parameters:
            filename: The file path to save to
        """
        with open(filename, 'w') as f:
            json.dump(self.as_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filename: str) -> 'SimulationParameters':
        """
        Load parameters from a JSON file.
        
        Parameters:
            filename: The file path to load from
            
        Returns:
            A new SimulationParameters instance with the loaded values
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file is not valid JSON
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def validate(self) -> None:
        """
        Validate the parameter values.
        
        Raises:
            ValueError: If any parameters are invalid
        """
        # Physical parameters
        if self.D <= 0:
            raise ValueError("Diffusion coefficient D must be positive")
        if self.tau <= 0:
            raise ValueError("Relaxation time tau must be positive")
        
        # Domain parameters
        if self.nx <= 1 or self.ny <= 1:
            raise ValueError("Number of grid points must be greater than 1")
        if self.dx <= 0 or self.dy <= 0:
            raise ValueError("Spatial step sizes must be positive")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Domain max values must be greater than min values")
        
        # Time parameters
        if self.dt <= 0:
            raise ValueError("Time step must be positive")
        
        # Simulation run parameters
        if self.num_steps <= 0:
            raise ValueError("Number of time steps must be positive")
        
        # Visualization parameters
        if self.evolution_steps <= 0:
            raise ValueError("Number of evolution steps must be positive")
        if self.viz_type not in self.VIZ_TYPES:
            raise ValueError(f"Visualization type must be one of {self.VIZ_TYPES}")
        if self.animation_speed <= 0:
            raise ValueError("Animation speed must be positive")
    
    def get_courant_numbers(self) -> Union[float, tuple]:
        """
        Calculate the Courant numbers for the current parameters.
        
        Returns:
            For 1D: a single Courant number
            For 2D: a tuple (courant_x, courant_y)
        """
        courant_x = np.sqrt(self.D * self.dt / (self.tau * self.dx**2))
        courant_y = np.sqrt(self.D * self.dt / (self.tau * self.dy**2))
        
        # Return a tuple for 2D, single value for 1D
        return (courant_x, courant_y)
    
    def get_propagation_speed(self) -> float:
        """
        Calculate the propagation speed for the current parameters.
        
        Returns:
            The propagation speed c = sqrt(D/tau)
        """
        return np.sqrt(self.D / self.tau)
    
    def is_stable(self) -> bool:
        """
        Check if the simulation is numerically stable.
        
        Returns:
            True if the Courant numbers are <= 1, False otherwise
        """
        courant = self.get_courant_numbers()
        if isinstance(courant, tuple):
            return courant[0] <= 1 and courant[1] <= 1
        return courant <= 1


class ConfigurationManager:
    """
    Manager for handling multiple simulation configurations.
    
    This class provides functionality to manage, save, and load multiple
    simulation configurations, including preset configurations.
    """
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Parameters:
            config_dir: Directory to store configuration files (default: ./configs)
        """
        if config_dir is None:
            # Default to a configs directory in the project root
            self.config_dir = Path("./configs")
        else:
            self.config_dir = Path(config_dir)
        
        # Create the directory if it doesn't exist
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Dictionary to store presets
        self.presets: Dict[str, SimulationParameters] = {}
        
        # Load default presets
        self._add_default_presets()
    
    def _add_default_presets(self) -> None:
        """Add default preset configurations."""
        # Wave-like behavior preset
        wave_preset = SimulationParameters(
            D=5.0,
            tau=2.0,
            num_steps=200,
            amplitude=1.5,
            sigma=0.5
        )
        self.presets["Wave-like Behavior"] = wave_preset
        
        # Pure diffusion preset (small tau)
        diffusion_preset = SimulationParameters(
            D=1.0,
            tau=0.1,
            num_steps=200,
            amplitude=1.0,
            sigma=0.8
        )
        self.presets["Pure Diffusion"] = diffusion_preset
        
        # High resolution preset
        hires_preset = SimulationParameters(
            nx=200,
            ny=200,
            dx=0.05,
            dy=0.05,
            dt=0.0005,
            num_steps=300,
            show_evolution=True,
            evolution_steps=30
        )
        self.presets["High Resolution"] = hires_preset
        
        # Fast computation preset
        fast_preset = SimulationParameters(
            nx=50,
            ny=50,
            dx=0.2,
            dy=0.2,
            dt=0.002,
            num_steps=50
        )
        self.presets["Fast Computation"] = fast_preset
    
    def get_preset(self, name: str) -> SimulationParameters:
        """
        Get a preset configuration by name.
        
        Parameters:
            name: Name of the preset
            
        Returns:
            A copy of the preset SimulationParameters
            
        Raises:
            KeyError: If the preset name doesn't exist
        """
        if name not in self.presets:
            raise KeyError(f"Preset '{name}' not found")
        
        # Return a copy of the preset
        preset_dict = self.presets[name].as_dict()
        return SimulationParameters.from_dict(preset_dict)
    
    def get_preset_names(self) -> list:
        """Get the names of all available presets."""
        return list(self.presets.keys())
    
    def save_configuration(self, params: SimulationParameters, name: str) -> str:
        """
        Save a configuration to a file.
        
        Parameters:
            params: The SimulationParameters to save
            name: Name to use for the saved configuration
            
        Returns:
            The full path to the saved file
        """
        # Validate the parameters before saving
        params.validate()
        
        # Create filename
        filename = self.config_dir / f"{name.replace(' ', '_').lower()}.json"
        
        # Save to file
        params.save(str(filename))
        
        return str(filename)
    
    def load_configuration(self, name: str) -> SimulationParameters:
        """
        Load a configuration from a file.
        
        Parameters:
            name: Name of the configuration to load
            
        Returns:
            The loaded SimulationParameters
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
        """
        # Check if it's a preset
        if name in self.presets:
            return self.get_preset(name)
        
        # Otherwise, load from file
        filename = self.config_dir / f"{name.replace(' ', '_').lower()}.json"
        
        if not filename.exists():
            raise FileNotFoundError(f"Configuration file '{filename}' not found")
        
        return SimulationParameters.load(str(filename))
    
    def list_saved_configurations(self) -> list:
        """
        List all saved configuration files.
        
        Returns:
            A list of configuration names (without the .json extension)
        """
        configs = []
        
        for file in self.config_dir.glob("*.json"):
            # Convert filename_with_underscores.json to "Filename With Underscores"
            name = file.stem.replace('_', ' ').title()
            configs.append(name)
        
        return configs
    
    def delete_configuration(self, name: str) -> None:
        """
        Delete a saved configuration.
        
        Parameters:
            name: Name of the configuration to delete
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If attempting to delete a built-in preset
        """
        # Check if it's a preset
        if name in self.presets:
            raise ValueError(f"Cannot delete built-in preset '{name}'")
        
        # Otherwise, delete the file
        filename = self.config_dir / f"{name.replace(' ', '_').lower()}.json"
        
        if not filename.exists():
            raise FileNotFoundError(f"Configuration file '{filename}' not found")
        
        os.remove(filename)
