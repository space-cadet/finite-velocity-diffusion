"""
Logging utilities for finite velocity diffusion simulations.

This module provides structured logging capabilities for debugging,
performance monitoring, and simulation tracking.
"""

import logging
import time
import os
import json
from dataclasses import asdict
from typing import Dict, Any, Optional, Union
from pathlib import Path

from simulation.workflow import SimulationConfig


class SimulationLogger:
    """
    Logger for finite velocity diffusion simulations.
    
    This class provides methods for logging simulation events, errors,
    and performance metrics.
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_level: int = logging.INFO,
        console_output: bool = True
    ):
        """
        Initialize the simulation logger.
        
        Parameters:
            log_dir: Directory for log files (default: ./logs)
            log_level: Logging level (default: INFO)
            console_output: Whether to also output to console (default: True)
        """
        self.log_dir = Path(log_dir) if log_dir else Path("./logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create a unique log file name based on timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_file = self.log_dir / f"simulation_{timestamp}.log"
        
        # Configure the logger
        self.logger = logging.getLogger("finite_velocity_diffusion")
        self.logger.setLevel(log_level)
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create a file handler
        file_handler = logging.FileHandler(self.log_file)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Add console handler if requested
        if console_output:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                "%(levelname)s: %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Store metrics for performance tracking
        self.metrics = {
            "simulation_count": 0,
            "total_runtime": 0.0,
            "errors": 0
        }
        
        # Timestamps for performance tracking
        self.start_time = None
    
    def start_simulation(self, config: SimulationConfig, dimension: str) -> None:
        """
        Log the start of a simulation.
        
        Parameters:
            config: Simulation configuration
            dimension: "1D" or "2D"
        """
        self.start_time = time.time()
        self.metrics["simulation_count"] += 1
        
        # Convert config to a serializable format
        config_dict = asdict(config)
        
        # Log basic info
        self.logger.info(f"Starting {dimension} simulation with parameters:")
        self.logger.info(f"  D = {config.D}, tau = {config.tau}")
        self.logger.info(f"  Grid: nx = {config.nx}, dx = {config.dx}")
        if dimension == "2D":
            self.logger.info(f"  Grid: ny = {config.ny}, dy = {config.dy}")
        self.logger.info(f"  Time: dt = {config.dt}, steps = {config.num_steps}")
        
        # Save full configuration to a JSON file
        config_file = self.log_dir / f"config_{self.metrics['simulation_count']}.json"
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def end_simulation(self, success: bool = True) -> float:
        """
        Log the end of a simulation.
        
        Parameters:
            success: Whether the simulation completed successfully
            
        Returns:
            Runtime in seconds
        """
        if self.start_time is None:
            self.logger.warning("end_simulation called without prior start_simulation")
            return 0.0
        
        runtime = time.time() - self.start_time
        self.metrics["total_runtime"] += runtime
        
        if success:
            self.logger.info(f"Simulation completed in {runtime:.3f} seconds")
        else:
            self.metrics["errors"] += 1
            self.logger.error(f"Simulation failed after {runtime:.3f} seconds")
        
        self.start_time = None
        return runtime
    
    def log_progress(self, progress: float, message: str) -> None:
        """
        Log simulation progress.
        
        Parameters:
            progress: Progress as a fraction (0.0 to 1.0)
            message: Status message
        """
        self.logger.info(f"Progress {progress*100:.1f}%: {message}")
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error that occurred during simulation.
        
        Parameters:
            error: The exception that occurred
            context: Optional contextual information about the error
        """
        self.metrics["errors"] += 1
        
        if context:
            self.logger.error(f"Error: {error}. Context: {context}")
        else:
            self.logger.error(f"Error: {error}")
    
    def log_analysis(self, analysis_results: Dict[str, Any]) -> None:
        """
        Log analysis results.
        
        Parameters:
            analysis_results: Dictionary of analysis results
        """
        self.logger.info("Analysis results:")
        
        # Log a summary of key metrics
        for category, metrics in analysis_results.items():
            self.logger.info(f"  {category}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float, str, bool)):
                    self.logger.info(f"    {metric}: {value}")
        
        # Save full results to a JSON file
        analysis_file = self.log_dir / f"analysis_{self.metrics['simulation_count']}.json"
        
        # Convert any non-serializable objects to strings
        serializable_results = self._make_serializable(analysis_results)
        
        with open(analysis_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert objects to JSON-serializable formats.
        
        Parameters:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            if obj.size <= 100:  # Only convert small arrays
                return obj.tolist()
            else:
                return f"<ndarray of shape {obj.shape}>"
        elif hasattr(obj, '__dict__'):
            return str(obj)  # Convert objects to strings
        else:
            return obj
    
    def get_metrics(self) -> Dict[str, Union[int, float]]:
        """
        Get performance metrics for all simulations.
        
        Returns:
            Dictionary of metrics
        """
        # Calculate average runtime
        if self.metrics["simulation_count"] > 0:
            avg_runtime = self.metrics["total_runtime"] / self.metrics["simulation_count"]
        else:
            avg_runtime = 0.0
        
        metrics = {
            **self.metrics,
            "average_runtime": avg_runtime,
            "success_rate": (self.metrics["simulation_count"] - self.metrics["errors"]) / max(1, self.metrics["simulation_count"])
        }
        
        return metrics


# Create a default logger instance for convenience
default_logger = SimulationLogger()
