"""
State management utilities for the Streamlit app.

This module provides functions for managing persistent parameters across
browser sessions using file-based storage.
"""

import streamlit as st
import json
import os
from utils import DEFAULT_PARAMETERS

# File to store saved parameters
PARAMS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_params.json")

# Default parameter values
DEFAULT_VALUES = {
    "dimension": "1D",
    "D": DEFAULT_PARAMETERS['D'],
    "tau": DEFAULT_PARAMETERS['tau'],
    "amplitude": 1.0,
    "sigma": 1.0,
    "num_steps": 100,
    "show_evolution": False,
    "evolution_steps": 20,
    "viz_type": "Static Plots",
    "animation_speed": 10
}

def get_parameter_key(param_name):
    """Generate a consistent key for storing parameters in session state"""
    return f"saved_{param_name}"

def initialize_session_state():
    """
    Initialize the session state with values from the saved file or defaults.
    
    Returns:
    --------
    dict
        Dictionary containing initialized parameters
    """
    # Check if we've already loaded parameters in this session
    if "params_loaded" not in st.session_state:
        # Load saved parameters from file
        saved_params = load_params_from_file()
        
        # Update session state with loaded parameters
        for param, value in saved_params.items():
            key = get_parameter_key(param)
            st.session_state[key] = value
        
        st.session_state["params_loaded"] = True
    
    # Return parameters from session state
    params = {}
    for param in DEFAULT_VALUES.keys():
        key = get_parameter_key(param)
        params[param] = st.session_state.get(key, DEFAULT_VALUES[param])
    
    return params

def save_parameters(params):
    """
    Save parameters to session state and file for persistence.
    
    Parameters:
    -----------
    params : dict
        Dictionary containing parameter values to save
    """
    # Update session state
    for param, value in params.items():
        key = get_parameter_key(param)
        st.session_state[key] = value
    
    # Save to file
    save_params_to_file(params)

def load_params_from_file():
    """
    Load parameters from saved file.
    
    Returns:
    --------
    dict
        Loaded parameters or defaults if file doesn't exist
    """
    try:
        if os.path.exists(PARAMS_FILE):
            with open(PARAMS_FILE, 'r') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading parameters: {e}")
    
    return DEFAULT_VALUES.copy()

def save_params_to_file(params):
    """
    Save parameters to file.
    
    Parameters:
    -----------
    params : dict
        Parameters to save
    """
    try:
        with open(PARAMS_FILE, 'w') as f:
            json.dump(params, f, indent=2)
    except IOError as e:
        print(f"Error saving parameters: {e}")

def clear_saved_parameters():
    """
    Clear all saved parameters.
    """
    # Remove file if it exists
    if os.path.exists(PARAMS_FILE):
        try:
            os.remove(PARAMS_FILE)
        except IOError as e:
            print(f"Error removing parameter file: {e}")
    
    # Clear from session state
    for param in DEFAULT_VALUES.keys():
        key = get_parameter_key(param)
        if key in st.session_state:
            del st.session_state[key]
