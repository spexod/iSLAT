"""
Debug configuration and utilities for iSLAT.

This module provides centralized debug level control for various components
of the iSLAT spectroscopy tool.
"""

import enum
from typing import Optional, Dict, Any

class DebugLevel(enum.IntEnum):
    """Debug levels for controlling output verbosity."""
    NONE = 0      # No debug output
    ERROR = 1     # Only errors and critical issues
    WARNING = 2   # Warnings and errors
    INFO = 3      # General information, warnings, and errors
    VERBOSE = 4   # Detailed information including cache operations
    TRACE = 5     # Maximum verbosity including internal operations

class DebugConfig:
    """
    Centralized debug configuration for iSLAT components.
    
    This class manages debug levels for different subsystems and provides
    utilities for conditional debug output.
    """
    
    def __init__(self):
        self._global_level = DebugLevel.WARNING
        self._component_levels: Dict[str, DebugLevel] = {}
        
        # Default component levels
        self._component_levels.update({
            'molecule_cache': DebugLevel.WARNING,
            'plot_renderer': DebugLevel.WARNING,
            'main_plot': DebugLevel.WARNING,
            'data_processor': DebugLevel.WARNING,
            'molecule_dict': DebugLevel.WARNING,
            'parameter_changes': DebugLevel.WARNING,
            'summed_flux': DebugLevel.WARNING,
            'intensity_calc': DebugLevel.WARNING,
            'spectrum_calc': DebugLevel.WARNING,
            'active_molecule': DebugLevel.WARNING,
            'line_inspection': DebugLevel.WARNING,
        })
    
    @property
    def global_level(self) -> DebugLevel:
        """Get the global debug level."""
        return self._global_level
    
    @global_level.setter
    def global_level(self, level: DebugLevel) -> None:
        """Set the global debug level."""
        if isinstance(level, int):
            level = DebugLevel(level)
        self._global_level = level
    
    def set_component_level(self, component: str, level: DebugLevel) -> None:
        """Set debug level for a specific component."""
        if isinstance(level, int):
            level = DebugLevel(level)
        self._component_levels[component] = level
    
    def get_component_level(self, component: str) -> DebugLevel:
        """Get debug level for a specific component."""
        return self._component_levels.get(component, self._global_level)
    
    def should_log(self, component: str, level: DebugLevel) -> bool:
        """Check if a message should be logged for the given component and level."""
        component_level = self.get_component_level(component)
        return level <= component_level
    
    def log(self, component: str, level: DebugLevel, message: str, **kwargs) -> None:
        """Log a message if the debug level permits."""
        if self.should_log(component, level):
            prefix = self._get_level_prefix(level)
            component_prefix = f"[{component.upper()}]"
            full_message = f"{prefix}{component_prefix} {message}"
            
            # Add any additional context
            if kwargs:
                context_str = " | ".join(f"{k}={v}" for k, v in kwargs.items())
                full_message += f" | {context_str}"
            
            print(full_message)
    
    def _get_level_prefix(self, level: DebugLevel) -> str:
        """Get prefix string for debug level."""
        prefixes = {
            DebugLevel.ERROR: "[ERROR] ",
            DebugLevel.WARNING: "[WARN] ",
            DebugLevel.INFO: "[INFO] ",
            DebugLevel.VERBOSE: "[DEBUG] ",
            DebugLevel.TRACE: "[TRACE] "
        }
        return prefixes.get(level, "")
    
    def error(self, component: str, message: str, **kwargs) -> None:
        """Log an error message."""
        self.log(component, DebugLevel.ERROR, message, **kwargs)
    
    def warning(self, component: str, message: str, **kwargs) -> None:
        """Log a warning message."""
        self.log(component, DebugLevel.WARNING, message, **kwargs)
    
    def info(self, component: str, message: str, **kwargs) -> None:
        """Log an info message."""
        self.log(component, DebugLevel.INFO, message, **kwargs)
    
    def verbose(self, component: str, message: str, **kwargs) -> None:
        """Log a verbose debug message."""
        self.log(component, DebugLevel.VERBOSE, message, **kwargs)
    
    def trace(self, component: str, message: str, **kwargs) -> None:
        """Log a trace message."""
        self.log(component, DebugLevel.TRACE, message, **kwargs)
    
    def enable_cache_debugging(self) -> None:
        """Enable detailed cache debugging output."""
        self.set_component_level('molecule_cache', DebugLevel.VERBOSE)
        self.set_component_level('plot_renderer', DebugLevel.VERBOSE)
        self.set_component_level('summed_flux', DebugLevel.VERBOSE)
        self.set_component_level('parameter_changes', DebugLevel.VERBOSE)
    
    def disable_cache_debugging(self) -> None:
        """Disable detailed cache debugging output."""
        self.set_component_level('molecule_cache', DebugLevel.WARNING)
        self.set_component_level('plot_renderer', DebugLevel.WARNING)
        self.set_component_level('summed_flux', DebugLevel.WARNING)
        self.set_component_level('parameter_changes', DebugLevel.WARNING)
    
    def enable_all_debugging(self) -> None:
        """Enable maximum debugging for all components."""
        self.global_level = DebugLevel.TRACE
        for component in self._component_levels.keys():
            self.set_component_level(component, DebugLevel.TRACE)
    
    def disable_all_debugging(self) -> None:
        """Disable all debugging output."""
        self.global_level = DebugLevel.NONE
        for component in self._component_levels.keys():
            self.set_component_level(component, DebugLevel.NONE)
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of current debug settings."""
        return {
            'global_level': self._global_level.name,
            'global_value': self._global_level.value,
            'component_levels': {
                comp: level.name for comp, level in self._component_levels.items()
            }
        }

# Global debug configuration instance
debug_config = DebugConfig()

def get_debug_config() -> DebugConfig:
    """Get the global debug configuration instance."""
    return debug_config

# Convenience functions for common debug operations
def set_cache_debug_level(level: DebugLevel) -> None:
    """Set debug level for all cache-related components."""
    debug_config.set_component_level('molecule_cache', level)
    debug_config.set_component_level('plot_renderer', level)
    debug_config.set_component_level('summed_flux', level)

def set_plotting_debug_level(level: DebugLevel) -> None:
    """Set debug level for all plotting-related components."""
    debug_config.set_component_level('plot_renderer', level)
    debug_config.set_component_level('main_plot', level)

def set_molecule_debug_level(level: DebugLevel) -> None:
    """Set debug level for all molecule-related components."""
    debug_config.set_component_level('molecule_cache', level)
    debug_config.set_component_level('molecule_dict', level)
    debug_config.set_component_level('parameter_changes', level)
    debug_config.set_component_level('intensity_calc', level)
    debug_config.set_component_level('spectrum_calc', level)

def set_active_molecule_debug_level(level: DebugLevel) -> None:
    """Set debug level for active molecule change tracking."""
    debug_config.set_component_level('active_molecule', level)
    debug_config.set_component_level('line_inspection', level)