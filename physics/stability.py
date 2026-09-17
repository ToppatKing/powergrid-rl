import numpy as np

class GridStability:
    def __init__(self, nominal_hz=60.0):
        self.nominal_hz = nominal_hz
        self.current_hz = nominal_hz
        
        # Damping factor (loads inherently draw less power if frequency drops)
        self.damping_D = 0.015  

    def calculate_system_inertia(self, generator_states):
        """
        Calculates total kinetic energy in the grid based on which 
        synchronous generators are online.
        """
        total_inertia = 0.0
        # Typical Inertia constants (H) in seconds
        inertia_constants = {
            "Coal-1": 4.0,
            "Coal-2": 4.0,
            "Gas": 2.5,
            "Nuclear": 6.0
        }
        
        for name, current_mw in generator_states.items():
            # If generator is producing more than its minimum, it is coupled to the grid
            if current_mw > 10.0 and name in inertia_constants:
                # Contribution to inertia is proportional to its capacity
                # (Assuming base MVA roughly equals max MW for simplicity)
                total_inertia += inertia_constants[name] * current_mw
                
        # Prevent division by zero if all synchronous generation is off
        # (This represents a tiny amount of synthetic inertia from smart inverters)
        return max(total_inertia, 50.0) 

    def step(self, total_gen_mw, total_load_mw, system_inertia, dt_seconds=1.0):
        """
        Applies the discrete Swing Equation.
        Total Gen and Load must include battery charging/discharging.
        """
        power_imbalance = total_gen_mw - total_load_mw
        
        # Delta frequency per unit (pu)
        df_pu = (self.current_hz - self.nominal_hz) / self.nominal_hz
        
        # Swing equation: 2H * d(df)/dt = P_gen - P_load - D * df
        # Rearranged for dt: d(df) = (P_imbalance - D * df) * dt / 2H
        
        rate_of_change_pu = (power_imbalance - (self.damping_D * df_pu * 1000)) / (2 * system_inertia)
        
        delta_f_pu = rate_of_change_pu * dt_seconds
        
        # Update frequency
        self.current_hz += delta_f_pu * self.nominal_hz
        
        return self.current_hz

    def is_blackout(self):
        """Standard under-frequency load shedding (UFLS) trip thresholds"""
        return self.current_hz < 59.3 or self.current_hz > 60.5
