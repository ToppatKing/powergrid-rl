import numpy as np

class BatteryStorage:
    def __init__(self, max_power_mw=200, capacity_mwh=1000, efficiency=0.92):
        self.max_power = max_power_mw
        self.capacity = capacity_mwh
        self.efficiency = efficiency
        
        # State
        self.current_energy_mwh = capacity_mwh / 2.0  # Start at 50% SoC
        self.current_power_mw = 0.0 # Positive = discharging to grid, Negative = charging

    def step(self, target_power_mw, dt_minutes):
        # Clip to physical inverter limits
        target_power_mw = np.clip(target_power_mw, -self.max_power, self.max_power)
        
        dt_hours = dt_minutes / 60.0
        
        if target_power_mw > 0: # Discharging (providing power to grid)
            energy_needed = target_power_mw * dt_hours
            if self.current_energy_mwh >= energy_needed:
                self.current_energy_mwh -= energy_needed
                self.current_power_mw = target_power_mw
            else:
                # Battery is empty
                self.current_power_mw = self.current_energy_mwh / dt_hours
                self.current_energy_mwh = 0.0
                
        elif target_power_mw < 0: # Charging (taking power from grid)
            charge_power = abs(target_power_mw)
            energy_added = (charge_power * dt_hours) * self.efficiency
            available_space = self.capacity - self.current_energy_mwh
            
            if available_space >= energy_added:
                self.current_energy_mwh += energy_added
                self.current_power_mw = target_power_mw
            else:
                # Battery is full
                actual_energy_added = available_space
                self.current_power_mw = -(actual_energy_added / self.efficiency) / dt_hours
                self.current_energy_mwh = self.capacity
        else:
            self.current_power_mw = 0.0
            
        return self.current_power_mw

    def get_soc(self):
        return self.current_energy_mwh / self.capacity
