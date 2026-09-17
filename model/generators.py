import numpy as np

class ControllableGenerator:
    def __init__(self, name, max_mw, min_mw, cost_per_mwh, carbon_per_mwh, ramp_rate_mw_per_min):
        self.name = name
        self.max_mw = max_mw
        self.min_mw = min_mw
        self.cost = cost_per_mwh
        self.carbon = carbon_per_mwh
        self.ramp_rate = ramp_rate_mw_per_min
        
        # State
        self.current_mw = min_mw

    def step(self, target_mw, dt_minutes):
        """Steps the physics forward by dt_minutes, attempting to reach target_mw"""
        target_mw = np.clip(target_mw, self.min_mw, self.max_mw)
        
        # Calculate max possible change in this time step
        max_change = self.ramp_rate * dt_minutes
        
        if target_mw > self.current_mw:
            self.current_mw = min(self.current_mw + max_change, target_mw)
        elif target_mw < self.current_mw:
            self.current_mw = max(self.current_mw - max_change, target_mw)
            
        return self.current_mw

    def get_operating_cost(self, dt_minutes):
        return (self.current_mw * self.cost) * (dt_minutes / 60.0)

    def get_emissions(self, dt_minutes):
        return (self.current_mw * self.carbon) * (dt_minutes / 60.0)

def create_generators():
    """Instantiates the specific setup requested."""
    return {
        "Coal-1": ControllableGenerator("Coal-1", 500, 100, 30.0, 0.82, ramp_rate_mw_per_min=5.0),
        "Coal-2": ControllableGenerator("Coal-2", 400, 80, 32.0, 0.82, ramp_rate_mw_per_min=4.0),
        "Gas": ControllableGenerator("Gas", 250, 25, 55.0, 0.45, ramp_rate_mw_per_min=20.0), # Fast ramping
        "Nuclear": ControllableGenerator("Nuclear", 900, 900, 10.0, 0.012, ramp_rate_mw_per_min=0.5) # Base load, very slow
    }
