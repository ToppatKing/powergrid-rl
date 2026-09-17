import numpy as np
import math

class DemandProfile:
    def __init__(self, base_load=1200.0, daily_variance=300.0):
        self.base_load = base_load
        self.variance = daily_variance
        self.current_load = base_load

    def step(self, time_of_day_hours):
        # 1st Peak at ~9:00, 2nd (higher) Peak at ~19:00
        # Model using sum of sine waves
        t = time_of_day_hours
        
        # Diurnal curve
        morning_peak = math.exp(-0.1 * (t - 9)**2) * 0.6
        evening_peak = math.exp(-0.08 * (t - 19)**2) * 1.0
        night_dip = -math.exp(-0.05 * (t - 3)**2) * 0.8
        
        shape = morning_peak + evening_peak + night_dip
        
        # Add high-frequency Gaussian noise (demand jitter)
        noise = np.random.normal(0, 15.0) # +/- 15 MW jitter
        
        self.current_load = self.base_load + (shape * self.variance) + noise
        
        # Hard physical limit so demand doesn't drop below unrealistic levels
        self.current_load = max(self.current_load, 500.0) 
        return self.current_load
