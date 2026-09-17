import numpy as np
import math

class StochasticRenewable:
    def __init__(self, name, max_mw, is_solar=False):
        self.name = name
        self.max_mw = max_mw
        self.is_solar = is_solar
        self.current_mw = 0.0
        
        # Ornstein-Uhlenbeck process parameters for weather persistence
        self.theta = 0.15
        self.mu = 0.5
        self.sigma = 0.2
        self.noise_state = 0.5

    def _ou_step(self):
        # Mean-reverting random walk
        dw = np.random.normal(0, 1)
        self.noise_state += self.theta * (self.mu - self.noise_state) + self.sigma * dw
        self.noise_state = np.clip(self.noise_state, 0, 1)
        return self.noise_state

    def step(self, time_of_day_hours):
        noise = self._ou_step()
        
        if self.is_solar:
            # Solar curve: 0 outside 6:00-18:00, sine wave peak at 12:00
            if 6.0 <= time_of_day_hours <= 18.0:
                normalized_time = (time_of_day_hours - 6.0) / 12.0
                ideal_solar = math.sin(normalized_time * math.pi)
            else:
                ideal_solar = 0.0
                
            # Apply cloud cover noise (reduces ideal output)
            self.current_mw = ideal_solar * self.max_mw * (0.4 + 0.6 * noise)
        else:
            # Wind is pure stochastic but persistent
            self.current_mw = self.max_mw * noise
            
        return self.current_mw
