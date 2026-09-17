import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Import our physical models
from models.generators import create_generators
from models.storage import BatteryStorage
from models.renewables import StochasticRenewable
from models.demand import DemandProfile
from physics.grid_graph import GridNetwork
from physics.power_flow import DCPowerFlow
from physics.stability import GridStability

class PowerGridEnv(gym.Env):
    """
    Custom Gymnasium Environment for Real-Time Power Grid Dispatch.
    The agent acts as the Automatic Generation Control (AGC) and Energy Management System.
    """
    def __init__(self):
        super(PowerGridEnv, self).__init__()
        
        # Simulation parameters
        self.dt_minutes = 1.0  # Agent makes a decision every 1 minute
        self.max_steps = int(24 * 60 / self.dt_minutes) # 1 full day per episode
        
        # 1. Action Space: [Coal-1, Coal-2, Gas, Nuclear, Battery]
        # Actions are continuous between [-1, 1] and will be scaled to MW targets
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        
        # 2. Observation Space: 16 variables representing grid state
        # Gen MWs (4), Solar MW, Wind MW, Demand MW, Battery SoC, Battery MW, 
        # Time of day, Grid Hz, Line Flows (5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.step_count = 0
        self.time_of_day_hours = 0.0 # Start at midnight
        
        # Instantiate physical entities
        self.generators = create_generators()
        self.battery = BatteryStorage()
        self.solar = StochasticRenewable("Solar", 300.0, is_solar=True)
        self.wind = StochasticRenewable("Wind", 400.0, is_solar=False)
        self.demand = DemandProfile()
        
        # Instantiate physics engines
        self.grid = GridNetwork()
        self.pf = DCPowerFlow(self.grid)
        self.stability = GridStability()
        
        return self._get_obs(), {}

    def _scale_action_to_mw(self, action):
        """Maps [-1, 1] RL actions to physical MW targets for each asset."""
        targets = {}
        
        def scale(val, min_val, max_val):
            # Convert [-1, 1] to [min_val, max_val]
            return min_val + (val + 1.0) / 2.0 * (max_val - min_val)
            
        targets["Coal-1"] = scale(action[0], self.generators["Coal-1"].min_mw, self.generators["Coal-1"].max_mw)
        targets["Coal-2"] = scale(action[1], self.generators["Coal-2"].min_mw, self.generators["Coal-2"].max_mw)
        targets["Gas"] = scale(action[2], self.generators["Gas"].min_mw, self.generators["Gas"].max_mw)
        targets["Nuclear"] = scale(action[3], self.generators["Nuclear"].min_mw, self.generators["Nuclear"].max_mw)
        
        # Battery target is charge (-200) to discharge (+200)
        targets["Battery"] = scale(action[4], -self.battery.max_power, self.battery.max_power)
        
        return targets

    def step(self, action):
        self.step_count += 1
        self.time_of_day_hours = (self.step_count * self.dt_minutes) / 60.0
        
        # 1. Unpack actions to physical targets
        targets = self._scale_action_to_mw(action)
        
        # 2. Step component physics forward
        gen_states = {}
        total_cost = 0.0
        total_carbon = 0.0
        
        for name, gen in self.generators.items():
            current_mw = gen.step(targets[name], self.dt_minutes)
            gen_states[name] = current_mw
            total_cost += gen.get_operating_cost(self.dt_minutes)
            total_carbon += gen.get_emissions(self.dt_minutes)
            
        bat_mw = self.battery.step(targets["Battery"], self.dt_minutes)
        solar_mw = self.solar.step(self.time_of_day_hours)
        wind_mw = self.wind.step(self.time_of_day_hours)
        demand_mw = self.demand.step(self.time_of_day_hours)
        
        # 3. Calculate Nodal Injections (Generation - Load)
        inj_bus0 = gen_states["Coal-1"] + gen_states["Coal-2"]
        inj_bus1 = gen_states["Gas"] + gen_states["Nuclear"]
        inj_bus2 = solar_mw + wind_mw + bat_mw
        inj_bus3 = -demand_mw
        
        injections = np.array([inj_bus0, inj_bus1, inj_bus2, inj_bus3])
        
        # 4. Solve DC Power Flow
        self.pf.solve(injections)
        
        # 5. Solve Grid Stability (Frequency)
        total_gen_mw = inj_bus0 + inj_bus1 + solar_mw + wind_mw + bat_mw
        inertia = self.stability.calculate_system_inertia(gen_states)
        hz = self.stability.step(total_gen_mw, demand_mw, inertia, dt_seconds=self.dt_minutes * 60)
        
        # 6. Calculate Reward
        reward, is_blackout = self._calculate_reward(total_cost, total_carbon)
        
        # 7. Check terminations
        terminated = is_blackout
        truncated = self.step_count >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _calculate_reward(self, total_cost, total_carbon):
        reward = 0.0
        is_blackout = False
        
        # Objective 1: Minimize cost and carbon (scaled down to keep reward variance manageable)
        reward -= (total_cost * 0.001)
        reward -= (total_carbon * 0.1)
        
        # Objective 2: Penalize Line Overloads (Thermal limits)
        overloads = self.grid.get_overloads()
        for overload in overloads:
            reward -= (overload['excess_mw'] * 0.05) # Heavy penalty for melting wires
            
        # Objective 3: Grid Stability (60 Hz is perfect)
        freq_error = abs(self.stability.current_hz - 60.0)
        if freq_error > 0.05: # Deadband: don't penalize tiny normal fluctuations
            reward -= (freq_error * 10.0) 
            
        # Fatal objective: Blackout
        if self.stability.is_blackout():
            reward -= 1000.0  # Massive penalty
            is_blackout = True
            
        return reward, is_blackout

    def _get_obs(self):
        """Constructs the observation vector."""
        obs = [
            self.generators["Coal-1"].current_mw,
            self.generators["Coal-2"].current_mw,
            self.generators["Gas"].current_mw,
            self.generators["Nuclear"].current_mw,
            self.solar.current_mw,
            self.wind.current_mw,
            self.demand.current_load,
            self.battery.get_soc(),
            self.battery.current_power_mw,
            self.time_of_day_hours,
            self.stability.current_hz
        ]
        
        # Add the 5 line flows
        for line in self.grid.lines:
            obs.append(line.current_flow)
            
        return np.array(obs, dtype=np.float32)
