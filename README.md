# Real-Time Power Grid RL Sandbox

A from-scratch reinforcement learning environment modeling dynamic electrical load distribution, AC impedance curves, thermal line limits, and grid frequency stability. The repository includes a custom Proximal Policy Optimization (PPO) agent that learns to act as an Automatic Generation Control (AGC) system.

##  Physics Engine Features
- **DC Power Flow**: Power routes based on network topology and line susceptance (Kirchhoff's laws), not just direct paths.
- **Swing Equation**: Accurately models grid frequency (Hz) based on the kinetic inertia of spinning fossil/nuclear turbines vs. zero-inertia renewables.
- **Asset Constraints**: Models realistic generator ramp rates (MW/min), min/max outputs, and battery round-trip efficiencies.
- **Stochastic Weather**: Uses Ornstein-Uhlenbeck processes for weather persistence in wind and solar output.

##  Grid Topology
The sandbox simulates a 4-bus ring network:
- **Bus 0 (Base)**: Coal-1 (500MW), Coal-2 (400MW)
- **Bus 1 (Fast/Base)**: Gas (250MW), Nuclear (900MW)
- **Bus 2 (Green/Storage)**: Wind (400MW), Solar (300MW), Battery (±200MW / 1000MWh)
- **Bus 3 (Demand)**: City center with a dynamic 1200 MW daily cyclical load profile.

##  Installation & Usage

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```
**2. Run the Training Loops**
```bash
python main.py 2>&1 | tee training.log
```
**3. View the Results**
```bash
python plot_results.py training.log
```

*Note: By default, the Pygame renderer will pop up to show every 10th episode. To train headless at maximum speed, change `RENDER = False` inside `main.py`.*

## Project Structure

power_grid_rl/
├── main.py                 # Entry point, training loop, and config
├── env/
│   └── grid_env.py         # Gymnasium wrapper connecting RL to physics
├── physics/
│   ├── grid_graph.py       # Topology, susceptance, thermal limits
│   ├── power_flow.py       # DC Power flow solver
│   └── stability.py        # Inertia and frequency calculations (Swing Eq)
├── models/
│   ├── generators.py       # Controllable fossil/nuclear plants
│   ├── renewables.py       # Weather-based stochastic assets
│   ├── storage.py          # Battery SoC and inverter limits
│   └── demand.py           # Daily diurnal load curves
├── rl/
│   ├── ppo.py              # Custom PPO Actor-Critic implementation (PyTorch)
│   └── memory.py           # Rollout buffer for on-policy learning
└── vis/
    └── renderer.py         # Pygame real-time dashboard

    
