import numpy as np
import time
from env.grid_env import PowerGridEnv
from rl.ppo import PPO
from rl.memory import RolloutBuffer
from vis.renderer import GridRenderer

def train():
    # Setup hyperparameters
    max_episodes = 5000
    max_steps = 1440 # 24 hours at 1-minute steps
    update_timestep = 2000 # Update PPO policy every N timesteps
    
    # Enable rendering (Set to False for much faster headless training)
    RENDER = True 
    RENDER_EVERY = 10 # Watch every 10th episode

    env = PowerGridEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    memory = RolloutBuffer()
    ppo = PPO(state_dim, action_dim)
    
    if RENDER:
        renderer = GridRenderer()
        
    timestep = 0
    
    print("Starting Training on Real-Time Power Grid Sandbox...")
    
    for ep in range(1, max_episodes + 1):
        state, _ = env.reset()
        ep_reward = 0
        
        do_render = RENDER and (ep % RENDER_EVERY == 0)
        
        for t in range(max_steps):
            timestep += 1
            
            # 1. Agent picks an action
            action = ppo.select_action(state, memory)
            
            # 2. Environment steps forward
            state, reward, done, trunc, _ = env.step(action)
            
            # 3. Store rewards
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            ep_reward += reward
            
            # 4. Render
            if do_render:
                renderer.render(env, ep, t, ep_reward)
                # Sleep briefly so the human eye can see the simulation
                time.sleep(0.01) 
                
            # 5. PPO Update
            if timestep % update_timestep == 0:
                ppo.update(memory)
                timestep = 0
                
            if done or trunc:
                break
                
        # Logging
        if ep % 10 == 0:
            print(f"Episode: {ep:4d} | Return: {ep_reward:8.2f} | Final Hz: {env.stability.current_hz:.2f} | Time: {env.time_of_day_hours:.1f}h")

if __name__ == "__main__":
    train()
