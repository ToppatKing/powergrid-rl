import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Shared features
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        
        # Actor head: Outputs the mean of the action distribution.
        # We use Tanh to bound the mean between [-1, 1] 
        self.actor_mean = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        
        # We use a state-independent, learnable parameter for standard deviation
        # This controls exploration. It shrinks as the agent learns.
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic head: Outputs the expected total future reward (Value)
        self.critic = nn.Sequential(
            nn.Linear(128, 1)
        )
        
    def forward(self):
        raise NotImplementedError
        
    def act(self, state):
        shared_features = self.shared_net(state)
        action_mean = self.actor_mean(shared_features)
        
        # Create a Normal distribution to sample actions from
        action_std = torch.exp(self.actor_log_std)
        dist = Normal(action_mean, action_std)
        
        # Sample an action and calculate its log probability
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        
        # Get the Critic's value estimation for this state
        state_value = self.critic(shared_features)
        
        return action.detach(), action_logprob.detach(), state_value.detach()
    
    def evaluate(self, state, action):
        shared_features = self.shared_net(state)
        action_mean = self.actor_mean(shared_features)
        
        action_std = torch.exp(self.actor_log_std)
        dist = Normal(action_mean, action_std)
        
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(shared_features)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3, 
                 gamma=0.99, K_epochs=40, eps_clip=0.2):
        
        self.gamma = gamma          # Discount factor for future rewards
        self.eps_clip = eps_clip    # PPO clip parameter to prevent policy updates from being too large
        self.K_epochs = K_epochs    # Number of times to optimize over the batch
        
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam([
            {'params': self.policy.shared_net.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_mean.parameters(), 'lr': lr_actor},
            {'params': [self.policy.actor_log_std], 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, action_logprob, state_value = self.policy_old.act(state)
            
        # Store to memory
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        memory.values.append(state_value)
        
        # Action is bounded physically by the environment later, 
        # but we clamp it here to [-1, 1] purely for numerical stability
        return torch.clamp(action, -1.0, 1.0).flatten().numpy()

    def update(self, memory):
        # 1. Calculate Monte Carlo estimates of discounted returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalize returns for stability
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Convert memory lists to tensors
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach()
        old_values = torch.squeeze(torch.stack(memory.values, dim=0)).detach()
        
        # Calculate Advantages (Returns - Values)
        advantages = rewards.detach() - old_values
        
        # 2. Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluate old actions and values under the CURRENT policy
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            # Find the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # 3. Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Final loss of clipped objective value
            # Negative sign because we want to maximize the surrogate objective (gradient ascent)
            # Add value loss and entropy bonus (encourages exploration)
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # Backpropagate
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear memory
        memory.clear()
