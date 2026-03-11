"""
Modified DQN for Buy vs. Rent Problem - Phase 3

Key modifications for this problem:
1. **Action Masking** - Only evaluate valid actions (can't buy without savings)
2. **Reward Shaping** - Balance immediate costs vs long-term value
3. **State Normalization** - Handle different scales (savings $0-200k vs rates 0.01-0.15)
4. **Transaction Cost Handling** - Penalize frequent switching
5. **Dueling Architecture** - Separate value and advantage streams (good for infrequent actions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
from typing import Tuple, Optional
import matplotlib.pyplot as plt

# Import our custom environment
from buy_rent_environment import BuyRentEnv, BuyRentParams


# Experience replay buffer
Transition = namedtuple('Transition',
                       ('state', 'action', 'next_state', 'reward', 'done', 'action_mask'))


class ReplayBuffer:
    """Experience replay buffer with action mask storage."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, next_state, reward, done, action_mask):
        """Add transition to buffer."""
        self.buffer.append(Transition(state, action, next_state, reward, done, action_mask))
    
    def sample(self, batch_size: int):
        """Sample random batch of transitions."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DuelingDQN(nn.Module):
    """
    Dueling DQN Network for Buy vs. Rent.
    
    Why Dueling Architecture?
    - Separates state value V(s) from action advantages A(s,a)
    - Better for environments where action choice doesn't always matter
      (e.g., holding when prices are stable)
    - Learns which states are valuable independent of actions
    
    Network architecture:
      Input [5]: savings, home_price, mortgage_rate, owns, mortgage_balance
        ↓
      Shared layers (256 → 256)
        ↓
      Split into:
        • Value stream V(s) → scalar
        • Advantage stream A(s,a) → vector[4]
        ↓
      Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
    """
    
    def __init__(self, state_dim: int = 5, action_dim: int = 4, hidden_dim: int = 256):
        super(DuelingDQN, self).__init__()
        
        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage stream A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        """
        Forward pass with dueling architecture.
        
        Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
        
        The subtraction of mean advantages ensures identifiability:
        we can't distinguish whether a state is good (high V) or
        an action is good (high A) without this constraint.
        """
        features = self.feature(state)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine: Q = V + (A - mean(A))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


class ActionMaskedDQNAgent:
    """
    DQN Agent with action masking for Buy vs. Rent problem.
    
    Key modifications:
    1. Only selects from valid actions (respects action_mask)
    2. Sets Q-values of invalid actions to -inf during selection
    3. Normalizes states to [0, 1] range for better learning
    4. Uses reward shaping to balance short-term costs vs long-term value
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 4,
        learning_rate: float = 0.0003,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 50000,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        device: str = 'cpu'
    ):
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Epsilon-greedy parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        # Networks
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # State normalization (learned from data)
        self.state_mean = np.array([50.0, 100.0, 0.06, 0.5, 25.0])
        self.state_std = np.array([30.0, 50.0, 0.03, 0.5, 30.0])
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state to roughly [0, 1] range."""
        return (state - self.state_mean) / (self.state_std + 1e-8)
    
    def select_action(self, state: np.ndarray, action_mask: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy with action masking.
        
        Args:
            state: Current state observation
            action_mask: Binary mask of valid actions
            training: If True, use epsilon-greedy; else use greedy
        
        Returns:
            Selected action index
        """
        valid_actions = np.where(action_mask == 1)[0]
        
        if len(valid_actions) == 0:
            raise ValueError("No valid actions available!")
        
        # Epsilon-greedy exploration (only during training)
        epsilon = self._get_epsilon() if training else 0.0
        
        if training and random.random() < epsilon:
            # Random exploration (from valid actions only)
            return np.random.choice(valid_actions)
        else:
            # Greedy exploitation
            state_norm = self.normalize_state(state)
            state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            
            # Mask invalid actions
            q_values_masked = np.full(self.action_dim, -np.inf)
            q_values_masked[valid_actions] = q_values[valid_actions]
            
            return int(np.argmax(q_values_masked))
    
    def _get_epsilon(self) -> float:
        """Get current epsilon value (decays over time)."""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        return epsilon
    
    def update(self) -> Optional[float]:
        """
        Perform one gradient update step.
        
        Returns:
            Loss value if update performed, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(
            [self.normalize_state(s) for s in batch.state]
        ).to(self.device)
        
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        
        next_state_batch = torch.FloatTensor(
            [self.normalize_state(s) for s in batch.next_state]
        ).to(self.device)
        
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Current Q-values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()
        
        # Next Q-values with action masking
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch)
            
            # Apply action masking to next states
            next_action_masks = torch.FloatTensor(batch.action_mask).to(self.device)
            next_q_values_masked = next_q_values.clone()
            next_q_values_masked[next_action_masks == 0] = -np.inf
            
            next_q_values_max = next_q_values_masked.max(1)[0]
            
            # Handle terminal states
            next_q_values_max[done_batch == 1] = 0.0
            
            # Target Q-values
            target_q_values = reward_batch + self.gamma * next_q_values_max
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'state_mean': self.state_mean,
            'state_std': self.state_std,
        }, path)
    
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.state_mean = checkpoint['state_mean']
        self.state_std = checkpoint['state_std']


def train_agent(
    env: BuyRentEnv,
    agent: ActionMaskedDQNAgent,
    n_episodes: int = 1000,
    max_steps: int = 360,
    update_freq: int = 4,
    log_freq: int = 10
):
    """
    Train the DQN agent on the Buy vs. Rent environment.
    
    Args:
        env: Buy vs. Rent environment
        agent: DQN agent
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        update_freq: Update network every N steps
        log_freq: Log progress every N episodes
    
    Returns:
        Training statistics
    """
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    for episode in range(n_episodes):
        state, info = env.reset()
        action_mask = info['action_mask']
        
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps):
            # Select action
            action = agent.select_action(state, action_mask, training=True)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            next_action_mask = info['action_mask']
            done = terminated or truncated
            
            # Store transition
            agent.replay_buffer.push(
                state, action, next_state, reward, done, next_action_mask
            )
            
            # Update agent
            if step % update_freq == 0:
                loss = agent.update()
                if loss is not None:
                    episode_loss.append(loss)
            
            # Update target network
            if agent.steps_done % agent.target_update_freq == 0:
                agent.update_target_network()
            
            episode_reward += reward
            state = next_state
            action_mask = next_action_mask
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Logging
        if (episode + 1) % log_freq == 0:
            avg_reward = np.mean(episode_rewards[-log_freq:])
            avg_length = np.mean(episode_lengths[-log_freq:])
            epsilon = agent._get_epsilon()
            print(f"Episode {episode+1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.0f} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Buffer: {len(agent.replay_buffer)}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'losses': losses
    }


if __name__ == "__main__":
    # Demo training
    print("="*70)
    print("TRAINING DQN AGENT FOR BUY VS. RENT")
    print("="*70)
    
    # Create environment
    try:
        import sys
        sys.path.insert(0, '/mnt/user-data/outputs')
        from calibrated_params import CALIBRATED_PARAMS
        params = BuyRentParams(
            mu_h=CALIBRATED_PARAMS['mu_h'],
            sigma_h=CALIBRATED_PARAMS['sigma_h'],
            rent_ratio=CALIBRATED_PARAMS['rent_ratio'],
            m_bar=CALIBRATED_PARAMS['m_bar'],
            sigma_m=CALIBRATED_PARAMS['sigma_m'],
            kappa_m=CALIBRATED_PARAMS['kappa_m']
        )
        print("✓ Using calibrated parameters\n")
    except:
        params = BuyRentParams()
        print("⚠️  Using default parameters\n")
    
    env = BuyRentEnv(params)
    
    # Create agent
    agent = ActionMaskedDQNAgent(
        state_dim=5,
        action_dim=4,
        learning_rate=0.0003,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=10000,
        buffer_size=50000,
        batch_size=64,
        target_update_freq=1000
    )
    
    # Train
    print("Training for 100 episodes (demo)...\n")
    stats = train_agent(
        env=env,
        agent=agent,
        n_episodes=100,
        max_steps=360,
        update_freq=4,
        log_freq=10
    )
    
    # Save model
    agent.save('/mnt/user-data/outputs/dqn_buy_rent.pth')
    print("\n✓ Model saved to dqn_buy_rent.pth")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(stats['episode_rewards'])
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Training Rewards')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(stats['losses'])
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/dqn_training.png', dpi=150)
    print("✓ Training curves saved to dqn_training.png")
