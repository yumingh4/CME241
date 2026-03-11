"""
Buy vs. Rent Environment - Phase 3
Generates sampling traces from the MDP for RL training.

This is a Gym-compatible environment that wraps the Phase 1/2 MDP
and provides the interface needed for modern RL libraries.
"""

import numpy as np
import gym
from gym import spaces
from dataclasses import dataclass, replace
from typing import Tuple, Dict, Optional
import pandas as pd


@dataclass(frozen=True)
class BuyRentState:
    """State of the Buy vs. Rent MDP."""
    savings: float
    home_price: float
    rent: float
    mortgage_rate: float
    owns: bool
    mortgage_balance: float = 0.0
    purchase_price: float = 0.0


@dataclass
class BuyRentParams:
    """Parameters calibrated from real data."""
    # Dynamics (load from calibrated_params.py if available)
    mu_h: float = 0.03
    sigma_h: float = 0.10
    kappa_m: float = 0.20
    m_bar: float = 0.06
    sigma_m: float = 0.01
    rent_ratio: float = 0.004
    sigma_r: float = 0.0005
    income: float = 5.0
    risk_free: float = 0.03
    
    # Costs
    down_payment_frac: float = 0.20
    closing_cost_frac: float = 0.03
    selling_cost_frac: float = 0.06
    
    # Utility
    ownership_utility: float = 0.5
    gamma: float = 0.95
    
    # Environment settings
    max_steps: int = 360  # 30 years * 12 months
    dt: float = 1/12      # monthly time step


class BuyRentEnv(gym.Env):
    """
    Gym environment for the Buy vs. Rent MDP.
    
    Observation space: [savings, home_price, mortgage_rate, owns, mortgage_balance]
    Action space: Discrete(4) - {RENT, BUY, STAY, SELL}
    
    Key modifications for this problem:
    1. Action masking - only valid actions available in each state
    2. State normalization - handle different scales (savings vs rates)
    3. Reward shaping - balance immediate costs vs long-term value
    """
    
    metadata = {'render.modes': ['human']}
    
    # Action encoding
    RENT_STAY = 0
    BUY = 1
    OWN_STAY = 2
    SELL = 3
    
    def __init__(self, params: Optional[BuyRentParams] = None, 
                 initial_state: Optional[BuyRentState] = None):
        super().__init__()
        
        self.params = params or BuyRentParams()
        self.p = self.params
        
        # Default initial state
        if initial_state is None:
            initial_state = BuyRentState(
                savings=20.0,          # $20k
                home_price=100.0,      # $100k home
                rent=0.4,              # $400/month (~0.4% of home price)
                mortgage_rate=0.06,    # 6%
                owns=False
            )
        self.initial_state = initial_state
        
        # Observation space: [savings, home_price, mortgage_rate, owns, mortgage_balance]
        # Note: rent is derived from home_price so not included
        self.observation_space = spaces.Box(
            low=np.array([0, 20, 0.01, 0, 0]),
            high=np.array([200, 300, 0.15, 1, 200]),
            dtype=np.float32
        )
        
        # Action space: 4 discrete actions
        self.action_space = spaces.Discrete(4)
        
        # Episode tracking
        self.state = None
        self.steps = 0
        self.episode_data = []
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        self.state = self.initial_state
        self.steps = 0
        self.episode_data = []
        
        obs = self._state_to_obs(self.state)
        info = self._get_info()
        
        return obs, info
    
    def _state_to_obs(self, state: BuyRentState) -> np.ndarray:
        """Convert state to observation vector."""
        return np.array([
            state.savings,
            state.home_price,
            state.mortgage_rate,
            float(state.owns),
            state.mortgage_balance
        ], dtype=np.float32)
    
    def _get_action_mask(self, state: BuyRentState) -> np.ndarray:
        """Return binary mask of valid actions in current state."""
        mask = np.zeros(4, dtype=np.float32)
        
        if state.owns:
            # Can only STAY or SELL when owning
            mask[self.OWN_STAY] = 1
            mask[self.SELL] = 1
        else:
            # Always can RENT
            mask[self.RENT_STAY] = 1
            
            # Can BUY if have enough savings
            cost = (self.p.down_payment_frac + self.p.closing_cost_frac) * state.home_price
            if state.savings >= cost:
                mask[self.BUY] = 1
        
        return mask
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take action in environment.
        
        Returns:
            obs, reward, terminated, truncated, info
        """
        # Check if action is valid
        action_mask = self._get_action_mask(self.state)
        if action_mask[action] == 0:
            # Invalid action - return large negative reward
            obs = self._state_to_obs(self.state)
            return obs, -100.0, True, False, {"invalid_action": True}
        
        # Execute action
        next_state, reward = self._transition(self.state, action)


        # --- ADD LIQUIDITY PENALTY HERE ---
        safety_buffer = 15.0  # $15k safety net
        if next_state.savings < safety_buffer:
            # Penalize based on how far below the buffer the agent is
            # Squaring it makes "extreme poverty" much more painful for the agent
            penalty = 0.1 * (safety_buffer - next_state.savings)**2
            reward -= penalty
        # ----------------------------------

        # Update state
        self.state = next_state
        
        # Update state
        self.state = next_state
        self.steps += 1
        
        # Record trajectory
        self.episode_data.append({
            'step': self.steps,
            'action': action,
            'reward': reward,
            'savings': next_state.savings,
            'home_price': next_state.home_price,
            'mortgage_rate': next_state.mortgage_rate,
            'owns': next_state.owns
        })
        
        # Check termination conditions
        terminated = (
            self.state.savings < 0 or  # Bankruptcy
            self.steps >= self.p.max_steps  # Time horizon reached
        )
        
        truncated = False  # No truncation in this environment
        
        obs = self._state_to_obs(self.state)
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _transition(self, state: BuyRentState, action: int) -> Tuple[BuyRentState, float]:
        """Execute state transition and return (next_state, reward)."""
        p = self.p
        
        # --- Evolve stochastic variables ---
        eps_h = np.random.randn()
        eps_m = np.random.randn()
        
        # Home price (Geometric Brownian Motion)
        new_h = state.home_price * np.exp(
            (p.mu_h - 0.5 * p.sigma_h**2) * p.dt + p.sigma_h * np.sqrt(p.dt) * eps_h
        )
        
        # Mortgage rate (Vasicek / OU process)
        new_m = state.mortgage_rate + p.kappa_m * (p.m_bar - state.mortgage_rate) * p.dt \
                + p.sigma_m * np.sqrt(p.dt) * eps_m
        new_m = np.clip(new_m, 0.01, 0.15)  # Bound between 1% and 15%
        
        # Rent (proportional to home price)
        new_r = p.rent_ratio * new_h
        
        # --- Process action ---
        w = state.savings
        reward = 0.0
        new_owns = state.owns
        new_mort_bal = state.mortgage_balance
        new_purch = state.purchase_price
        
        if action == self.RENT_STAY:
            # Pay rent, earn income, get return on savings
            housing_cost = state.rent
            w = (w - housing_cost + p.income) * (1 + p.risk_free * p.dt)
            reward = -housing_cost
            
        elif action == self.BUY:
            # Pay down payment + closing costs
            down = p.down_payment_frac * state.home_price
            closing = p.closing_cost_frac * state.home_price
            w = w - down - closing
            new_owns = True
            new_mort_bal = state.home_price - down
            new_purch = state.home_price
            reward = -closing  # Transaction cost
            
        elif action == self.OWN_STAY:
            # Pay mortgage (simplified: interest only)
            payment = state.mortgage_balance * state.mortgage_rate * p.dt
            w = (w - payment + p.income) * (1 + p.risk_free * p.dt)
            reward = -payment + p.ownership_utility  # Include ownership utility
            
        elif action == self.SELL:
            # Sell home, realize equity, pay selling costs
            equity = state.home_price - state.mortgage_balance
            sell_cost = p.selling_cost_frac * state.home_price
            w = w + equity - sell_cost
            new_owns = False
            new_mort_bal = 0.0
            new_purch = 0.0
            reward = -sell_cost  # Transaction cost
        
        next_state = BuyRentState(
            savings=w,
            home_price=new_h,
            rent=new_r,
            mortgage_rate=new_m,
            owns=new_owns,
            mortgage_balance=new_mort_bal,
            purchase_price=new_purch
        )
        
        return next_state, reward
    
    def _get_info(self) -> Dict:
        """Return auxiliary information."""
        return {
            'action_mask': self._get_action_mask(self.state),
            'net_worth': self._calculate_net_worth(self.state),
            'steps': self.steps
        }
    
    def _calculate_net_worth(self, state: BuyRentState) -> float:
        """Calculate total net worth (savings + home equity)."""
        if state.owns:
            equity = state.home_price - state.mortgage_balance
            return state.savings + equity
        else:
            return state.savings
    
    def get_trajectory(self) -> pd.DataFrame:
        """Return trajectory data from current episode."""
        return pd.DataFrame(self.episode_data)
    
    def render(self, mode='human'):
        """Render current state."""
        if mode == 'human':
            status = "OWNING" if self.state.owns else "RENTING"
            print(f"\nStep {self.steps}: {status}")
            print(f"  Savings: ${self.state.savings:.2f}k")
            print(f"  Home Price: ${self.state.home_price:.2f}k")
            print(f"  Mortgage Rate: {self.state.mortgage_rate*100:.2f}%")
            if self.state.owns:
                print(f"  Mortgage Balance: ${self.state.mortgage_balance:.2f}k")
                equity = self.state.home_price - self.state.mortgage_balance
                print(f"  Home Equity: ${equity:.2f}k")
            net_worth = self._calculate_net_worth(self.state)
            print(f"  Net Worth: ${net_worth:.2f}k")


def generate_sampling_traces(
    n_episodes: int = 100,
    params: Optional[BuyRentParams] = None,
    policy: str = 'random',
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate sampling traces from the environment.
    
    Args:
        n_episodes: Number of episodes to simulate
        params: Environment parameters
        policy: Policy to use ('random', 'always_rent', 'greedy_buy')
        save_path: If provided, save traces to CSV
    
    Returns:
        DataFrame with all trajectories
    """
    env = BuyRentEnv(params)
    all_trajectories = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        
        while not done:
            # Select action based on policy
            action_mask = info['action_mask']
            valid_actions = np.where(action_mask == 1)[0]
            
            if policy == 'random':
                action = np.random.choice(valid_actions)
            elif policy == 'always_rent':
                action = env.RENT_STAY if env.RENT_STAY in valid_actions else valid_actions[0]
            elif policy == 'greedy_buy':
                # Buy if can afford, otherwise rent
                if env.BUY in valid_actions:
                    action = env.BUY
                elif env.OWN_STAY in valid_actions:
                    action = env.OWN_STAY
                else:
                    action = env.RENT_STAY
            else:
                action = np.random.choice(valid_actions)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # Get trajectory
        traj = env.get_trajectory()
        traj['episode'] = ep
        all_trajectories.append(traj)
    
    # Combine all trajectories
    df = pd.concat(all_trajectories, ignore_index=True)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved {n_episodes} episodes to {save_path}")
    
    return df


if __name__ == "__main__":
    # Demo: Generate sampling traces
    print("Generating sampling traces...")
    
    # Try to load calibrated parameters
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
        print("✓ Using calibrated parameters from real data")
    except:
        params = BuyRentParams()
        print("⚠️  Using default parameters")
    
    # Generate traces with different policies
    print("\n1. Random policy:")
    random_traces = generate_sampling_traces(
        n_episodes=10,
        params=params,
        policy='random',
        save_path='/mnt/user-data/outputs/traces_random.csv'
    )
    
    print("\n2. Always rent policy:")
    rent_traces = generate_sampling_traces(
        n_episodes=10,
        params=params,
        policy='always_rent',
        save_path='/mnt/user-data/outputs/traces_rent.csv'
    )
    
    print("\n3. Greedy buy policy:")
    buy_traces = generate_sampling_traces(
        n_episodes=10,
        params=params,
        policy='greedy_buy',
        save_path='/mnt/user-data/outputs/traces_buy.csv'
    )
    
    print("\n✓ Sampling traces generated successfully!")
    print(f"  Random policy: {len(random_traces)} transitions")
    print(f"  Always rent: {len(rent_traces)} transitions")
    print(f"  Greedy buy: {len(buy_traces)} transitions")
