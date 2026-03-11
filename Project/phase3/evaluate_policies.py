"""
Evaluation Script for Buy vs. Rent DQN

Compares the learned policy against baselines:
1. Random policy
2. Always rent
3. Greedy buy (buy as soon as possible, hold forever)
4. DP optimal policy (if available from Phase 2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from buy_rent_environment import BuyRentEnv, BuyRentParams
from modified_dqn import ActionMaskedDQNAgent


class PolicyEvaluator:
    """Evaluate and compare different policies."""
    
    def __init__(self, env: BuyRentEnv, n_episodes: int = 100):
        self.env = env
        self.n_episodes = n_episodes
        
    def evaluate_policy(
        self,
        policy_fn,
        policy_name: str,
        verbose: bool = False
    ) -> Dict:
        """
        Evaluate a policy over multiple episodes.
        
        Args:
            policy_fn: Function(state, action_mask, env) -> action
            policy_name: Name for logging
            verbose: Print progress
        
        Returns:
            Dictionary with metrics
        """
        episode_rewards = []
        episode_lengths = []
        final_net_worths = []
        num_buys = []
        num_sells = []
        
        for ep in range(self.n_episodes):
            state, info = self.env.reset(seed=ep)
            done = False
            episode_reward = 0
            steps = 0
            buys = 0
            sells = 0
            
            while not done:
                action_mask = info['action_mask']
                action = policy_fn(state, action_mask, self.env)
                
                if action == self.env.BUY:
                    buys += 1
                elif action == self.env.SELL:
                    sells += 1
                
                state, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                steps += 1
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            final_net_worths.append(info['net_worth'])
            num_buys.append(buys)
            num_sells.append(sells)
            
            if verbose and (ep + 1) % 20 == 0:
                print(f"  {policy_name}: Episode {ep+1}/{self.n_episodes} complete")
        
        return {
            'name': policy_name,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_net_worth': np.mean(final_net_worths),
            'std_net_worth': np.std(final_net_worths),
            'mean_length': np.mean(episode_lengths),
            'mean_buys': np.mean(num_buys),
            'mean_sells': np.mean(num_sells),
            'all_rewards': episode_rewards,
            'all_net_worths': final_net_worths
        }


def random_policy(state, action_mask, env):
    """Random policy - choose uniformly from valid actions."""
    valid_actions = np.where(action_mask == 1)[0]
    return np.random.choice(valid_actions)


def always_rent_policy(state, action_mask, env):
    """Always rent policy - never buy."""
    if env.RENT_STAY in np.where(action_mask == 1)[0]:
        return env.RENT_STAY
    elif env.OWN_STAY in np.where(action_mask == 1)[0]:
        return env.OWN_STAY
    else:
        return np.where(action_mask == 1)[0][0]


def greedy_buy_policy(state, action_mask, env):
    """Greedy buy policy - buy ASAP, hold forever."""
    valid_actions = np.where(action_mask == 1)[0]
    
    # Priority: BUY > STAY > RENT
    if env.BUY in valid_actions:
        return env.BUY
    elif env.OWN_STAY in valid_actions:
        return env.OWN_STAY
    else:
        return env.RENT_STAY


def make_dqn_policy(agent: ActionMaskedDQNAgent):
    """Create policy function from trained DQN agent."""
    def policy(state, action_mask, env):
        return agent.select_action(state, action_mask, training=False)
    return policy


def plot_comparison(results: List[Dict], save_path: str = None):
    """Plot comparison of different policies."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = [r['name'] for r in results]
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    # 1. Mean Reward Comparison
    ax = axes[0, 0]
    mean_rewards = [r['mean_reward'] for r in results]
    std_rewards = [r['std_reward'] for r in results]
    x = np.arange(len(names))
    ax.bar(x, mean_rewards, color=colors, alpha=0.7)
    ax.errorbar(x, mean_rewards, yerr=std_rewards, fmt='none', 
                ecolor='black', capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Mean Cumulative Reward')
    ax.set_title('Policy Comparison: Total Reward')
    ax.grid(True, alpha=0.3)
    
    # 2. Final Net Worth Comparison
    ax = axes[0, 1]
    mean_nw = [r['mean_net_worth'] for r in results]
    std_nw = [r['std_net_worth'] for r in results]
    ax.bar(x, mean_nw, color=colors, alpha=0.7)
    ax.errorbar(x, mean_nw, yerr=std_nw, fmt='none', 
                ecolor='black', capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Final Net Worth ($k)')
    ax.set_title('Policy Comparison: Final Wealth')
    ax.grid(True, alpha=0.3)
    
    # 3. Reward Distribution
    ax = axes[1, 0]
    for i, r in enumerate(results):
        ax.hist(r['all_rewards'], bins=30, alpha=0.5, 
                label=r['name'], color=colors[i])
    ax.set_xlabel('Cumulative Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution Across Episodes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Actions Taken
    ax = axes[1, 1]
    buys = [r['mean_buys'] for r in results]
    sells = [r['mean_sells'] for r in results]
    width = 0.35
    ax.bar(x - width/2, buys, width, label='Buys', alpha=0.7)
    ax.bar(x + width/2, sells, width, label='Sells', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Average Count')
    ax.set_title('Transaction Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {save_path}")
    
    return fig


def print_comparison_table(results: List[Dict]):
    """Print comparison table in console."""
    print("\n" + "="*80)
    print("POLICY COMPARISON RESULTS")
    print("="*80)
    print(f"{'Policy':<20} {'Reward':<15} {'Net Worth':<15} {'Buys':<10} {'Sells':<10}")
    print("-"*80)
    
    for r in results:
        print(f"{r['name']:<20} "
              f"{r['mean_reward']:>7.2f} ± {r['std_reward']:<5.2f} "
              f"${r['mean_net_worth']:>6.1f}k ± {r['std_net_worth']:<4.1f} "
              f"{r['mean_buys']:>5.2f}    "
              f"{r['mean_sells']:>5.2f}")
    
    print("="*80)
    
    # Find best policy
    best_idx = np.argmax([r['mean_reward'] for r in results])
    print(f"\n🏆 Best Policy: {results[best_idx]['name']}")
    print(f"   Mean Reward: {results[best_idx]['mean_reward']:.2f}")
    print(f"   Final Net Worth: ${results[best_idx]['mean_net_worth']:.1f}k")


def analyze_learned_policy(
    agent: ActionMaskedDQNAgent,
    env: BuyRentEnv,
    n_scenarios: int = 5
):
    """
    Analyze what the learned policy does in different scenarios.
    """
    print("\n" + "="*80)
    print("LEARNED POLICY ANALYSIS")
    print("="*80)
    
    scenarios = [
        {
            'name': 'Low savings, cheap market',
            'savings': 10.0,
            'home_price': 60.0,
            'mortgage_rate': 0.04
        },
        {
            'name': 'Medium savings, normal market',
            'savings': 50.0,
            'home_price': 100.0,
            'mortgage_rate': 0.06
        },
        {
            'name': 'High savings, expensive market',
            'savings': 80.0,
            'home_price': 150.0,
            'mortgage_rate': 0.08
        },
        {
            'name': 'High savings, low rates',
            'savings': 70.0,
            'home_price': 100.0,
            'mortgage_rate': 0.03
        },
        {
            'name': 'Low savings, high rates',
            'savings': 20.0,
            'home_price': 100.0,
            'mortgage_rate': 0.10
        }
    ]
    
    action_names = ['RENT', 'BUY', 'STAY', 'SELL']
    
    for scenario in scenarios:
        print(f"\n📋 {scenario['name']}")
        print(f"   Savings: ${scenario['savings']:.1f}k")
        print(f"   Home Price: ${scenario['home_price']:.1f}k")
        print(f"   Mortgage Rate: {scenario['mortgage_rate']*100:.1f}%")
        
        # Create state
        state = np.array([
            scenario['savings'],
            scenario['home_price'],
            scenario['mortgage_rate'],
            0.0,  # Not owning
            0.0   # No mortgage
        ], dtype=np.float32)
        
        # Get action mask for renting state
        action_mask = np.zeros(4)
        action_mask[env.RENT_STAY] = 1
        cost = (env.p.down_payment_frac + env.p.closing_cost_frac) * scenario['home_price']
        if scenario['savings'] >= cost:
            action_mask[env.BUY] = 1
        
        # Get action
        action = agent.select_action(state, action_mask, training=False)
        
        print(f"   ➜ Decision: {action_names[action]}")
        
        if action == env.BUY:
            print(f"   💡 Agent chooses to BUY")
            print(f"      Down payment needed: ${cost:.1f}k")
            print(f"      Remaining savings: ${scenario['savings'] - cost:.1f}k")


if __name__ == "__main__":
    print("="*80)
    print("EVALUATING BUY VS. RENT POLICIES")
    print("="*80)
    
    # Load calibrated parameters
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
    
    # Create environment
    env = BuyRentEnv(params)
    evaluator = PolicyEvaluator(env, n_episodes=50)
    
    # Evaluate baseline policies
    print("Evaluating baseline policies...")
    
    results = []
    
    print("\n1. Random Policy")
    results.append(evaluator.evaluate_policy(random_policy, "Random", verbose=True))
    
    print("\n2. Always Rent")
    results.append(evaluator.evaluate_policy(always_rent_policy, "Always Rent", verbose=True))
    
    print("\n3. Greedy Buy")
    results.append(evaluator.evaluate_policy(greedy_buy_policy, "Greedy Buy", verbose=True))
    
    # Try to load and evaluate DQN policy
    try:
        print("\n4. Trained DQN")
        agent = ActionMaskedDQNAgent()
        agent.load('/mnt/user-data/outputs/dqn_buy_rent.pth')
        dqn_policy = make_dqn_policy(agent)
        results.append(evaluator.evaluate_policy(dqn_policy, "Learned DQN", verbose=True))
        
        # Analyze learned policy
        analyze_learned_policy(agent, env)
        
    except FileNotFoundError:
        print("\n⚠️  No trained model found. Run modified_dqn.py first.")
    
    # Display results
    print_comparison_table(results)
    
    # Plot comparison
    fig = plot_comparison(results, save_path='/mnt/user-data/outputs/policy_comparison.png')
    
    print("\n✓ Evaluation complete!")
