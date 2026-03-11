# Phase 3 Guide: RL for Buy vs. Rent

## 📋 Overview

This phase implements the **Reinforcement Learning** solution to the Buy vs. Rent problem using a **modified Deep Q-Network (DQN)**.

---

## 🎯 What You're Solving

**The "Version 2" problem from Phase 1:**
- State: `(savings, home_price, mortgage_rate, owns, mortgage_balance)` - continuous except `owns`
- Actions: `{RENT, BUY, STAY, SELL}` - discrete
- Dynamics: GBM for prices, Vasicek for rates (continuous, stochastic)
- Horizon: 30 years (360 months)

This is the **realistic version** - not too simple, not commercially complex.

---

## 🔧 What We Modified in Standard DQN

### 1. **Action Masking** (Most Important Modification)

**Problem:** Not all actions are valid in every state.
- Can't `BUY` without sufficient savings
- Can only `STAY`/`SELL` when owning
- Can only `RENT`/`BUY` when renting

**Standard DQN:** Learns Q(s,a) for all actions, might try invalid actions

**Our Modification:**
```python
# Set Q-values of invalid actions to -inf
q_values_masked = np.full(4, -np.inf)
q_values_masked[valid_actions] = q_values[valid_actions]
action = np.argmax(q_values_masked)
```

**Why this matters:** Without masking, agent wastes time exploring impossible actions and gets confused by large negative rewards.

---

### 2. **Dueling Architecture**

**Problem:** In this problem, most of the time the action doesn't matter much (e.g., when prices are stable, `STAY` is optimal).

**Standard DQN:** Learns Q(s,a) directly - mixes value of being in state s with value of taking action a

**Our Modification:** Separate into:
```
Q(s,a) = V(s) + [A(s,a) - mean(A(s,·))]
```
- **V(s)** = "How good is this state?" (independent of action)
- **A(s,a)** = "How much better is action a than average?"

**Why this matters:** When holding is often optimal, the network learns "this is a good state to be in" separately from "selling would be bad here."

---

### 3. **State Normalization**

**Problem:** State dimensions have very different scales:
- Savings: $0k - $200k
- Home price: $50k - $300k  
- Mortgage rate: 0.01 - 0.15 (1%-15%)
- Owns: 0 or 1
- Mortgage balance: $0k - $200k

**Our Modification:**
```python
state_norm = (state - mean) / std
```

**Why this matters:** Neural networks learn better when all inputs are roughly the same scale (around [-1, 1]).

---

### 4. **Reward Shaping** (Implicit)

**Problem:** Transaction costs are large one-time penalties, but we care about long-term net worth.

**Our Approach:**
- Negative rewards for housing costs (rent/mortgage)
- Large negative rewards for transaction costs (buying/selling)
- Small positive reward for ownership utility
- Let the discount factor (γ=0.95) handle long-term value

**Why this matters:** Agent learns to avoid frequent switching due to transaction costs while still recognizing when switching is worth it.

---

### 5. **Experience Replay with Action Masks**

**Problem:** Need to store which actions were valid in each state.

**Our Modification:**
```python
Transition = (state, action, next_state, reward, done, action_mask)
```

**Why this matters:** During batch updates, we apply the next state's action mask to avoid learning from invalid action Q-values.

---

## 🏗️ Architecture Diagram

```
Input State [5D]
  ↓
Feature Extraction (256 neurons)
  ↓
Split into two streams:
  ├─→ Value Stream V(s) → [1]
  └─→ Advantage Stream A(s,a) → [4]
  ↓
Combine: Q(s,a) = V(s) + [A(s,a) - mean(A)]
  ↓
Apply Action Mask
  ↓
Select Action (ε-greedy)
```

---

## 📚 RL Libraries to Use

### Recommended Libraries (Pick One):

1. **Stable-Baselines3** (Easiest, Best for This Project) ⭐
   ```bash
   pip install stable-baselines3 gymnasium
   ```
   - Industry-standard RL library
   - Has DQN with customization options
   - Easy to add action masking
   - Good documentation
   - **Use this unless you want to code from scratch**

2. **RLlib (Ray)** (If You Want Scalability)
   ```bash
   pip install ray[rllib]
   ```
   - Scales to multi-GPU/distributed
   - Has action masking built-in
   - Steeper learning curve

3. **PyTorch + Our Code** (What I Provided) ⭐
   ```bash
   pip install torch gymnasium
   ```
   - Full control over modifications
   - Educational - you understand every line
   - Good for presentations (can explain exactly what you changed)

### NOT Recommended:
- ❌ The course's `rl` library (educational only, not performant)
- ❌ TensorFlow Agents (more complex than necessary)
- ❌ Pure NumPy (too slow for neural networks)

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install torch gymnasium pandas matplotlib numpy
```

### Step 2: Generate Sampling Traces

```bash
python buy_rent_environment.py
```

This creates:
- `traces_random.csv` - Random policy baseline
- `traces_rent.csv` - Always rent baseline
- `traces_buy.csv` - Greedy buy baseline

These are your **sampling traces** from the MDP!

### Step 3: Train the Modified DQN

```bash
python modified_dqn.py
```

This will:
- Train for 100 episodes (increase to 1000+ for better performance)
- Save model to `dqn_buy_rent.pth`
- Generate training curves `dqn_training.png`

### Step 4: Evaluate the Learned Policy

```python
from modified_dqn import ActionMaskedDQNAgent
from buy_rent_environment import BuyRentEnv

# Load trained agent
agent = ActionMaskedDQNAgent()
agent.load('dqn_buy_rent.pth')

# Test on environment
env = BuyRentEnv()
state, info = env.reset()
done = False
total_reward = 0

while not done:
    action = agent.select_action(state, info['action_mask'], training=False)
    state, reward, done, _, info = env.step(action)
    total_reward += reward
    env.render()

print(f"Total reward: {total_reward}")
```

---

## 📊 What to Include in Your Report

### 1. **Problem Setup**
- State space: 5D continuous (with 1 discrete dimension)
- Action space: 4 discrete actions with state-dependent availability
- Dynamics: GBM + Vasicek model calibrated from real data
- Why standard DQN won't work directly (action masking needed)

### 2. **Modifications Made**
Explain each of the 5 modifications above:
- **Action masking** - why it's necessary
- **Dueling architecture** - why it helps
- **State normalization** - prevents learning issues
- **Reward shaping** - balances short/long term
- **Modified replay buffer** - stores action masks

### 3. **Hyperparameters**
```
Learning rate: 0.0003
Discount factor γ: 0.95
Epsilon decay: 10,000 steps
Replay buffer: 50,000 transitions
Batch size: 64
Target network update: every 1,000 steps
Network: [5] → [256, 256] → split → [1 value, 4 advantages]
```

### 4. **Results**

Show:
- **Training curves** (reward vs episode, loss vs episode)
- **Comparison to baselines** (random, always rent, greedy buy)
- **Sample trajectories** from learned policy
- **Buy threshold analysis** - when does the agent choose to buy?
- **Sensitivity to parameters** - what if μ_h changes? what if rates spike?

### 5. **Analysis**

- Does the policy make economic sense?
  - Buys when savings are high and rates are low?
  - Holds when already owning (transaction cost inertia)?
  - Sells when prices spike above long-run mean?

- How does it compare to Phase 2 (DP solution)?
  - Similar buy thresholds?
  - Handles continuous states better?

---

## 🎓 Why These Modifications Matter

| Modification | Without It | With It |
|--------------|-----------|---------|
| **Action Masking** | Agent wastes 50%+ of training trying invalid actions | Only explores feasible action space |
| **Dueling Architecture** | Slow learning in stable markets | Learns V(s) separately from action values |
| **State Normalization** | Gradient updates dominated by large values (prices) | All features contribute equally to learning |
| **Reward Shaping** | Myopic behavior (minimize immediate costs) | Balances short-term costs vs long-term value |
| **Modified Replay** | Invalid actions in batch updates | Correctly masks impossible actions |

---

## 🔬 Extensions for Bonus Points

1. **Prioritized Experience Replay** - sample important transitions more often
2. **Double DQN** - reduce overestimation bias
3. **Multi-step returns** - use n-step TD targets instead of 1-step
4. **Noisy Networks** - replace ε-greedy with parameter noise
5. **Compare to Policy Gradient** - implement PPO and compare

---

## 🎬 For Your Presentation (March 13)

### Show:
1. **The environment** - run a few steps with `env.render()`
2. **Action masking in action** - show how valid actions change
3. **Training curves** - demonstrate learning
4. **Learned behavior** - show agent buying when conditions are right
5. **Comparison** - learned policy vs baselines (random, always rent)

### Emphasize:
- This is **not** just vanilla DQN from a textbook
- You **modified** it specifically for your problem
- The modifications are **necessary** (show what breaks without them)
- Results are **economically interpretable** (not just black box)

---

## ⚙️ Troubleshooting

**Q: Agent never buys, just rents forever?**
- Check if ownership utility is too small
- Try increasing initial savings so buying is feasible
- Reduce transaction costs temporarily to debug

**Q: Training is unstable / diverges?**
- Reduce learning rate
- Increase batch size
- Add gradient clipping (already included)
- Check state normalization

**Q: Action masking not working?**
- Print valid actions each step
- Verify mask is binary [0, 1]
- Check that mask is stored in replay buffer

**Q: Too slow to train?**
- Reduce network size (256 → 128)
- Use GPU (`device='cuda'`)
- Decrease n_episodes or max_steps
- Decrease buffer size

---

## 📝 Files You'll Submit

1. `buy_rent_environment.py` - Gym environment
2. `modified_dqn.py` - Modified DQN agent
3. `train.py` - Training script (can combine with modified_dqn.py)
4. `evaluate.py` - Evaluation script
5. `report.pdf` - Write-up explaining everything
6. `traces_*.csv` - Sampling traces (proof your environment works)
7. `dqn_buy_rent.pth` - Trained model weights
8. `results/` - Plots, analysis, comparisons

---

**Good luck! You got this! 🚀**
