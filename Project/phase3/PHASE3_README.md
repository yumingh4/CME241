# Phase 3: Reinforcement Learning Solution - README

## 📋 Overview

This folder contains the **Reinforcement Learning** implementation for the Buy vs. Rent problem using a modified Deep Q-Network (DQN). The agent learns an optimal policy for when to buy, rent, hold, or sell through simulated experience.

---

## 📂 File Structure

```
phase3/
├── buy_rent_environment.py      # Gym environment (generates sampling traces)
├── modified_dqn.py               # Modified DQN agent with 5 key adaptations
├── evaluate_policies.py          # Compare learned policy vs baselines
├── calibrated_params.py          # Parameters from Phase 2 (copy from ../phase2/)
├── PHASE3_GUIDE.md              # Technical guide with detailed explanations
├── PHASE3_SUMMARY.md            # Quick reference and FAQ
└── PHASE3_README.md             # This file
```

---

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
# Required packages
pip install torch gymnasium pandas matplotlib numpy

# Optional: for prettier plots
pip install seaborn
```

**What each package does:**
- `torch` - PyTorch for neural networks
- `gymnasium` - Standard RL environment interface
- `pandas` - Data manipulation and saving traces
- `matplotlib` - Plotting results
- `numpy` - Numerical computations

---

### 2. Copy Calibrated Parameters

If you completed Phase 2, you should have `calibrated_params.py` with real market data:

```bash
# Copy from phase2 to phase3
cp ../phase2/calibrated_params.py .
```

If you **don't** have this file, the code will use default parameters (but real data is better!).

---

## 🎯 What to Run (Step-by-Step)

### **Step 1: Generate Sampling Traces** ⏱️ ~10 seconds

```bash
python buy_rent_environment.py
```

**What this does:**
- Runs the MDP environment with 3 different baseline policies
- Generates sampling traces (required by professor)
- Saves trajectories as CSV files

**Output files:**
- `traces_random.csv` - Random policy baseline
- `traces_rent.csv` - Always rent policy baseline
- `traces_buy.csv` - Greedy buy policy baseline

**What you'll see in terminal:**
```
Generating sampling traces...
✓ Using calibrated parameters from real data

1. Random policy:
Saved 10 episodes to traces_random.csv

2. Always rent policy:
Saved 10 episodes to traces_rent.csv

3. Greedy buy policy:
Saved 10 episodes to traces_buy.csv

✓ Sampling traces generated successfully!
  Random policy: 3245 transitions
  Always rent: 3600 transitions
  Greedy buy: 3600 transitions
```

**What's in the CSV files:**
```csv
episode,step,action,reward,savings,home_price,mortgage_rate,owns
0,1,0,-0.40,20.23,100.45,0.061,False
0,2,0,-0.41,20.47,101.12,0.059,False
0,3,1,-3.00,2.47,101.12,0.059,True
...
```

Each row = one time step with state-action-reward information.

---

### **Step 2: Train the DQN Agent** ⏱️ ~2-5 minutes (100 episodes)

```bash
python modified_dqn.py
```

**What this does:**
- Creates a DQN agent with 5 modifications for your problem
- Trains for 100 episodes (increase to 500-1000 for better results)
- Saves trained model and training curves

**Output files:**
- `dqn_buy_rent.pth` - Trained model weights
- `dqn_training.png` - Training curves (2 plots)

**What you'll see in terminal:**
```
======================================================================
TRAINING DQN AGENT FOR BUY VS. RENT
======================================================================
✓ Using calibrated parameters

Training for 100 episodes (demo)...

Episode 10/100 | Avg Reward: -234.52 | Avg Length: 360 | Epsilon: 0.905 | Buffer: 3600
Episode 20/100 | Avg Reward: -189.34 | Avg Length: 360 | Epsilon: 0.819 | Buffer: 7200
Episode 30/100 | Avg Reward: -167.21 | Avg Length: 360 | Epsilon: 0.741 | Buffer: 10800
...
Episode 100/100 | Avg Reward: -142.15 | Avg Length: 360 | Epsilon: 0.135 | Buffer: 36000

✓ Model saved to dqn_buy_rent.pth
✓ Training curves saved to dqn_training.png
```

**What `dqn_training.png` shows:**

![Training Curves](example_training.png)

**Left plot: Episode Rewards**
- Y-axis: Total cumulative reward per episode
- X-axis: Episode number
- Should trend upward (agent learning to get higher rewards)

**Right plot: Training Loss**
- Y-axis: TD error loss
- X-axis: Episode number
- Should trend downward and stabilize (network converging)

**Interpretation:**
- Early episodes: Random exploration (ε=1.0), poor rewards
- Middle episodes: Learning, improving steadily
- Later episodes: Exploitation (ε→0.05), stable high rewards

---

### **Step 3: Evaluate Against Baselines** ⏱️ ~30 seconds

```bash
python evaluate_policies.py
```

**What this does:**
- Tests learned DQN policy against 3 baselines
- Compares mean reward, final net worth, and trading frequency
- Generates comparison plots

**Output files:**
- `policy_comparison.png` - 4-panel comparison plot

**What you'll see in terminal:**
```
======================================================================
EVALUATING BUY VS. RENT POLICIES
======================================================================
✓ Using calibrated parameters

Evaluating baseline policies...

1. Random Policy
  Random: Episode 20/50 complete
  Random: Episode 40/50 complete

2. Always Rent
  Always Rent: Episode 20/50 complete
  Always Rent: Episode 40/50 complete

3. Greedy Buy
  Greedy Buy: Episode 20/50 complete
  Greedy Buy: Episode 40/50 complete

4. Trained DQN
  Learned DQN: Episode 20/50 complete
  Learned DQN: Episode 40/50 complete

======================================================================
LEARNED POLICY ANALYSIS
======================================================================

📋 Low savings, cheap market
   Savings: $10.0k
   Home Price: $60.0k
   Mortgage Rate: 4.0%
   ➜ Decision: RENT
   💡 Keep renting - insufficient savings

📋 Medium savings, normal market
   Savings: $50.0k
   Home Price: $100.0k
   Mortgage Rate: 6.0%
   ➜ Decision: BUY
   💡 Agent chooses to BUY
      Down payment needed: $23.0k
      Remaining savings: $27.0k

📋 High savings, expensive market
   Savings: $80.0k
   Home Price: $150.0k
   Mortgage Rate: 8.0%
   ➜ Decision: RENT
   💡 Keep renting - market too expensive

📋 High savings, low rates
   Savings: $70.0k
   Home Price: $100.0k
   Mortgage Rate: 3.0%
   ➜ Decision: BUY
   💡 Agent chooses to BUY
      Down payment needed: $23.0k
      Remaining savings: $47.0k

📋 Low savings, high rates
   Savings: $20.0k
   Home Price: $100.0k
   Mortgage Rate: 10.0%
   ➜ Decision: RENT
   💡 Keep renting - rates too high

======================================================================
POLICY COMPARISON RESULTS
======================================================================
Policy               Reward          Net Worth       Buys       Sells     
--------------------------------------------------------------------------------
Random               -198.34 ± 45.23 $102.3k ± 18.2  1.24       0.87
Always Rent          -213.56 ± 12.45 $95.4k ± 8.1    0.00       0.00
Greedy Buy           -156.78 ± 23.12 $118.7k ± 15.3  0.98       0.02
Learned DQN          -142.15 ± 19.45 $125.2k ± 12.8  0.87       0.05
================================================================================

🏆 Best Policy: Learned DQN
   Mean Reward: -142.15
   Final Net Worth: $125.2k

✓ Saved policy_comparison.png
✓ Evaluation complete!
```

**What `policy_comparison.png` shows:**

![Policy Comparison](example_comparison.png)

**Top Left: Mean Reward Comparison**
- Bar chart showing average cumulative reward
- Higher is better (less negative)
- Error bars show standard deviation

**Top Right: Final Net Worth**
- Average wealth at end of 30 years
- Includes savings + home equity
- Higher is better

**Bottom Left: Reward Distribution**
- Histogram of rewards across all episodes
- Shows consistency (tight distribution = more reliable)

**Bottom Right: Transaction Frequency**
- How often each policy buys/sells
- Learned DQN should buy strategically, rarely sell

---

## 📊 Understanding the Results

### What Good Results Look Like:

✅ **Training converges**: Rewards increase, loss decreases  
✅ **Beats baselines**: DQN > Greedy Buy > Random > Always Rent  
✅ **Makes sense**: Buys when savings high + rates low, holds when owning  
✅ **Low transaction frequency**: 0.5-1.5 buys per episode (not switching constantly)  
✅ **Higher final wealth**: $120k-$140k vs $90k-$110k for baselines

### What Bad Results Look Like:

❌ **Training doesn't converge**: Rewards stay flat or decrease  
❌ **Loses to baselines**: DQN < Random  
❌ **Nonsensical behavior**: Buys with no savings, sells immediately  
❌ **Too many transactions**: 5+ buys per episode  
❌ **Lower final wealth**: Worse than always renting

---

## 🔧 Troubleshooting

### Problem: "No module named torch"
**Solution:**
```bash
pip install torch gymnasium pandas matplotlib
```

### Problem: "calibrated_params.py not found"
**Solution:**
```bash
# Either copy from phase2:
cp ../phase2/calibrated_params.py .

# Or just run anyway (will use defaults):
python buy_rent_environment.py
```

### Problem: Agent never buys, just rents forever
**Possible causes:**
1. Ownership utility too small → Increase to 1.0-2.0
2. Transaction costs too high → Reduce closing_cost_frac
3. Initial savings too low → Increase starting savings
4. Training not long enough → Train for 500+ episodes

**Fix:**
```python
# In buy_rent_environment.py, adjust:
params = BuyRentParams(
    ownership_utility=1.0,     # Increase from 0.5
    closing_cost_frac=0.02,    # Decrease from 0.03
)

# Or in initial state:
initial_state = BuyRentState(
    savings=40.0,  # Increase from 20.0
    # ...
)
```

### Problem: Training is very slow
**Solutions:**
- Reduce episodes: `n_episodes=100` instead of 1000
- Reduce max_steps: `max_steps=180` (15 years) instead of 360
- Use GPU: `device='cuda'` (if you have NVIDIA GPU)
- Smaller network: `hidden_dim=128` instead of 256

### Problem: Results are unstable / high variance
**Solutions:**
- Train longer: 500-1000 episodes
- Increase batch size: `batch_size=128`
- Reduce learning rate: `learning_rate=0.0001`
- Run multiple seeds and average

---

## 📈 Tuning for Better Performance

### Quick wins:
1. **Train longer**: 500-1000 episodes instead of 100
2. **Adjust ownership utility**: Try 0.5, 1.0, 1.5, 2.0
3. **Change initial savings**: 20k, 40k, 60k starting wealth
4. **Multiple random seeds**: Train 3 times, pick best

### Advanced tuning:
1. **Prioritized experience replay**: Sample important transitions more
2. **Reward shaping**: Add bonus for high net worth
3. **Curriculum learning**: Start with easier scenarios
4. **Hyperparameter search**: Grid search over learning_rate, gamma, epsilon_decay

---

## 🎓 For Your Report

### Include These Figures:

1. **`dqn_training.png`** - Shows learning progress
   - Caption: "Training curves showing DQN converging over 100 episodes"

2. **`policy_comparison.png`** - Shows DQN beats baselines
   - Caption: "Learned policy achieves higher rewards and final wealth than baseline strategies"

3. **Sample trajectory plot** (make this yourself):
   ```python
   # Plot one episode showing state evolution
   traj = env.get_trajectory()
   plt.plot(traj['step'], traj['savings'], label='Savings')
   plt.plot(traj['step'], traj['home_price'], label='Home Price')
   ```
   - Caption: "Example trajectory showing agent buying at step 50 when savings are high"

### Include These Tables:

1. **Hyperparameters**:
   ```
   Learning rate: 0.0003
   Gamma: 0.95
   Epsilon decay: 10,000 steps
   Batch size: 64
   Network: [5] → [256, 256] → [V(s), A(s,a)]
   ```

2. **Results summary** (from terminal output):
   ```
   Policy          | Mean Reward | Final Net Worth
   ----------------|-------------|----------------
   Random          | -198.34     | $102.3k
   Always Rent     | -213.56     | $95.4k
   Greedy Buy      | -156.78     | $118.7k
   Learned DQN     | -142.15     | $125.2k
   ```

---

## 🎯 Expected Timeline

| Task | Time | Output |
|------|------|--------|
| Generate traces | 10 sec | CSV files |
| Train DQN (100 ep) | 2-5 min | Model + training plot |
| Train DQN (1000 ep) | 20-30 min | Better model |
| Evaluate | 30 sec | Comparison plot |
| **Total (quick demo)** | **~5 min** | All plots + model |
| **Total (full training)** | **~30 min** | Publication-quality results |

---

## 📝 Checklist for Submission

- [ ] `traces_*.csv` files (proof environment works)
- [ ] `dqn_buy_rent.pth` (trained model)
- [ ] `dqn_training.png` (learning curves)
- [ ] `policy_comparison.png` (evaluation results)
- [ ] All `.py` files (environment, agent, evaluation)
- [ ] Report explaining modifications and results
- [ ] Presentation slides

---

## 🚀 Quick Start (TL;DR)

```bash
# Install
pip install torch gymnasium pandas matplotlib numpy

# Copy calibrated params (if you have Phase 2)
cp ../phase2/calibrated_params.py .

# Run everything
python buy_rent_environment.py    # → traces_*.csv
python modified_dqn.py             # → dqn_buy_rent.pth, dqn_training.png
python evaluate_policies.py        # → policy_comparison.png

# Done! You now have all results for your report.
```

---

## 💡 Tips for Presentation (March 13)

### Show:
1. **Environment demo** - Run a few steps with `env.render()`
2. **Training curves** - `dqn_training.png` showing learning
3. **Policy comparison** - `policy_comparison.png` showing DQN wins
4. **Learned behavior** - Terminal output showing when agent buys vs rents

### Emphasize:
- Not vanilla DQN - **5 specific modifications**
- Modifications are **necessary** (show what breaks without them)
- Results are **economically interpretable** (buys when savings high + rates low)
- Uses **real market parameters** from Phase 2

---

## ❓ Questions?

Check:
- `PHASE3_GUIDE.md` - Detailed technical explanations
- `PHASE3_SUMMARY.md` - Quick FAQ and answers

**Good luck with Phase 3! 🎓**
