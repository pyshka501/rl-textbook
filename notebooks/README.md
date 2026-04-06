# Practical Notebooks

Interactive Jupyter notebooks accompanying each chapter. **Click any badge to launch instantly in Google Colab — no installation required.**

## Running the Notebooks

### Google Colab (Recommended — 1 click)
Click the **Open in Colab** badge next to any notebook below. All dependencies install automatically in the first cell.

### Kaggle
Upload the notebook to Kaggle, enable "Internet" in settings, and select GPU T4 accelerator.

### Local
```bash
pip install -r requirements.txt
jupyter lab
```

## Notebook Index

| Ch | Notebook | Open in Colab | Topics | Runtime |
|----|----------|---------------|--------|---------|
| 1 | `ch01_introduction.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch01_introduction.ipynb) | Agent-environment loop, gridworld | ~2 min |
| 2 | `ch02_bandits.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch02_bandits.ipynb) | ε-greedy, UCB, Thompson Sampling | ~3 min |
| 3 | `ch03_mdp_bellman.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch03_mdp_bellman.ipynb) | MDP definition, Bellman evaluation | ~2 min |
| 4 | `ch04_dynamic_programming.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch04_dynamic_programming.ipynb) | Policy/Value Iteration, GridWorld | ~3 min |
| 5 | `ch05_monte_carlo.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch05_monte_carlo.ipynb) | First-visit MC, Blackjack | ~5 min |
| 6 | `ch06_td_learning.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch06_td_learning.ipynb) | TD(0), SARSA, Q-Learning | ~5 min |
| 7 | `ch07_deep_rl.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch07_deep_rl.ipynb) | DQN, experience replay | ~8 min |
| 8 | `ch08_policy_gradients.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch08_policy_gradients.ipynb) | REINFORCE, baselines, RLOO | ~8 min |
| 9 | `ch09_actor_critic_ppo.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch09_actor_critic_ppo.ipynb) | A2C, PPO-Clip | ~10 min |
| 10 | `ch10_reward_models.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch10_reward_models.ipynb) | Reward shaping, Bradley-Terry | ~5 min |
| 11 | `ch11_lm_post_training.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch11_lm_post_training.ipynb) | GPT-2, SFT, sampling | ~10 min |
| 12 | `ch12_dpo_alignment.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch12_dpo_alignment.ipynb) | DPO loss, implicit reward | ~10 min |
| 13 | `ch13_grpo.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch13_grpo.ipynb) | Group-relative advantages | ~8 min |
| 14 | `ch14_practical_rlhf.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch14_practical_rlhf.ipynb) | TRL library, PPO for LMs | ~15 min |
| 15 | `ch15_reasoning.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch15_reasoning.ipynb) | CoT, majority vote, tools | ~5 min |
| 16 | `ch16_multi_agent.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch16_multi_agent.ipynb) | Prisoner's Dilemma, self-play | ~5 min |
| 17 | `ch17_open_problems.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch17_open_problems.ipynb) | Decision Transformer, offline RL | ~8 min |

> **GPU:** Chapters 1–6 run on CPU. Chapters 7–17 benefit from a free T4 GPU (select Runtime → Change runtime type → T4 GPU in Colab).
