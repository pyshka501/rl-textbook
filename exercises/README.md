# Exercise Solutions — Reinforcement Learning Textbook

This directory contains detailed solution files for all 17 chapters of the textbook. Each file provides worked solutions with mathematical derivations, proofs, code snippets, and conceptual explanations.

## Directory

| File | Chapter | Title | Exercises |
|------|---------|-------|-----------|
| [ch01_solutions.md](ch01_solutions.md) | 1 | Introduction to Reinforcement Learning | 4 |
| [ch02_solutions.md](ch02_solutions.md) | 2 | Multi-Armed Bandits | 10 |
| [ch03_solutions.md](ch03_solutions.md) | 3 | Markov Decision Processes and the Bellman Equation | 10 |
| [ch04_solutions.md](ch04_solutions.md) | 4 | Dynamic Programming | 10 |
| [ch05_solutions.md](ch05_solutions.md) | 5 | Monte Carlo Methods | 10 |
| [ch06_solutions.md](ch06_solutions.md) | 6 | Temporal-Difference Learning | 12 |
| [ch07_solutions.md](ch07_solutions.md) | 7 | Value Function Approximation and Deep RL | 9 |
| [ch08_solutions.md](ch08_solutions.md) | 8 | Policy Gradient Methods | 10 |
| [ch09_solutions.md](ch09_solutions.md) | 9 | Actor-Critic Methods and PPO | 10 |
| [ch10_solutions.md](ch10_solutions.md) | 10 | Reward Shaping and Reward Models | 9 |
| [ch11_solutions.md](ch11_solutions.md) | 11 | Language Models and Post-Training | 10 |
| [ch12_solutions.md](ch12_solutions.md) | 12 | Direct Preference Optimisation | 9 |
| [ch13_solutions.md](ch13_solutions.md) | 13 | GRPO and Online RL with Verifiable Rewards | 10 |
| [ch14_solutions.md](ch14_solutions.md) | 14 | Practical RLHF | 7 |
| [ch15_solutions.md](ch15_solutions.md) | 15 | Reasoning, Tool Use, and Reinforcement Finetuning | 9 |
| [ch16_solutions.md](ch16_solutions.md) | 16 | Multi-Agent Systems and RL | 8 |
| [ch17_solutions.md](ch17_solutions.md) | 17 | Open Problems and Emerging Directions | 7 discussion topics |

**Total: 154 exercises and discussion problems solved across 17 chapters.**

---

## Solution Format

Each file follows this structure:

```
# Chapter X: Title — Solutions

## Exercise X.1
**Statement:** [brief restatement of the exercise]

**Solution:** [detailed solution with math and/or code]

---
```

- **Mathematical notation:** LaTeX inline math `$...$` and display math `$$...$$`
- **Proofs:** End with $\blacksquare$ or "Q.E.D."
- **Code:** Python snippets for computational exercises
- **Tables:** Markdown tables for comparisons

---

## Coverage by Type

### Theoretical / Proof Exercises
- Bandit lower bounds (Ch 2): Lai–Robbins bound, UCB derivation
- Bellman equations (Ch 3–4): contraction proofs, convergence rates
- Policy gradient theorem (Ch 8): score function, causality lemma
- DPO derivation (Ch 12): from KL objective to final loss
- GRPO properties (Ch 13): bias analysis, KL estimator

### Computational / Numerical Exercises
- MDP value functions (Ch 3, 4): matrix inversions, iteration counts
- GAE computation (Ch 6, 9, 14): backward recursion examples
- Reward model loss (Ch 10): gradients and saturation
- GRPO advantages (Ch 13): worked numerical examples

### Algorithm Design / Open-Ended
- Exploration strategies (Ch 2, 7)
- Parallelism strategies (Ch 14)
- Multi-agent mechanism design (Ch 16)
- Open research problems (Ch 17)

### Coding Exercises
- Thompson Sampling simulation (Ch 2)
- GRPO training loop (Ch 13)
- Offline RL (Ch 17 notebook)

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $V^\pi(s)$ | State value function under policy $\pi$ |
| $Q^\pi(s,a)$ | Action-value function |
| $A^\pi(s,a)$ | Advantage function $Q^\pi - V^\pi$ |
| $\pi_\theta$ | Policy parameterised by $\theta$ |
| $\pi_\mathrm{ref}$ | Reference/base model policy |
| $\delta_t$ | TD error at time $t$ |
| $\hat{A}_t^{\mathrm{GAE}}$ | GAE advantage estimate |
| $\rho_{t:T}$ | Importance sampling ratio |
| $\mathrm{KL}(\pi\|\pi')$ | KL divergence from $\pi'$ to $\pi$ |
| $\sigma(x)$ | Logistic sigmoid $1/(1+e^{-x})$ |
| $\gamma$ | Discount factor |
| $\lambda$ | TD($\lambda$) trace decay |
| $\beta$ | KL regularisation coefficient |
| $\varepsilon$ | PPO clipping parameter |
