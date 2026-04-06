# Chapter 2: Multi-Armed Bandits — Solutions

## Exercise 2.1
**Statement:** Prove that $\varepsilon$-greedy with fixed $\varepsilon > 0$ and at least one strictly sub-optimal action satisfies $\mathbb{E}[\mathcal{R}_T] = \Omega(T)$.

**Solution:**

Let $a^*$ be an optimal action and $a'$ a strictly sub-optimal action with gap $\Delta_{a'} = q_*(a^*) - q_*(a') > 0$.

Under $\varepsilon$-greedy, at every round $t$ the sub-optimal arm $a'$ is selected with probability at least $\varepsilon / K$ (where $K$ is the number of arms), since with probability $\varepsilon$ the agent explores uniformly.

Therefore:
$$\mathbb{E}[\mathcal{R}_T] = \sum_{t=1}^{T} \sum_{a \neq a^*} \Delta_a \, \mathbb{E}[N_t(a)] \ge \Delta_{a'} \cdot \frac{\varepsilon}{K} \cdot T.$$

Setting $c = \Delta_{a'} \varepsilon / K > 0$ gives $\mathbb{E}[\mathcal{R}_T] \ge cT = \Omega(T)$. $\blacksquare$

---

## Exercise 2.2
**Statement:** Suppose the optimal action has the highest current estimate. Find: (a) probability of selecting the optimal action; (b) probability of any sub-optimal action; (c) expected exploratory steps in $T$ rounds.

**Solution:**

With $K$ arms and $\varepsilon$-greedy:

**(a)** The optimal arm is selected greedily with prob $1-\varepsilon$, plus with prob $\varepsilon \cdot (1/K)$ during exploration:
$$P(\text{select } a^*) = 1 - \varepsilon + \frac{\varepsilon}{K}.$$

**(b)** Each sub-optimal arm $a \neq a^*$ is selected only during exploration:
$$P(\text{select } a) = \frac{\varepsilon}{K}.$$

**(c)** In each round, exploration occurs with probability $\varepsilon$, so the expected number of exploratory steps in $T$ rounds is $\varepsilon T$.

---

## Exercise 2.3
**Statement:** Let $R_1, \ldots, R_n$ be i.i.d. with mean $q_*(a)$ and variance $\sigma_a^2$. Prove $Q_n(a)$ is unbiased with variance $\sigma_a^2/n$.

**Solution:**

**Unbiasedness:**
$$\mathbb{E}[Q_n(a)] = \mathbb{E}\!\left[\frac{1}{n}\sum_{i=1}^{n} R_i\right] = \frac{1}{n}\sum_{i=1}^{n}\mathbb{E}[R_i] = \frac{n \cdot q_*(a)}{n} = q_*(a).$$

**Variance:** Since the $R_i$ are i.i.d.:
$$\mathrm{Var}[Q_n(a)] = \mathrm{Var}\!\left[\frac{1}{n}\sum_{i=1}^n R_i\right] = \frac{1}{n^2}\sum_{i=1}^n \mathrm{Var}[R_i] = \frac{n \sigma_a^2}{n^2} = \frac{\sigma_a^2}{n}. \quad \blacksquare$$

---

## Exercise 2.4
**Statement:** Derive the explicit form of the constant-step-size update $Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^{n}\alpha(1-\alpha)^{n-i}R_i$.

**Solution:**

The update rule is $Q_{n+1} = Q_n + \alpha(R_n - Q_n) = (1-\alpha)Q_n + \alpha R_n$.

Unrolling recursively:
$$Q_{n+1} = (1-\alpha)Q_n + \alpha R_n$$
$$= (1-\alpha)[(1-\alpha)Q_{n-1} + \alpha R_{n-1}] + \alpha R_n$$
$$= (1-\alpha)^2 Q_{n-1} + \alpha(1-\alpha)R_{n-1} + \alpha R_n.$$

Continuing for $n$ steps:
$$Q_{n+1} = (1-\alpha)^n Q_1 + \sum_{i=1}^{n} \alpha(1-\alpha)^{n-i} R_i. \quad \blacksquare$$

**Weights on past observations:** The weight on $R_i$ is $\alpha(1-\alpha)^{n-i}$. Since $0 < 1-\alpha < 1$, older observations (small $i$, large $n-i$) receive exponentially smaller weight. The sum of all weights is:
$$(1-\alpha)^n + \sum_{i=1}^n \alpha(1-\alpha)^{n-i} = (1-\alpha)^n + \alpha \cdot \frac{1-(1-\alpha)^n}{\alpha} = 1.$$

As $n \to \infty$, $(1-\alpha)^n \to 0$ so the initial value $Q_1$ is forgotten — an **exponential recency-weighted average**.

---

## Exercise 2.5
**Statement:** Construct a two-armed bandit where pure greedy locks onto the sub-optimal arm forever with positive probability.

**Solution:**

**Setup:** Two arms with deterministic rewards:
- Arm 1: $R = 1$ always ($q_*(1) = 1$)
- Arm 2: $R = 2$ always ($q_*(2) = 2$, the optimal arm)

Initialize $Q_1(1) = Q_1(2) = 0$. The greedy algorithm breaks ties by choosing arm 1 (say, lowest index).

**Execution:**
- Round 1: tie → select arm 1 → $Q_2(1) = 1$, $Q_2(2) = 0$.
- Round 2: $Q(1) > Q(2)$ → select arm 1 → $Q_3(1) = 1$, $Q_3(2) = 0$.
- All subsequent rounds: arm 1 has higher estimate → always select arm 1.

The algorithm never tries arm 2 and permanently misses the optimal arm. With probability 1 (the tie-break is deterministic), it locks onto the sub-optimal arm.

**With stochastic rewards:** Let Arm 1 ~ Bernoulli(0.9) and Arm 2 ~ Bernoulli(0.5). Starting from $Q_1(1)=Q_1(2)=0$, if the first pull of arm 1 returns $R=1$ and the first pull of arm 2 returns $R=0$, then greedy always exploits arm 1 thereafter with positive probability, despite arm 2 being optimal.

---

## Exercise 2.6
**Statement:** Show any algorithm with $P(A_t = a) \ge \delta > 0$ for all $a, t$ has $\mathbb{E}[\mathcal{R}_T] \ge cT$.

**Solution:**

Since there is at least one sub-optimal arm $a'$ with gap $\Delta_{a'} > 0$, and the algorithm selects it with probability $P(A_t = a') \ge \delta$ at every round:
$$\mathbb{E}[\mathcal{R}_T] = \sum_{t=1}^T \sum_a \Delta_a \, P(A_t = a) \ge \sum_{t=1}^T \Delta_{a'} \cdot \delta = \delta \Delta_{a'} T.$$

Set $c = \delta \Delta_{a'} > 0$. Then $\mathbb{E}[\mathcal{R}_T] \ge cT$. $\blacksquare$

---

## Exercise 2.7
**Statement:** Derive the UCB1 exploration bonus $\sqrt{2\ln t / N_t(a)}$ from Hoeffding's inequality.

**Solution:**

Hoeffding's inequality for bounded random variables $R_i \in [0,1]$ states:
$$P\!\left(\hat{\mu}_n - \mu \ge \epsilon\right) \le \exp(-2n\epsilon^2),$$
where $\hat{\mu}_n = \frac{1}{n}\sum_{i=1}^n R_i$ and $n = N_t(a)$.

We want the upper confidence bound to hold with failure probability $t^{-4}$ (setting $\exp(-2n\epsilon^2) = t^{-4}$):
$$2n\epsilon^2 = 4\ln t \implies \epsilon = \sqrt{\frac{2\ln t}{n}}.$$

Applying a union bound over all $K$ arms: the bound $q_*(a) \le Q_t(a) + \sqrt{2\ln t/N_t(a)}$ fails for any arm with probability at most $K \cdot t^{-4} \to 0$.

The UCB1 policy selects:
$$A_t = \arg\max_a \left[ Q_t(a) + \sqrt{\frac{2\ln t}{N_t(a)}} \right]. \quad \blacksquare$$

---

## Exercise 2.8
**Statement:** Simulate Thompson Sampling for a 3-armed Bernoulli bandit with $\theta = (0.3, 0.5, 0.7)$, Beta(1,1) priors, for $T=1000$ steps. Plot posteriors at $t = 10, 100, 1000$.

**Solution:**

Thompson Sampling maintains Beta posteriors. After observing $s$ successes and $f$ failures for arm $a$, the posterior is Beta$(1+s_a, 1+f_a)$.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

np.random.seed(42)

theta_true = [0.3, 0.5, 0.7]
K = 3
T = 1000

# Prior: Beta(1,1)
alpha = np.ones(K)
beta_  = np.ones(K)

snapshots = {10: None, 100: None, 1000: None}
rewards = []

for t in range(1, T+1):
    # Sample from posteriors
    samples = [np.random.beta(alpha[a], beta_[a]) for a in range(K)]
    arm = np.argmax(samples)
    # Observe reward
    r = int(np.random.random() < theta_true[arm])
    rewards.append(r)
    alpha[arm] += r
    beta_[arm] += (1 - r)
    if t in snapshots:
        snapshots[t] = (alpha.copy(), beta_.copy())

# Plot
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
x = np.linspace(0, 1, 200)
colors = ['#e94560', '#4fc3f7', '#69f0ae']
for idx, (t_snap, (a, b)) in enumerate(snapshots.items()):
    ax = axes[idx]
    for i in range(K):
        ax.plot(x, beta_dist.pdf(x, a[i], b[i]), color=colors[i],
                lw=2, label=f'Arm {i+1} (true θ={theta_true[i]})')
        ax.axvline(theta_true[i], color=colors[i], linestyle='--', alpha=0.5)
    ax.set_title(f't = {t_snap}')
    ax.set_xlabel('θ'); ax.set_ylabel('Density')
    ax.legend(fontsize=7)
plt.suptitle('Thompson Sampling: Posterior Evolution')
plt.tight_layout()
plt.show()

print(f"Cumulative regret at T=1000: {T*0.7 - sum(rewards):.1f}")
```

**Key observations:**
- At $t=10$: posteriors are broad, overlapping significantly.
- At $t=100$: arm 3 ($\theta=0.7$) begins to separate from the others.
- At $t=1000$: posteriors are concentrated near true values; arm 3 dominates sampling.

---

## Exercise 2.9
**Statement:** For Gaussian bandits with unit variance, the KL divergence is $\Delta_a^2/2$. Substitute into the Lai–Robbins bound.

**Solution:**

The Lai–Robbins lower bound states that for any consistent algorithm:
$$\liminf_{T\to\infty} \frac{\mathbb{E}[N_T(a)]}{\ln T} \ge \frac{1}{\mathrm{KL}(q_*(a) \| q_*(a^*))}.$$

For Gaussian arms with unit variance, $\mathrm{KL}(\mathcal{N}(\mu_a,1)\|\mathcal{N}(\mu^*,1)) = \frac{(\mu^* - \mu_a)^2}{2} = \frac{\Delta_a^2}{2}$.

Substituting:
$$\liminf_{T\to\infty} \frac{\mathbb{E}[N_T(a)]}{\ln T} \ge \frac{2}{\Delta_a^2}.$$

**Interpretation:** Arms with smaller gap $\Delta_a$ (harder to distinguish from the optimum) require *more* samples before they can be confidently eliminated. The necessary pulls scale as $2\ln T / \Delta_a^2$ — the same dependence as UCB1's exploration bonus, confirming UCB1 is asymptotically optimal.

---

## Exercise 2.10
**Statement:** Show the ridge regression estimate is MAP under a Gaussian prior, and that $\sqrt{x_t^\top A_t^{-1} x_t}$ is the posterior standard deviation.

**Solution:**

**MAP derivation:**  
The likelihood of observations is $R_s \mid x_{A_s,s}, \theta^* \sim \mathcal{N}(x_{A_s,s}^\top \theta^*, \sigma^2)$.  
The prior is $\theta^* \sim \mathcal{N}(0, \sigma^2 \lambda^{-1} I)$.

The log-posterior is:
$$\log p(\theta^* \mid \text{data}) = -\frac{1}{2\sigma^2}\sum_{s\le t}(R_s - x_{A_s,s}^\top \theta^*)^2 - \frac{\lambda}{2\sigma^2}\|\theta^*\|^2 + \text{const}.$$

Maximising is equivalent to minimising $\sum_s (R_s - x_s^\top \theta)^2 + \lambda\|\theta\|^2$, giving the ridge solution:
$$\hat{\theta}_t = A_t^{-1} b_t, \quad A_t = \lambda I + \sum_{s\le t} x_s x_s^\top, \quad b_t = \sum_{s\le t} R_s x_s. \quad \blacksquare$$

**Posterior standard deviation:**  
The posterior is $\theta^* \mid \text{data} \sim \mathcal{N}(\hat{\theta}_t, \sigma^2 A_t^{-1})$.  
For the linear predictor $x_t^\top \theta^*$:
$$\mathrm{Var}[x_t^\top \theta^* \mid \text{data}] = \sigma^2 \, x_t^\top A_t^{-1} x_t.$$

The posterior standard deviation is $\sigma\sqrt{x_t^\top A_t^{-1} x_t}$, which (up to $\sigma$) equals the LinUCB exploration bonus. $\blacksquare$
