# Chapter 13: GRPO and Online RL with Verifiable Rewards — Solutions

## Exercise 13.1
**Statement:** GRPO advantage by hand with $G=5$ completions, rewards $r=(0,1,1,0,1)$.

**Solution:**

$$\mu_G = \frac{0+1+1+0+1}{5} = \frac{3}{5} = 0.6.$$

$$\sigma_G = \sqrt{\frac{(0-0.6)^2+(1-0.6)^2+(1-0.6)^2+(0-0.6)^2+(1-0.6)^2}{5}} = \sqrt{\frac{0.36+0.16+0.16+0.36+0.16}{5}} = \sqrt{\frac{1.2}{5}} = \sqrt{0.24} \approx 0.490.$$

$$\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}:$$

| $i$ | $r_i$ | $\hat{A}_i$ |
|-----|--------|------------|
| 1 | 0 | $(0-0.6)/0.490 = -1.225$ |
| 2 | 1 | $(1-0.6)/0.490 = +0.816$ |
| 3 | 1 | $+0.816$ |
| 4 | 0 | $-1.225$ |
| 5 | 1 | $+0.816$ |

**Sum:** $-1.225+0.816+0.816-1.225+0.816 = -1.225\times2 + 0.816\times3 = -2.45+2.449 \approx -0.001 \approx 0$.

Exact zero is not guaranteed because $\sum_i \hat{A}_i = \sum_i (r_i-\mu)/\sigma = 0/\sigma = 0$. Wait — this IS exactly zero since $\sum_i(r_i-\mu) = \sum r_i - G\mu = G\mu - G\mu = 0$. The small discrepancy is floating-point. $\sum_i\hat{A}_i = 0$ exactly. $\blacksquare$

---

## Exercise 13.2
**Statement:** RLOO advantage for same rewards; compare with GRPO.

**Solution:**

RLOO advantage: $\hat{A}_i^{\mathrm{RLOO}} = \frac{K}{K-1}(r_i - \bar{r})$, where $\bar{r} = \mu_G = 0.6$, $K=5$.

$$\frac{K}{K-1} = \frac{5}{4} = 1.25.$$

| $i$ | $r_i$ | RLOO: $1.25(r_i-0.6)$ | GRPO: $\hat{A}_i$ |
|-----|--------|----------------------|-------------------|
| 1 | 0 | $-0.75$ | $-1.225$ |
| 2 | 1 | $+0.50$ | $+0.816$ |
| 3 | 1 | $+0.50$ | $+0.816$ |
| 4 | 0 | $-0.75$ | $-1.225$ |
| 5 | 1 | $+0.50$ | $+0.816$ |

**Direction agreement:** Both estimators agree on the sign of advantages: completions with $r_i=1$ get positive advantages, $r_i=0$ get negative. The *direction* of the gradient is the same.

They give the same direction when $\sigma_G > 0$ — i.e., whenever there is variation in the group rewards. The GRPO normalises by $\sigma_G$, making it robust to reward scale; RLOO uses only the group mean as a baseline (no variance normalisation).

---

## Exercise 13.3
**Statement:** Degenerate groups with all equal rewards; why desirable; modification.

**Solution:**

When all completions receive the same reward $r_i = c$: $\mu_G = c$, $\sigma_G = 0$, so the GRPO advantage $\hat{A}_i = (c-c)/0 = 0/0$ is undefined (division by zero in standard form).

In practice, $\hat{A}_i = 0$ for all $i$ (set by convention when $\sigma_G = 0$).

**Why desirable:** If all completions are equally good (or equally bad), there is no contrastive signal — no way to distinguish which token sequences are better. Providing zero gradient avoids making arbitrary updates that could harm already-correct completions.

**Modification:** To extract signal from uniform groups, one could:
1. **Absolute threshold:** If $c \ge \tau$ (all correct), still push up the probability using $\hat{A}_i = \epsilon > 0$; if $c < \tau$ (all wrong), push down.
2. **Reference model comparison:** Use $\hat{A}_i = r_i - \mathbb{E}_{\pi_\mathrm{ref}}[r]$ (compare to the reference policy's expected reward rather than the group mean).
3. **Mixed batches:** Combine groups to ensure variation, but this mixes different prompts and conflates their signals.

---

## Exercise 13.4
**Statement:** Prove the bias of the group-mean baseline in REINFORCE is $\frac{1}{G}\nabla_\theta J(\theta)$.

**Solution:**

The REINFORCE gradient with leave-one-out (LOO) baseline is unbiased:
$$\hat{g}_i^{\mathrm{LOO}} = (r_i - \bar{r}_{-i})\nabla_\theta\log\pi_\theta(y_i|x), \quad \bar{r}_{-i} = \frac{1}{G-1}\sum_{j\neq i}r_j.$$

The group-mean baseline uses $\bar{r} = \frac{1}{G}\sum_j r_j$ instead.

Bias of the group-mean estimator (compared to LOO):
$$\mathrm{Bias} = \mathbb{E}\left[\sum_i (r_i - \bar{r})\nabla_\theta\log\pi_\theta(y_i|x)\right] - \mathbb{E}\left[\sum_i(r_i-\bar{r}_{-i})\nabla_\theta\log\pi_\theta(y_i|x)\right].$$

Note $\bar{r} - \bar{r}_{-i} = \frac{1}{G}r_i - \frac{1}{G(G-1)}\sum_{j\neq i}r_j + ... = \frac{1}{G}(r_i-\bar{r}_{-i}) \cdot \frac{G-1}{G-1}$... 

More directly: $\bar{r} = \frac{1}{G}r_i + \frac{G-1}{G}\bar{r}_{-i}$, so $r_i - \bar{r} = \frac{G-1}{G}(r_i - \bar{r}_{-i})$.

The group-mean gradient is $\frac{G-1}{G}$ times the LOO gradient. Since the LOO gradient is unbiased ($\mathbb{E}[\hat{g}^{\mathrm{LOO}}] = \nabla_\theta J$):
$$\mathbb{E}\left[\sum_i(r_i-\bar{r})\nabla_\theta\log\pi_\theta\right] = \frac{G-1}{G}\nabla_\theta J.$$

Bias $= \frac{G-1}{G}\nabla_\theta J - \nabla_\theta J = -\frac{1}{G}\nabla_\theta J$.

The magnitude of the bias is $\frac{1}{G}\|\nabla_\theta J\|$. As $G\to\infty$, the bias $\to 0$. $\blacksquare$

---

## Exercise 13.5
**Statement:** Best-of-$N$ accuracy as function of $p$ and $N$; find $N$ for $p=0.1$ achieving 90% accuracy.

**Solution:**

Best-of-$N$ succeeds if at least one of $N$ independent samples is correct:
$$P(\text{at least one correct}) = 1 - (1-p)^N.$$

For $p=0.1$, target accuracy $0.9$:
$$1 - 0.9^N = 0.9 \implies 0.9^N = 0.1$$
$$N = \frac{\ln 0.1}{\ln 0.9} = \frac{-2.303}{-0.1054} \approx \mathbf{21.9}.$$

So $N = 22$ samples are needed. $\blacksquare$

---

## Exercise 13.6
**Statement:** Show the GRPO KL estimator is non-negative and equals zero iff $\pi_\theta = \pi_\mathrm{ref}$.

**Solution:**

The GRPO KL estimator at token position $t$ is:
$$\widehat{\mathrm{KL}}_t = \frac{\pi_\theta(y_t|s_t)}{\pi_\mathrm{ref}(y_t|s_t)} - \log\frac{\pi_\theta(y_t|s_t)}{\pi_\mathrm{ref}(y_t|s_t)} - 1.$$

Let $u = \pi_\theta(y_t|s_t)/\pi_\mathrm{ref}(y_t|s_t) > 0$. Then $\widehat{\mathrm{KL}}_t = u - \log u - 1$.

Using the inequality $e^x \ge 1+x$ (with equality iff $x=0$): setting $x = \log u$:
$$e^{\log u} \ge 1 + \log u \implies u \ge 1 + \log u \implies u - \log u - 1 \ge 0. \quad \blacksquare$$

Equality holds iff $\log u = 0$, i.e., $u = 1$, i.e., $\pi_\theta(y_t|s_t) = \pi_\mathrm{ref}(y_t|s_t)$. $\blacksquare$

---

## Exercise 13.7
**Statement:** Probability of non-trivial contrastive signal in a group of size $G$ for binary rewards.

**Solution:**

A group is "trivially uniform" if all $G$ samples are correct (reward 1) or all are incorrect (reward 0).

$$P(\text{non-trivial}) = 1 - p^G - (1-p)^G.$$

| $G$ | $p=0.1$ | $p=0.3$ | $p=0.5$ | $p=0.7$ | $p=0.9$ |
|-----|---------|---------|---------|---------|---------|
| 2   | 0.18    | 0.42    | 0.50    | 0.42    | 0.18    |
| 4   | 0.34    | 0.67    | 0.875   | 0.67    | 0.34    |
| 8   | 0.57    | 0.91    | 0.992   | 0.91    | 0.57    |
| 16  | 0.81    | 0.998   | 1.000   | 0.998   | 0.81    |

```python
import numpy as np
import matplotlib.pyplot as plt

ps = [0.1, 0.3, 0.5, 0.7, 0.9]
G_vals = np.arange(1, 33)
for p in ps:
    prob = 1 - p**G_vals - (1-p)**G_vals
    plt.plot(G_vals, prob, label=f'p={p}')
plt.xlabel('Group size G'); plt.ylabel('P(non-trivial signal)')
plt.legend(); plt.title('Coverage probability vs group size')
plt.show()
```

**Key insight:** For extreme $p$ (near 0 or 1), large $G$ is needed to get contrastive signal. For $p=0.5$, even $G=4$ gives 87.5% chance of useful signal.

---

## Exercise 13.8
**Statement:** Propose GRPO modification incorporating step-level (process) rewards.

**Solution:**

Let $r_i^{(t)}$ be the process reward assigned after step $t$ of completion $i$ (instead of a single outcome reward $r_i$).

**Modified advantage:** Compute a step-level group baseline:
$$\mu_G^{(t)} = \frac{1}{G}\sum_{j=1}^G r_j^{(t)}, \quad \sigma_G^{(t)} = \mathrm{std}_j\{r_j^{(t)}\}.$$

The group-normalised step advantage:
$$\hat{A}_i^{(t)} = \frac{r_i^{(t)} - \mu_G^{(t)}}{\sigma_G^{(t)} + \epsilon}.$$

**Modified objective:** For each token $t$ in completion $i$, use the corresponding step advantage:
$$\mathcal{L}_\mathrm{GRPO-process} = -\sum_{i=1}^G\sum_{t=1}^{T_i}\hat{A}_i^{(t)}\cdot\min\!\left(r_{it}(\theta), \mathrm{clip}(r_{it}(\theta), 1-\varepsilon, 1+\varepsilon)\right),$$
where $r_{it}(\theta) = \pi_\theta(y_{it}|s_{it})/\pi_{\theta_\mathrm{old}}(y_{it}|s_{it})$.

This assigns different advantage signals to different reasoning steps, providing dense credit assignment without a learned critic — only the group-level comparison of step rewards is needed.

---

## Exercise 13.9
**Statement:** Compare rejection sampling fine-tuning vs GRPO for $p=0.2$, $N=16$.

**Solution:**

**Rejection sampling fine-tuning (RSF):** Generate 16 completions per prompt; keep only the correct ones; train on them for one epoch.

- With $p=0.2$: expected correct completions per prompt = $16\times0.2 = 3.2$.
- Uses 3.2/16 = 20% of generated tokens for training.
- Gradient: computed only on positive examples — no signal from incorrect completions.
- Data utilisation: **low** (80% of generated data discarded).

**GRPO with $G=16$:** All 16 completions used, advantages computed by group normalisation.

- 16/16 = 100% of generated tokens contribute to the gradient.
- Both correct completions (positive advantage) and incorrect ones (negative advantage) provide signal.
- Gradient quality: **higher** — negative examples explicitly discourage wrong reasoning patterns.

**Trade-offs:**

| Dimension | RSF | GRPO |
|-----------|-----|------|
| Data utilisation | 20% (only correct) | 100% |
| Gradient quality | Positive signal only | Positive + negative |
| Compute cost | Same generation cost | Same, but full backward pass |
| Implementation | Simpler | Requires PPO clipping + KL |
| Stability | More stable (supervised) | Can be unstable if rewards noisy |

GRPO uses data more efficiently and provides richer gradient signal; RSF is simpler and more stable but wasteful.

---

## Exercise 13.10
**Statement:** Implement simplified GRPO training loop in Python pseudocode.

**Solution:**

```python
import torch
import torch.nn.functional as F

def grpo_step(model, ref_model, prompts, reward_fn, 
               G=8, clip_eps=0.2, beta=0.01):
    """
    Simplified GRPO training step.
    
    Args:
        model: policy model (trainable)
        ref_model: frozen reference model
        prompts: list of input prompts
        reward_fn: callable(prompt, completion) -> float
        G: group size (completions per prompt)
        clip_eps: PPO clipping parameter
        beta: KL penalty coefficient
    """
    all_losses = []
    
    for x in prompts:
        # (i) Sample G completions per prompt
        completions = [model.generate(x) for _ in range(G)]
        
        # (ii) Compute group-normalised advantages
        rewards = torch.tensor([reward_fn(x, y) for y in completions])
        mu = rewards.mean()
        sigma = rewards.std() + 1e-8  # avoid division by zero
        advantages = (rewards - mu) / sigma  # shape: (G,)
        
        # (iii) Compute log-probs under current and old policy
        with torch.no_grad():
            old_log_probs = torch.stack([
                model.log_probs(x, y) for y in completions  # shape: (G, T)
            ])
            ref_log_probs = torch.stack([
                ref_model.log_probs(x, y) for y in completions
            ])
        
        curr_log_probs = torch.stack([
            model.log_probs(x, y) for y in completions
        ])
        
        # Per-token probability ratio
        ratio = torch.exp(curr_log_probs - old_log_probs)  # (G, T)
        
        # (iii) Clipped surrogate loss
        # Broadcast advantages over tokens: (G,) -> (G, 1)
        A = advantages.unsqueeze(1)
        clipped_ratio = ratio.clamp(1 - clip_eps, 1 + clip_eps)
        surrogate = torch.min(ratio * A, clipped_ratio * A)
        policy_loss = -surrogate.mean()
        
        # (iv) KL penalty: E[pi_theta/pi_ref - log(pi_theta/pi_ref) - 1]
        log_ratio_kl = curr_log_probs - ref_log_probs  # (G, T)
        kl_est = (torch.exp(log_ratio_kl) - log_ratio_kl - 1).mean()
        
        loss = policy_loss + beta * kl_est
        all_losses.append(loss)
    
    total_loss = torch.stack(all_losses).mean()
    total_loss.backward()
    return total_loss.item()
```

**Key design choices:**
1. Advantages are computed per-group (not across groups), providing prompt-specific baselines.
2. The KL estimator uses the variance-reduced form $e^x - x - 1 \ge 0$ (Exercise 13.6).
3. The clipped surrogate prevents large policy updates per step.
4. The reference model is frozen (no gradients) to avoid modifying the baseline.
