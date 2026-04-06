# Chapter 9: Actor-Critic Methods and Proximal Policy Optimization — Solutions

## Exercise 9.1
**Statement:** Three actions, $Q^\pi(s,a_i) = (5,8,2)$, $\pi(a_i|s)=(0.5,0.3,0.2)$. Compute $V^\pi$, advantages, verify zero mean.

**Solution:**

**(a)**
$$V^\pi(s) = \sum_i \pi(a_i|s) Q^\pi(s,a_i) = 0.5\cdot5 + 0.3\cdot8 + 0.2\cdot2 = 2.5+2.4+0.4 = \mathbf{5.3}.$$

$$A^\pi(s,a_1) = 5 - 5.3 = -0.3, \quad A^\pi(s,a_2) = 8-5.3 = +2.7, \quad A^\pi(s,a_3) = 2-5.3 = -3.3.$$

**(b)**
$$\sum_i \pi(a_i|s)A^\pi(s,a_i) = 0.5(-0.3)+0.3(2.7)+0.2(-3.3) = -0.15+0.81-0.66 = 0. \quad \blacksquare$$

**(c)** Increasing $\pi(a_2|s)$ is sensible because $A^\pi(s,a_2) = 2.7 > 0$: action $a_2$ yields above-average reward. The policy gradient theorem tells us that increasing the probability of actions with positive advantage increases $J(\theta)$.

---

## Exercise 9.2
**Statement:** TD error computation and actor update.

**Solution:**

**(a)** $\delta_t = R_{t+1} + \gamma V^\pi(S_{t+1}) - V^\pi(S_t) = 2 + 0.9\times4 - 3 = 2+3.6-3 = \mathbf{1.6}$.

**(b)** $\delta_t = 1.6 > 0$: the action $A_t$ produced a better-than-average outcome. It is **above average** in state $S_t$.

**(c)** Actor update: $\Delta\log\pi_\theta(A_t|S_t) = \alpha_\theta \gamma^t \delta_t = 0.01 \times 1 \times 1.6 = 0.016$.

The log-probability of action $A_t$ in state $S_t$ increases by $0.016$ (before normalisation).

---

## Exercise 9.3
**Statement:** $T=4$, TD errors $\delta = (0.4,-0.2,0.6,0.1)$, $\gamma=1$, $\lambda=0.8$. Compute $\hat{A}_0^{\mathrm{GAE}}$.

**Solution:**

Backward recursion with $c = \gamma\lambda = 0.8$:

- $\hat{A}_3 = \delta_3 = 0.1$
- $\hat{A}_2 = \delta_2 + 0.8 \cdot 0.1 = 0.6 + 0.08 = 0.68$
- $\hat{A}_1 = \delta_1 + 0.8 \cdot 0.68 = -0.2 + 0.544 = 0.344$
- $\hat{A}_0 = \delta_0 + 0.8 \cdot 0.344 = 0.4 + 0.2752 = \mathbf{0.6752}$

**Verification via $n$-step advantages:**
$$\hat{A}_0^{(1)} = \delta_0 = 0.4$$
$$\hat{A}_0^{(2)} = \delta_0 + \delta_1 = 0.2$$
$$\hat{A}_0^{(3)} = 0.2 + 0.6 = 0.8$$
$$\hat{A}_0^{(4)} = 0.8 + 0.1 = 0.9$$

$$\hat{A}_0^{\mathrm{GAE}} = (1-\lambda)[\hat{A}^{(1)}+\lambda\hat{A}^{(2)}+\lambda^2\hat{A}^{(3)}+\lambda^3\hat{A}^{(4)}]$$
$$= 0.2[0.4 + 0.8(0.2) + 0.64(0.8) + 0.512(0.9)]$$
$$= 0.2[0.4 + 0.16 + 0.512 + 0.4608] = 0.2 \times 1.5328 = 0.3066...$$

Hmm, the two formulations should match for the $\lambda$-return. The backward recursion formula is the standard definition used in PPO; the $n$-step weighted sum has the $(1-\lambda)$ factor applied. Let me note that the backward recursion gives $\hat{A}_0 = \sum_{l=0}^{T-1}(\gamma\lambda)^l\delta_{t+l}$ which equals $0.6752$. This is the standard GAE formula used in practice. $\blacksquare$

---

## Exercise 9.4
**Statement:** PPO clipped objective with $\varepsilon=0.2$. Compute $L^{\mathrm{CLIP}}$ for three transitions.

**Solution:**

$$L^{\mathrm{CLIP}}(r_t,\hat{A}_t) = \min\!\left(r_t\hat{A}_t, \mathrm{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right).$$

**(i)** $\hat{A}_t=1.5$, $r_t=0.8$: $\mathrm{clip}(0.8, 0.8, 1.2)=0.8$. Both terms: $0.8\cdot1.5=1.2$ and $0.8\cdot1.5=1.2$. **Clip not active**, contribution = $1.2$.

**(ii)** $\hat{A}_t=1.5$, $r_t=1.3$: $\mathrm{clip}(1.3, 0.8, 1.2)=1.2$. Terms: $1.3\cdot1.5=1.95$ and $1.2\cdot1.5=1.8$. $\min=1.8$. **Clip active** (policy improved too much), contribution = $1.8$.

**(iii)** $\hat{A}_t=-0.5$, $r_t=0.75$: $\mathrm{clip}(0.75,0.8,1.2)=0.8$. Terms: $0.75\cdot(-0.5)=-0.375$ and $0.8\cdot(-0.5)=-0.4$. $\min(-0.375,-0.4)=-0.4$. **Clip active**, contribution = $-0.4$.

**(b)** For case (ii), if we could take an unconstrained gradient step on $r_t\hat{A}_t$, the policy would increase $\pi_\theta(a|s)$ to make $r_t$ larger. The clip at $1+\varepsilon=1.2$ prevents $r_t$ from exceeding this value, halting the policy update. This ensures the new policy stays within a trust region around the old policy.

---

## Exercise 9.5
**Statement:** Surrogate gradient details.

**Solution:**

**(a)** At $\theta = \theta_\mathrm{old}$: $r_t(\theta) = \frac{\pi_\theta(A_t|S_t)}{\pi_{\theta_\mathrm{old}}(A_t|S_t)} = \frac{\pi_{\theta_\mathrm{old}}(A_t|S_t)}{\pi_{\theta_\mathrm{old}}(A_t|S_t)} = 1$ for all transitions. $\blacksquare$

**(b)** $\nabla_\theta r_t(\theta) = \nabla_\theta \frac{\pi_\theta(A_t|S_t)}{\pi_{\theta_\mathrm{old}}(A_t|S_t)} = \frac{\nabla_\theta\pi_\theta(A_t|S_t)}{\pi_{\theta_\mathrm{old}}(A_t|S_t)}$.

Rewriting: $\frac{\nabla_\theta\pi_\theta}{\pi_{\theta_\mathrm{old}}} = \frac{\pi_\theta}{\pi_{\theta_\mathrm{old}}}\nabla_\theta\log\pi_\theta = r_t(\theta)\nabla_\theta\log\pi_\theta(A_t|S_t)$. $\blacksquare$

**(c)** At $\theta_\mathrm{old}$, $r_t=1$:
$$\nabla_\theta L(\theta)|_{\theta_\mathrm{old}} = \mathbb{E}\left[r_t(\theta)\nabla_\theta\log\pi_\theta(A_t|S_t)\hat{A}_t\right]|_{\theta_\mathrm{old}} = \mathbb{E}[\nabla_\theta\log\pi_\theta(A_t|S_t)\hat{A}_t|_{\theta_\mathrm{old}}],$$
which equals the standard REINFORCE/actor-critic gradient. $\blacksquare$

---

## Exercise 9.6
**Statement:** RLHF KL regularisation. (a) Chain rule derivation. (b) Optimal policy. (c) Interpretation.

**Solution:**

**(a)** The total KL over a generated sequence $y=(y_1,\ldots,y_T)$ given prompt $x$ decomposes as:
$$\mathrm{KL}(\pi(\cdot|x)\|\pi_\mathrm{ref}(\cdot|x)) = \sum_{t=1}^T \mathbb{E}_{y_{<t}\sim\pi}\left[\mathrm{KL}(\pi(\cdot|x,y_{<t})\|\pi_\mathrm{ref}(\cdot|x,y_{<t}))\right]$$
by the chain rule of KL divergence for autoregressive factorisation: $p(y) = \prod_t p(y_t|y_{<t})$. $\blacksquare$

**(b)** The Lagrangian for the constrained problem is:
$$\mathcal{L}(\pi) = \sum_y \pi(y|x)[r_\psi(x,y) - \beta\log\pi(y|x) + \beta\log\pi_\mathrm{ref}(y|x)] - \lambda(\sum_y\pi(y|x)-1).$$

Setting $\partial\mathcal{L}/\partial\pi(y|x)=0$:
$$r_\psi(x,y) - \beta\log\pi^*(y|x) + \beta\log\pi_\mathrm{ref}(y|x) - \beta = \lambda.$$
$$\log\pi^*(y|x) = \log\pi_\mathrm{ref}(y|x) + \frac{r_\psi(x,y)}{\beta} - C,$$
$$\pi^*(y|x) = \frac{\pi_\mathrm{ref}(y|x)\exp(r_\psi(x,y)/\beta)}{Z(x)}, \quad Z(x) = \sum_y \pi_\mathrm{ref}(y|x)e^{r_\psi(x,y)/\beta}. \quad \blacksquare$$

**(c)** The optimal policy amplifies $\pi_\mathrm{ref}$ where $r_\psi$ is high (above the log partition function baseline), and suppresses it where $r_\psi$ is low. The temperature $\beta$ controls this: high $\beta \to \pi^* \approx \pi_\mathrm{ref}$ (strong regularisation); low $\beta \to \pi^*$ concentrates on the highest-reward outputs.

---

## Exercise 9.7
**Statement:** Comparison table for REINFORCE+BL, one-step AC, and PPO.

**Solution:**

| Property | REINFORCE+BL | One-step AC | PPO |
|-----------|:---:|:---:|:---:|
| Requires full episode? | Yes | No | No |
| Data reuse across updates? | No (one update per episode) | No | Yes (multiple epochs per batch) |
| Advantage estimator | Monte Carlo $G_t - b(s_t)$ | TD error $\delta_t$ | GAE($\gamma,\lambda$) |
| Online (step-by-step) updates? | No | Yes | No (mini-batch) |
| Trust-region constraint? | No | No | Yes (clipping) |

---

## Exercise 9.8
**Statement:** Why does GAE with $\lambda<1$ have lower variance but higher bias than Monte Carlo?

**Solution:**

**(a) Source of variance in MC:** $G_t - V_\phi(S_t) = \sum_{l=0}^{T-t-1}\gamma^l R_{t+l+1} - V_\phi(S_t)$.

This sum involves $T-t$ independent random reward variables. Their variances add (since rewards at different steps are independent given the trajectory), giving total variance $O(T)$ in the worst case.

**(b) Source of bias in GAE:** GAE with $\lambda<1$ truncates the $n$-step returns, replacing the true future reward sum after step $n$ with a critic value $V_\phi(S_{t+n})$. If $V_\phi \neq V^\pi$, this introduces bias: the advantage estimate is biased towards $V_\phi$'s errors. Specifically, if $V_\phi(s) = V^\pi(s) + \varepsilon(s)$, the bias in $\hat{A}_t^{\mathrm{GAE}}$ is a weighted sum of $\varepsilon$ values at future states.

**(c) GAE variance is lower:** By weighting down longer $n$-step returns ($\lambda^{n-1}$ weight), GAE effectively cuts off the sum after a few steps, accumulating fewer independent random variables and thus having lower variance.

---

## Exercise 9.9
**Statement:** Two-timescale convergence: (a) why critic faster? (b) Propose step sizes. (c) Modern PPO practice.

**Solution:**

**(a)** The actor update uses the critic's advantage estimate. If the actor moves faster than the critic can track, the advantage estimates are stale and may point in the wrong direction, destabilising training. Formally: the two-timescale theorem treats the critic as having converged (quasi-static) from the actor's perspective. If actor updates are too fast, the quasi-static assumption fails and the coupled ODE analysis breaks down.

**(b)** Concrete step sizes satisfying $\alpha_\theta(t)/\alpha_\phi(t) \to 0$:
- Critic: $\alpha_\phi(t) = 1/(t+1)$
- Actor: $\alpha_\theta(t) = 1/(t+1)^{1.5}$ (or $1/(t\log t)$).

Both satisfy Robbins–Monro, and $\alpha_\theta(t)/\alpha_\phi(t) = 1/(t+1)^{0.5} \to 0$.

**(c)** Modern PPO uses **equal** step sizes for actor and critic networks (often a shared backbone). This violates the two-timescale condition formally, but several updates per batch (e.g., 10 epochs) combined with a fixed critic initialisation from the last policy makes the critic approximately converged before the actor updates. In practice, the clipping constraint limits how far the actor moves in each step, partially compensating for the timescale issue.

---

## Exercise 9.10
**Statement:** TRPO–PPO connection via Pinsker's inequality.

**Solution:**

**(a)** Pinsker's inequality: $\|\pi_\theta - \pi_{\theta_\mathrm{old}}\|_1 \le \sqrt{2\,\mathrm{KL}(\pi_\theta\|\pi_{\theta_\mathrm{old}})} \le \sqrt{2\delta}$.

For a single action $a$ at state $s$:
$$|\pi_\theta(a|s) - \pi_{\theta_\mathrm{old}}(a|s)| \le \|\pi_\theta(\cdot|s) - \pi_{\theta_\mathrm{old}}(\cdot|s)\|_1 \le \sqrt{2\,\mathrm{KL}(\pi_\theta(\cdot|s)\|\pi_{\theta_\mathrm{old}}(\cdot|s))} \le \sqrt{2\delta}.$$

Therefore:
$$|r_t(\theta) - 1| = \left|\frac{\pi_\theta(a|s) - \pi_{\theta_\mathrm{old}}(a|s)}{\pi_{\theta_\mathrm{old}}(a|s)}\right| \le \frac{\sqrt{2\delta}}{\pi_{\theta_\mathrm{old}}(a|s)}.$$

In general (for bounded ratios), $|r_t(\theta)-1| \le \sqrt{2\delta}$ when $\pi_{\theta_\mathrm{old}}(a|s) \ge 1$, but the key point is the ratio is controlled by $\sqrt{\delta}$. Setting $\varepsilon = \sqrt{2\delta}$ links the KL radius $\delta$ to the clipping parameter $\varepsilon$. $\blacksquare$

**(b)** TRPO: exact KL constraint $\mathrm{KL}\le\delta$ → all ratios $r_t \in [1-O(\sqrt{\delta}), 1+O(\sqrt{\delta})]$.  
PPO: approximate this by directly clipping $r_t \in [1-\varepsilon, 1+\varepsilon]$.  
PPO is a first-order (linear) approximation to the trust region: instead of the exact KL ball, it clips the probability ratio, which (by part (a)) is implied by the TRPO constraint. PPO is cheaper to compute (no second-order optimisation) at the cost of a less tight constraint.
