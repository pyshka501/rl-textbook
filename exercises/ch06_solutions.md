# Chapter 6: Temporal-Difference Learning — Solutions

## Exercise 6.1
**Statement:** $V(S_t)=3$, $R_{t+1}=1$, $\gamma=0.9$, $V(S_{t+1})=4$. (a) TD error. (b) TD(0) update with $\alpha=0.1$. (c) Interpret sign.

**Solution:**

**(a)** $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t) = 1 + 0.9 \times 4 - 3 = 1 + 3.6 - 3 = \mathbf{1.6}$.

**(b)** $V(S_t) \leftarrow V(S_t) + \alpha\delta_t = 3 + 0.1 \times 1.6 = \mathbf{3.16}$.

**(c)** $\delta_t > 0$: the bootstrapped target $R_{t+1} + \gamma V(S_{t+1}) = 4.6$ exceeds the current estimate $V(S_t) = 3$. The state was **undervalued** — the agent received better-than-expected reward, so the value is updated upward.

---

## Exercise 6.2
**Statement:** Prove $\mathbb{E}_\pi[\delta_t | S_t=s] = 0$ when $V = V^\pi$. Explain why it fails for $V \neq V^\pi$.

**Solution:**

$$\mathbb{E}_\pi[\delta_t | S_t=s] = \mathbb{E}_\pi[R_{t+1} + \gamma V(S_{t+1}) | S_t=s] - V(s)$$
$$= \mathbb{E}_\pi[R_{t+1} | S_t=s] + \gamma \mathbb{E}_\pi[V(S_{t+1}) | S_t=s] - V(s).$$

When $V = V^\pi$, the Bellman equation gives:
$$V^\pi(s) = \mathbb{E}_\pi[R_{t+1} | S_t=s] + \gamma \sum_{s'} p(s'|s,\pi(s)) V^\pi(s').$$

So $\mathbb{E}_\pi[\delta_t|S_t=s] = V^\pi(s) - V^\pi(s) = 0$. $\blacksquare$

**Step where Bellman equation is used:** The identity $V^\pi(s) = \mathbb{E}[R_{t+1}|S_t=s] + \gamma\mathbb{E}[V^\pi(S_{t+1})|S_t=s]$ is precisely the Bellman equation for $V^\pi$.

**When $V \neq V^\pi$:** The bootstrapped target $R_{t+1} + \gamma V(S_{t+1})$ uses $V$ (not $V^\pi$) as a surrogate for the future return. The expected target is then $V^\pi(s) + \gamma\mathbb{E}[V(S_{t+1}) - V^\pi(S_{t+1})|S_t=s]$, which differs from $V(s)$ by both the approximation error of $V$ and the mismatch of bootstrapping.

---

## Exercise 6.3
**Statement:** Cliff Walking: (a) why SARSA prefers the safe path; (b) effect of decreasing $\varepsilon$; (c) preference at test time.

**Solution:**

**(a)** SARSA is **on-policy**: it updates using the action actually taken, including exploratory steps. With $\varepsilon=0.1$, there is a $10\%$ chance of a random action at each step. On the cliff-edge path, a single random action causes a fall (reward $-100$). SARSA's Q-values reflect this risk: the TD updates incorporate the occasional $-100$ transitions, making the cliff-edge values low. The safer inland path has consistently low negative rewards ($-1$ per step) with no catastrophic outcomes, so SARSA prefers it.

**(b)** With $\varepsilon = 0.01$:
- **Q-learning**: still learns $Q^*$ (converges to optimal deterministic policy values). With less exploration, it more reliably learns the cliff-edge path is optimal for the greedy policy.
- **SARSA**: with less exploration, the probability of accidentally falling from the cliff-edge path drops, so the cliff-edge path's Q-values are less penalised. SARSA now also tends to learn the cliff-edge path.

**(c)** At test time with $\varepsilon=0$: **Q-learning** is preferred, because it learned the true optimal Q-function $Q^*$ (the cliff-edge path with optimal value). SARSA learned a safe but sub-optimal policy suited to its own exploratory behaviour. Since exploration stops at test time, Q-learning's policy is optimal.

---

## Exercise 6.4
**Statement:** Two-action MDP, true $Q^* = 0$. (a) Show $\mathbb{E}[\max(\hat{Q}_1, \hat{Q}_2)] > 0$. (b) How does bias grow with actions? (c) How does Double Q fix it?

**Solution:**

**(a)** Let $X_1, X_2 \sim \mathcal{N}(0, \sigma^2)$ i.i.d. Then $M = \max(X_1, X_2)$.

$$\mathbb{E}[\max(X_1,X_2)] = \mathbb{E}[|X_1|] / \sqrt{2} \cdot \sqrt{2} = \sigma \sqrt{2/\pi} > 0.$$

More precisely, for two i.i.d. $\mathcal{N}(0,\sigma^2)$: $\mathbb{E}[\max(X_1,X_2)] = \sigma\sqrt{2/\pi} \approx 0.564\sigma > 0$. $\blacksquare$

**(b)** For $K$ i.i.d. $\mathcal{N}(0,\sigma^2)$ variables, $\mathbb{E}[\max_{i=1}^K X_i] \approx \sigma\sqrt{2\ln K}$ (extreme value theory). The maximisation bias grows as $O(\sqrt{\ln K})$ with the number of actions.

**(c)** Double Q-learning maintains two independent Q-tables $Q_A$ and $Q_B$. The target uses one network to *select* the action and the other to *evaluate* it:
$$\text{Target} = R + \gamma Q_B(S', \arg\max_a Q_A(S',a)).$$

Since the action selection and evaluation use independent estimates, the two estimation errors are independent (mean zero), and their product has zero expectation. The maximisation bias is eliminated: $\mathbb{E}[Q_B(s', \arg\max_a Q_A(s',a))] = Q^*(s', a^*) + O(\sigma^2)$, where the bias is of order $\sigma^2$ (second-order) rather than $\sigma$ (first-order). $\blacksquare$

---

## Exercise 6.5
**Statement:** Five-state random walk, trajectory $B\to C\to D\to E\to R$ with rewards $(0,0,0,1)$, $\gamma=1$. Compute TD(0), 2-step, MC, and $\lambda$-return targets for state $B$.

**Solution:**

All values initialised at $0.5$. Trajectory: $B(0), C(1), D(2), E(3), R(\text{terminal})$.

**(a) TD(0) target for $B$:** $G_0^{(1)} = R_1 + \gamma V(C) = 0 + 1 \times 0.5 = \mathbf{0.5}$.

**(b) 2-step return for $B$:** $G_0^{(2)} = R_1 + \gamma R_2 + \gamma^2 V(D) = 0 + 0 + 0.5 = \mathbf{0.5}$.

**(c) MC return for $B$:** $G_0 = 0 + 0 + 0 + 1 = \mathbf{1}$.

**(d) $\lambda$-return with $\lambda=0.5$:**
$$G_0^\lambda = (1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} G_0^{(n)} = (1-\lambda)[G_0^{(1)} + \lambda G_0^{(2)} + \lambda^2 G_0^{(3)} + \lambda^3 G_0^{(4)}]$$
where $G_0^{(1)}=0.5$, $G_0^{(2)}=0.5$, $G_0^{(3)} = R_1+R_2+R_3+\gamma^3 V(E) = 0.5$, $G_0^{(4)} = R_1+R_2+R_3+R_4 = 1$ (MC).

$$G_0^\lambda = 0.5[0.5 + 0.5(0.5) + 0.25(0.5) + 0.125(1)]$$
$$= 0.5[0.5 + 0.25 + 0.125 + 0.125] = 0.5 \times 1.0 = \mathbf{0.5}.$$

(Note: $n=4$ is the full episode, all steps have $R=0$ except the terminal step.)

---

## Exercise 6.6
**Statement:** Trajectory $A,B,A,C$ with $\gamma=0.9$, $\lambda=0.8$. Compute eligibility traces.

**Solution:**

Accumulating traces: $e_t(s) = \gamma\lambda e_{t-1}(s) + \mathbf{1}[S_t=s]$.

| $t$ | $s$ visited | $e_t(A)$ | $e_t(B)$ | $e_t(C)$ |
|-----|-------------|-----------|-----------|-----------|
| 0   | $A$         | $0\cdot0.72 + 1 = 1$ | $0$ | $0$ |
| 1   | $B$         | $1\cdot0.72 = 0.72$ | $0\cdot0.72+1=1$ | $0$ |
| 2   | $A$         | $0.72\cdot0.72+1 = 0.5184+1=1.5184$ | $1\cdot0.72=0.72$ | $0$ |
| 3   | $C$         | $1.5184\cdot0.72=1.093$ | $0.72\cdot0.72=0.518$ | $0+1=1$ |

Here $\gamma\lambda = 0.9 \times 0.8 = 0.72$.

**Verification:** $e_3(A) = 1.093 > e_3(B) = 0.518$: state $A$ was visited twice and more recently than $B$, so it correctly has the highest trace. $\blacksquare$

---

## Exercise 6.7
**Statement:** State and verify the forward–backward equivalence for a 2-step episode.

**Solution:**

**Forward–backward equivalence theorem:** The $\lambda$-return update $\sum_{t=0}^{T-1}\alpha(G_t^\lambda - V(S_t))\nabla V(S_t)$ equals the eligibility trace update $\sum_{t=0}^{T-1}\alpha\delta_t e_t$ (in the offline batch case).

**Verification for 2-step episode** ($S_0, S_1, S_2=T$):

TD errors: $\delta_0 = R_1 + \gamma V(S_1) - V(S_0)$, $\delta_1 = R_2 + \gamma \cdot 0 - V(S_1) = R_2 - V(S_1)$.

$\lambda$-return for $S_0$: $G_0^\lambda = (1-\lambda)G_0^{(1)} + \lambda G_0^{(2)}$, where $G_0^{(1)} = R_1 + \gamma V(S_1)$, $G_0^{(2)} = R_1 + \gamma R_2$.

Forward update on $S_0$: $\alpha(G_0^\lambda - V(S_0))$.

Backward: traces at $t=0$: $e_0(S_0)=1$, $e_0(S_1)=0$. At $t=1$: $e_1(S_0) = \gamma\lambda$, $e_1(S_1)=1$.

Backward update on $V(S_0)$: $\alpha\delta_0 \cdot 1 + \alpha\delta_1 \cdot \gamma\lambda = \alpha[\delta_0 + \gamma\lambda\delta_1]$.

Forward update: $\alpha(G_0^\lambda - V(S_0)) = \alpha[(1-\lambda)(R_1+\gamma V(S_1))+\lambda(R_1+\gamma R_2) - V(S_0)]$
$= \alpha[R_1 + \gamma V(S_1) + \lambda\gamma(R_2 - V(S_1)) - V(S_0)]$
$= \alpha[\delta_0 + \lambda\gamma\delta_1]$. ✓ The two are equal. $\blacksquare$

---

## Exercise 6.8
**Statement:** GAE computation with $\gamma=0.99$, $\lambda=0.9$, TD errors $\delta = (1,0,-1,2,0)$.

**Solution:**

Backward recursion: $\hat{A}_T = 0$, $\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}$.

Let $c = \gamma\lambda = 0.99 \times 0.9 = 0.891$.

**(a)**
- $\hat{A}_4 = 0$
- $\hat{A}_3 = 2 + 0.891 \times 0 = 2.000$
- $\hat{A}_2 = -1 + 0.891 \times 2 = -1 + 1.782 = 0.782$
- $\hat{A}_1 = 0 + 0.891 \times 0.782 = 0.697$
- $\hat{A}_0 = 1 + 0.891 \times 0.697 = 1 + 0.621 = 1.621$

**(b)** If $\lambda=0$: $\hat{A}_t = \delta_t$, so $(1, 0, -1, 2, 0)$.  
If $\lambda=1$: $\hat{A}_t = \sum_{k=t}^{4}\gamma^{k-t}\delta_k$, giving:
- $\hat{A}_0 = 1 + 0.99(0) + 0.99^2(-1) + 0.99^3(2) + 0 = 1 - 0.980 + 1.941 = 1.961$

**(c)** For $\hat{A}_0$, a larger $\lambda$ places more weight on the future signal $\delta_3 = 2$. With $\lambda=1$: $\hat{A}_0 = 1.961$ (strongest reflection of $\delta_3$). With $\lambda=0$: $\hat{A}_0 = 1.0$ (ignores $\delta_3$ entirely). The value $\lambda=1$ gives the strongest propagation of $\delta_3=2$ back to $t=0$.

---

## Exercise 6.9
**Statement:** Deadly triad: (a) removing one element restores convergence. (b) MC + FA: missing element. (c) On-policy TD + linear FA: missing element. (d) Retain all three safely.

**Solution:**

**(a)**
- Remove **bootstrapping**: use MC targets (known ground truth), convergence follows since it's supervised regression.
- Remove **function approximation**: tabular TD converges under standard conditions.
- Remove **off-policy**: on-policy TD with linear FA converges to the TD fixed point.

**(b) MC + linear FA:** Missing **bootstrapping**. MC uses full-episode returns as targets — no bootstrap. Without bootstrapping, the update is a supervised regression with an (asymptotically) unbiased target.

**(c) On-policy TD + linear FA:** Missing **off-policy**. The on-policy distribution ensures $D_\mu$ is the stationary distribution, which prevents divergence. Tsitsiklis & Van Roy proved convergence to the TD fixed point in this setting.

**(d) Approaches that retain all three but mitigate divergence:**
- **Gradient TD (GTD2, TDC)**: use a true gradient of an objective (MSPBE), restoring convergence guarantees even off-policy with FA.
- **Experience replay with target networks** (DQN): slows down the moving target problem, empirically stabilises training.
- **Conservative or regularised objectives**: add $L_2$ regularisation on the weights.

---

## Exercise 6.10
**Statement:** Show $\alpha_t = c/(t+t_0)$ satisfies Robbins–Monro conditions. Show constant $\alpha_t = \alpha$ fails the second. Why is constant preferred in non-stationary settings?

**Solution:**

The Robbins–Monro conditions are: (1) $\sum_t \alpha_t = \infty$, (2) $\sum_t \alpha_t^2 < \infty$.

**(a)** For $\alpha_t = c/(t+t_0)$:

(1) $\sum_{t=0}^\infty \frac{c}{t+t_0}$: this is $c$ times the harmonic series (with offset), which diverges. ✓

(2) $\sum_{t=0}^\infty \frac{c^2}{(t+t_0)^2}$: this is $c^2$ times $\sum_t t^{-2}$, a $p$-series with $p=2>1$, which converges. ✓ $\blacksquare$

**(b)** For $\alpha_t = \alpha > 0$: $\sum_t \alpha^2 = \infty$. The second condition fails. $\blacksquare$

**(c) Non-stationary environments:** When $V^\pi$ changes over time (e.g., due to a changing reward distribution), the agent needs to *track* the moving target. A constant step size $\alpha$ gives exponentially weighted recency: old observations are forgotten (weight $(1-\alpha)^k \to 0$). This is desirable — the agent responds to recent changes. With a decaying step size converging to 0, the agent freezes its estimates and cannot adapt.

---

## Exercise 6.11
**Statement:** Show that Expected SARSA with greedy policy reduces to Q-learning.

**Solution:**

Expected SARSA update:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\left[R_{t+1} + \gamma \sum_a \pi(a|S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t)\right].$$

For the deterministic greedy policy $\pi(a|s) = \mathbf{1}[a = \arg\max_{a'} Q(s,a')]$:
$$\sum_a \pi(a|S_{t+1}) Q(S_{t+1}, a) = Q(S_{t+1}, \arg\max_{a'}Q(S_{t+1},a')) = \max_{a'} Q(S_{t+1}, a').$$

Substituting:
$$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\left[R_{t+1} + \gamma \max_{a'} Q(S_{t+1},a') - Q(S_t,A_t)\right],$$
which is exactly the Q-learning update. $\blacksquare$

---

## Exercise 6.12
**Statement:** Fill in variance table for TD(0), 10-step TD, MC with $T=100$, $\sigma^2=1$, $\gamma=0.99$.

**Solution:**

Using the approximation $\mathrm{Var}[G_t^{(n)}] \approx \sum_{k=0}^{n-1}\gamma^{2k}\sigma^2$ (with perfect critic assumption):

**(a) TD(0) ($n=1$):**
$$\mathrm{Var} \approx \sigma^2 = 1.$$

**(b) 10-step TD ($n=10$):**
$$\mathrm{Var} \approx \sigma^2 \sum_{k=0}^{9}0.99^{2k} = \sum_{k=0}^9 0.9801^k \approx \frac{1-0.9801^{10}}{1-0.9801} \approx \frac{0.182}{0.0199} \approx 9.1.$$

**(c) Monte Carlo ($n=T=100$):**
$$\mathrm{Var} \approx \sigma^2 \sum_{k=0}^{99}0.99^{2k} \approx \frac{1}{1-0.99^2} = \frac{1}{0.0199} \approx 50.$$

**Bias:** As $n$ increases from 1 to $T$:
- Bias decreases (TD(0) has high bias from bootstrapping an imperfect $V$; MC has zero bias).
- Variance increases (more random rewards accumulated).

The bias–variance tradeoff is controlled by $n$ (or equivalently $\lambda$ in TD($\lambda$)): small $n$ → high bias, low variance; large $n$ → low bias, high variance.
