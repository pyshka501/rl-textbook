# Chapter 7: Value Function Approximation and Deep Reinforcement Learning — Solutions

## Exercise 7.1
**Statement:** $n$-dimensional binary state vector, $|\mathcal{A}|=4$. (a) Tabular storage. (b) Linear FA storage ratio. (c) Break-even $n$.

**Solution:**

**(a)** Number of tabular Q-function entries: $2^n \times |\mathcal{A}| = 4 \cdot 2^n$.

**(b)** Linear FA with $d=10n$ features requires storing $d \times |\mathcal{A}| = 40n$ parameters.

Storage ratio: $\frac{40n}{4 \cdot 2^n} = \frac{10n}{2^n}$.

**(c)** Break-even: $\frac{10n}{2^n} < 1 \Leftrightarrow 10n < 2^n$.

Testing: $n=4$: $40 < 16$ (no). $n=5$: $50 < 32$ (no). $n=6$: $60 < 64$ (yes!).

At **$n=6$**, linear FA first requires less storage than the tabular representation.

---

## Exercise 7.2
**Statement:** $|\mathcal{S}|=4$, $d=2$, $\mu=\text{uniform}$, and given $\Phi$. Compute projection matrix and apply it.

**Solution:**

$$\Phi = \begin{pmatrix}1&0\\1&0\\0&1\\0&1\end{pmatrix}, \quad D_\mu = \frac{1}{4}I_4.$$

**(a)** $\Phi$ has full column rank: columns $(1,1,0,0)^\top$ and $(0,0,1,1)^\top$ are linearly independent. ✓

**(b)** $\Phi^\top D_\mu \Phi = \frac{1}{4}\Phi^\top\Phi$:
$$\Phi^\top\Phi = \begin{pmatrix}2&0\\0&2\end{pmatrix} \implies \Phi^\top D_\mu\Phi = \begin{pmatrix}1/2&0\\0&1/2\end{pmatrix} \implies (\Phi^\top D_\mu\Phi)^{-1} = \begin{pmatrix}2&0\\0&2\end{pmatrix}.$$

$$\Pi_\mu = \Phi(\Phi^\top D_\mu\Phi)^{-1}\Phi^\top D_\mu = \frac{1}{4}\begin{pmatrix}1&0\\1&0\\0&1\\0&1\end{pmatrix}\begin{pmatrix}2&0\\0&2\end{pmatrix}\begin{pmatrix}1&1&0&0\\0&0&1&1\end{pmatrix}$$

$$= \frac{1}{2}\begin{pmatrix}1&1&0&0\\1&1&0&0\\0&0&1&1\\0&0&1&1\end{pmatrix}.$$

**(c)** $\Pi_\mu v$ where $v=(3,1,2,4)^\top$:
$$\Pi_\mu v = \frac{1}{2}\begin{pmatrix}1&1&0&0\\1&1&0&0\\0&0&1&1\\0&0&1&1\end{pmatrix}\begin{pmatrix}3\\1\\2\\4\end{pmatrix} = \frac{1}{2}\begin{pmatrix}4\\4\\6\\6\end{pmatrix} = (2,2,3,3)^\top.$$

**(d)** $\|v - \Pi_\mu v\|_\mu^2 = \frac{1}{4}[(3-2)^2+(1-2)^2+(2-3)^2+(4-3)^2] = \frac{4}{4}=1$.

$\|v - f\|_\mu^2$ for $f=(2,2,2,2)^\top$: $\frac{1}{4}[(1)^2+(-1)^2+(0)^2+(2)^2]=\frac{6}{4}=1.5 > 1$. ✓

---

## Exercise 7.3
**Statement:** Semi-gradient vs full gradient. Identify the missing term.

**Solution:**

**(a) Semi-gradient** (treating $y = r + \gamma V_\mathbf{w}(s')$ as constant):
$$-\frac{1}{2}\nabla_\mathbf{w}[y - V_\mathbf{w}(s)]^2 = (y - V_\mathbf{w}(s))\nabla_\mathbf{w} V_\mathbf{w}(s).$$

**(b) Full gradient** (treating $y$ as a function of $\mathbf{w}$):
$$-\frac{1}{2}\nabla_\mathbf{w}[y - V_\mathbf{w}(s)]^2 = (y-V_\mathbf{w}(s))\nabla_\mathbf{w}V_\mathbf{w}(s) - (y-V_\mathbf{w}(s))\gamma\nabla_\mathbf{w}V_\mathbf{w}(s').$$

**(c) The missing term:** $-(y-V_\mathbf{w}(s))\gamma\nabla_\mathbf{w}V_\mathbf{w}(s')$, i.e., the gradient of the bootstrap target with respect to the parameters.

**Condition for equality:** The missing term is zero when $\nabla_\mathbf{w}V_\mathbf{w}(s') = 0$ — e.g., for a terminal state $s'$ where $V_\mathbf{w}(s')=0$ always, or when using a target network with separate (frozen) parameters.

---

## Exercise 7.4
**Statement:** TD fixed-point bound $\|\hat{V}-V^\pi\|_\mu \le \frac{1}{\sqrt{1-\gamma^2}}\|\hat{V}^*-V^\pi\|_\mu$. Compute for $\gamma \in \{0.5, 0.9, 0.99, 0.999\}$.

**Solution:**

The amplification factor is $f(\gamma) = 1/\sqrt{1-\gamma^2}$:

| $\gamma$ | $1-\gamma^2$ | $f(\gamma) = 1/\sqrt{1-\gamma^2}$ |
|----------|-------------|----------------------------------|
| 0.5      | 0.75        | $1/\sqrt{0.75} \approx 1.15$     |
| 0.9      | 0.19        | $1/\sqrt{0.19} \approx 2.29$     |
| 0.99     | 0.0199      | $1/\sqrt{0.0199} \approx 7.09$   |
| 0.999    | 0.001999    | $1/\sqrt{0.001999} \approx 22.4$ |

**(b) Intuition for $\gamma \to 1$:** Bootstrapping compounds errors across long horizons. With $\gamma \approx 1$, each TD update propagates the approximation error at $V_\mathbf{w}(s')$ back to $V_\mathbf{w}(s)$ with almost no discount, allowing small per-step errors to accumulate without decay.

**(c)** The bound is generally not tight. Tightness requires an MDP and feature set where the worst-case correlation between bootstrapping and approximation error is maximised at every state simultaneously — a very special structure.

---

## Exercise 7.5
**Statement:** Identify deadly triad elements and convergence for each algorithm.

**Solution:**

| Algorithm | Bootstrapping | FA | Off-policy | Convergence |
|-----------|:---:|:---:|:---:|:---:|
| (1) MC + linear FA, on-policy | No | Yes | No | **Guaranteed** (2 of 3 missing) |
| (2) Q-learning, tabular, $10^6$ states | Yes | No (tabular) | Yes | **Guaranteed** (tabular Q-learning) |
| (3) Off-policy TD(0) + neural network | Yes | Yes | Yes | **Not guaranteed** (all three present) |
| (4) SARSA + linear FA, on-policy | Yes | Yes | No | **Converges** to TD fixed point |
| (5) DQN on Atari | Yes | Yes | Yes | **Not guaranteed** (but empirically stable with target networks + replay) |

---

## Exercise 7.6
**Statement:** Show DQN loss reduces to Q-learning for tabular representation.

**Solution:**

**Tabular representation with one-hot features:** $Q_\theta(s,a) = \theta_{s,a}$ (one parameter per $(s,a)$).

**DQN loss** with target $y_j = r_j + \gamma \max_{a'} Q_{\theta^-}(s_j', a')$:
$$L(\theta) = \sum_j (y_j - Q_\theta(s_j, a_j))^2.$$

Taking the gradient w.r.t. $\theta_{s,a}$ and updating with $\alpha=1$, no target network ($\theta^- = \theta$):
$$\theta_{s,a} \leftarrow \theta_{s,a} + \sum_{j: (s_j,a_j)=(s,a)} (y_j - Q_\theta(s,a)).$$

For a single sample:
$$Q(s,a) \leftarrow Q(s,a) + (r + \gamma\max_{a'} Q(s',a') - Q(s,a)),$$
which is exactly the Q-learning update with step size $\alpha=1$. $\blacksquare$

**Implication:** DQN is a deep generalisation of Q-learning. The target network ($\theta^-$) and experience replay are engineering choices that stabilise the neural network version but are absent from the tabular setting.

---

## Exercise 7.7
**Statement:** (a) Show maximisation bias. (b) Is $\mathbb{E}[\tilde{Q}(s', \arg\max_a Q(s',a))]$ still biased? (c) Relate to Double DQN.

**Solution:**

**(a)** Let $Q(s,a) = Q^*(s,a) + \varepsilon_{s,a}$ with $\mathbb{E}[\varepsilon_{s,a}]=0$.

$$\mathbb{E}[\max_a Q(s,a)] = \mathbb{E}[\max_a (Q^*(s,a)+\varepsilon_{s,a})] \ge \max_a Q^*(s,a),$$
since $\mathbb{E}[\max_a X_a] \ge \max_a \mathbb{E}[X_a]$ by Jensen's inequality (max is convex). $\blacksquare$

**(b)** With independent $\tilde{Q}(s,a) = Q^*(s,a) + \tilde{\varepsilon}_{s,a}$, $\tilde{\varepsilon}$ independent of $\varepsilon$:

Let $a^*(s') = \arg\max_a Q(s',a)$. Then:
$$\mathbb{E}[\tilde{Q}(s', a^*(s'))] = \mathbb{E}[Q^*(s',a^*(s')) + \tilde{\varepsilon}_{s',a^*(s')}] = \mathbb{E}[Q^*(s',a^*(s'))].$$

Since $\tilde{\varepsilon}$ is independent of the choice of $a^*$, it has zero mean. However, $\mathbb{E}[Q^*(s',a^*(s'))] = \mathbb{E}[\max_a Q^*(s',a) + (Q^*(s',a^*)-Q^*(s',a^*))]$... more carefully: $a^*$ depends on $\varepsilon$ (not $\tilde{\varepsilon}$), so:
$$\mathbb{E}[\tilde{Q}(s',a^*)] = \mathbb{E}[Q^*(s',a^*(s'))] \le \max_a Q^*(s',a).$$

The bias is reduced (the evaluation is unbiased conditional on the action selection) — not necessarily zero since $a^*$ may not be the true optimum. The Double DQN bias is of order $O(\sigma^2)$ rather than $O(\sigma)$.

**(c)** Double DQN separates action selection (using $\theta$) and value estimation (using $\theta^-$):
$$y_j^{\text{DDQN}} = r_j + \gamma Q_{\theta^-}(s_j', \arg\max_{a'} Q_\theta(s_j',a')).$$
This mirrors the theoretical analysis: independent networks for selection and evaluation reduce the maximisation bias from $O(\sigma)$ to $O(\sigma^2)$.

---

## Exercise 7.8
**Statement:** PER IS weights. (a) Equal priority → $w_i=1$. (b) Why IS corrections needed for $\beta=1$. (c) When PER hurts.

**Solution:**

**(a)** Equal priority $p_i = p$: $P(i) = p/\sum_j p = 1/N$. IS weight: $w_i = (N \cdot 1/N)^{-\beta} = 1^{-\beta} = 1$. $\blacksquare$

**(b)** PER samples high-priority (high-TD-error) transitions more often. Without IS correction ($\beta=0$), the gradient updates are biased: we compute $\mathbb{E}_{P}[\nabla L]$ instead of $\mathbb{E}_{\text{uniform}}[\nabla L]$. For convergence to a correct value function we need unbiased gradient estimates (importance sampling ratio $1/P(i) \cdot 1/N$ corrects this when $\beta=1$). With $\beta=0$, the SGD gradient is biased, so the fixed point of the update may not be the correct $Q^\pi$.

**(c)** PER can hurt when:
- The initial TD errors are dominated by noise rather than signal (early training), causing high-variance transitions to be over-replayed.
- The priority metric (TD error) is a poor proxy for learning utility (e.g., in sparse reward settings where all non-terminal transitions have equal TD error).
- The replay buffer is too small and high-priority transitions are replayed so frequently that the model overfits to them (catastrophic forgetting of other transitions).

---

## Exercise 7.9
**Statement:** Fitted Value Iteration. (a) Linear FA → linear least squares. (b) Tabular → exact value iteration convergence.

**Solution:**

**(a)** For linear FA $Q_\mathbf{w}(s,a) = \mathbf{w}^\top \phi(s,a)$:
$$\mathbf{w}_{k+1} = \arg\min_\mathbf{w} \sum_j (\underbrace{r_j + \gamma\max_{a'}Q_{\mathbf{w}_k}(s_j',a')}_{\text{fixed target}} - \mathbf{w}^\top\phi(s_j,a_j))^2.$$

This is a least-squares regression of fixed targets $y_j^{(k)} = r_j + \gamma\max_{a'}\mathbf{w}_k^\top\phi(s_j',a')$ onto features $\phi(s_j,a_j)$. The solution is:
$$\mathbf{w}_{k+1} = (\Phi^\top\Phi)^{-1}\Phi^\top \mathbf{y}^{(k)},$$
a standard linear least-squares problem. $\blacksquare$

**(b)** For tabular FA (one parameter per $(s,a)$), the regression is trivial: the closed-form solution is $Q_{k+1}(s,a) = \frac{1}{|D_{s,a}|}\sum_{j\in D_{s,a}} y_j^{(k)}$, the average of Bellman targets for each $(s,a)$.

When the dataset $\mathcal{D}$ covers all $(s,a,r,s')$ transitions and the MDP is deterministic, this is exact:
$$Q_{k+1}(s,a) = r(s,a) + \gamma\max_{a'} Q_k(s',a') = (T^* Q_k)(s,a).$$

This is exactly the tabular value iteration update, which converges to $Q^*$ by the contraction property of $T^*$. $\blacksquare$
