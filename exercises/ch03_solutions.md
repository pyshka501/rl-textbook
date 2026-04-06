# Chapter 3: Markov Decision Processes and the Bellman Equation — Solutions

## Exercise 3.1
**Statement:** Give an example where a raw sensor observation is not Markov but $k$ frames are. Explain why.

**Solution:**

**Example:** A Pong game pixel frame.

A single pixel frame does not capture ball velocity — given only the current frame, the agent cannot predict where the ball will be next (it could be moving in any direction). The single observation violates the Markov property: $P(S_{t+1} \mid S_t = \text{frame}_t) \neq P(S_{t+1} \mid \text{all history})$.

**Augmented state:** Stack the last $k=4$ frames. Now the "state" includes both ball position and the change in position across frames, which encodes velocity. The next frame is fully determined (in a deterministic game) by the current 4-frame stack, so the Markov property holds:
$$P(S_{t+1} \mid S_t^{(k)}) = P(S_{t+1} \mid S_t^{(k)}, S_{t-1}^{(k)}, \ldots)$$
because $S_t^{(k)}$ already subsumes all relevant history.

---

## Exercise 3.2
**Statement:** Continuing task with constant reward $c > 0$ and $\gamma \in [0,1)$. (a) Compute $G_t$. (b) Show $G_t = c/(1-\gamma)$. (c) Why is $\gamma < 1$ essential? (d) What happens when $\gamma = 1$?

**Solution:**

**(a)–(b):**
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} = \sum_{k=0}^{\infty} \gamma^k c = c \sum_{k=0}^{\infty} \gamma^k = \frac{c}{1-\gamma}. \quad \blacksquare$$

**(c)** The geometric series $\sum_{k=0}^\infty \gamma^k$ converges if and only if $|\gamma| < 1$. If $\gamma \ge 1$, the series diverges to $+\infty$ for $c > 0$, making the return infinite and hence the value function undefined.

**(d)** When $\gamma = 1$: $G_t = \sum_{k=0}^\infty c = +\infty$. The return is infinite for any $c > 0$, so value functions cannot be defined for continuing tasks. (Finite-horizon formulations or average-reward criteria are needed instead.)

---

## Exercise 3.3
**Statement:** Verify $r(s,a) = \sum_{s'} p(s'|s,a)\, r(s,a,s')$.

**Solution:**

By definition, $r(s,a,s') = \mathbb{E}[R_{t+1} \mid S_t=s, A_t=a, S_{t+1}=s']$.

The two-argument expected reward is defined as:
$$r(s,a) = \mathbb{E}[R_{t+1} \mid S_t=s, A_t=a] = \sum_{r} r \sum_{s'} p(s',r \mid s,a).$$

Using the law of total expectation, conditioning on $S_{t+1}$:
$$r(s,a) = \sum_{s'} p(s'|s,a) \, \mathbb{E}[R_{t+1} \mid S_t=s, A_t=a, S_{t+1}=s'] = \sum_{s'} p(s'|s,a)\, r(s,a,s'). \quad \blacksquare$$

---

## Exercise 3.4
**Statement:** In the Student MDP, suppose in state $s_3$ the student studies more w.p. 0.5 and passes w.p. 0.5 with $\gamma = 0.9$. (a) Bellman equation for $V_\pi(s_3)$. (b) Solve. (c) Compare with $V_*(s_3) = 10$.

**Solution:**

**(a)** Let $s_3$ be the state, with actions "study more" (leading back to $s_3$ with reward $r_s$) and "pass" (terminal, reward $r_p = 10$, say). Under the mixed policy:
$$V_\pi(s_3) = 0.5[r_s + \gamma V_\pi(s_3)] + 0.5[r_p + \gamma \cdot 0]$$
$$= 0.5[0 + 0.9 \, V_\pi(s_3)] + 0.5[10].$$

**(b)** Solving for $V_\pi(s_3)$ (assuming "study more" gives reward 0 and "pass" gives reward 10):
$$V_\pi(s_3) = 0.45\, V_\pi(s_3) + 5$$
$$V_\pi(s_3)(1 - 0.45) = 5$$
$$V_\pi(s_3) = \frac{5}{0.55} \approx 9.09.$$

**(c)** The optimal policy always passes: $V_*(s_3) = 10 > 9.09 = V_\pi(s_3)$. The mixed policy is sub-optimal because the expected future reward under "study more" is lower than passing immediately (assuming the studying loop has zero reward), yet the policy wastes probability on it.

---

## Exercise 3.5
**Statement:** Prove the Bellman equation for $Q_\pi$ from the definition.

**Solution:**

Starting from the definition and the recursive return:
$$Q_\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t=s, A_t=a]$$
$$= \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t=s, A_t=a] \quad \text{(linearity of expectation)}$$
$$= \mathbb{E}[R_{t+1} \mid S_t=s, A_t=a] + \gamma \mathbb{E}_\pi[G_{t+1} \mid S_t=s, A_t=a].$$

For the second term, condition on $(S_{t+1}, A_{t+1})$:
$$\mathbb{E}_\pi[G_{t+1} \mid S_t=s, A_t=a] = \sum_{s'} p(s'|s,a) \sum_{a'} \pi(a'|s') \, \mathbb{E}_\pi[G_{t+1}|S_{t+1}=s', A_{t+1}=a']$$
$$= \sum_{s'} p(s'|s,a) \sum_{a'} \pi(a'|s') Q_\pi(s',a').$$

Also, $\mathbb{E}[R_{t+1}|S_t=s,A_t=a] = \sum_{s'} p(s'|s,a) r(s,a,s')$. Combining:
$$Q_\pi(s,a) = \sum_{s'} p(s'|s,a)\left[r(s,a,s') + \gamma \sum_{a'}\pi(a'|s') Q_\pi(s',a')\right]. \quad \blacksquare$$

---

## Exercise 3.6
**Statement:** Two-state MDP with given $P_\pi$, $\mathbf{r}_\pi$, $\gamma=0.8$. (a) Matrix Bellman equation. (b) Compute $I - \gamma P_\pi$. (c) Solve for $\mathbf{v}_\pi$.

**Solution:**

**(a)** The matrix Bellman equation is:
$$\mathbf{v}_\pi = \mathbf{r}_\pi + \gamma P_\pi \mathbf{v}_\pi = (I - \gamma P_\pi)^{-1} \mathbf{r}_\pi.$$

**(b)**
$$I - 0.8 P_\pi = \begin{pmatrix}1&0\\0&1\end{pmatrix} - 0.8\begin{pmatrix}0.3&0.7\\0.6&0.4\end{pmatrix} = \begin{pmatrix}1-0.24 & -0.56\\ -0.48 & 1-0.32\end{pmatrix} = \begin{pmatrix}0.76 & -0.56\\ -0.48 & 0.68\end{pmatrix}.$$

**(c)** The determinant is $\det = 0.76 \times 0.68 - (-0.56)(-0.48) = 0.5168 - 0.2688 = 0.248$.

$$\mathbf{v}_\pi = \frac{1}{0.248}\begin{pmatrix}0.68 & 0.56\\ 0.48 & 0.76\end{pmatrix}\begin{pmatrix}2\\5\end{pmatrix} = \frac{1}{0.248}\begin{pmatrix}1.36+2.80\\0.96+3.80\end{pmatrix} = \frac{1}{0.248}\begin{pmatrix}4.16\\4.76\end{pmatrix} \approx \begin{pmatrix}16.77\\19.19\end{pmatrix}.$$

---

## Exercise 3.7
**Statement:** Complete the contraction proof for the optimality operator $\mathcal{T}_*$.

**Solution:**

The optimality Bellman operator is $(\mathcal{T}_* v)(s) = \max_a \left[ r(s,a) + \gamma \sum_{s'} p(s'|s,a) v(s') \right]$.

For any two value functions $u, v$ and any state $s$:
$$(\mathcal{T}_* u)(s) - (\mathcal{T}_* v)(s) = \max_a [r(s,a) + \gamma P_a u(s)] - \max_a [r(s,a) + \gamma P_a v(s)].$$

Using the identity $|\max_a f(a) - \max_a g(a)| \le \max_a |f(a) - g(a)|$:
$$|(\mathcal{T}_* u)(s) - (\mathcal{T}_* v)(s)| \le \max_a \gamma \left|\sum_{s'} p(s'|s,a)(u(s') - v(s'))\right|$$
$$\le \gamma \max_a \sum_{s'} p(s'|s,a) |u(s') - v(s')| \le \gamma \|u-v\|_\infty.$$

Taking the sup over $s$: $\|\mathcal{T}_* u - \mathcal{T}_* v\|_\infty \le \gamma \|u-v\|_\infty$. Since $\gamma < 1$, $\mathcal{T}_*$ is a $\gamma$-contraction. $\blacksquare$

---

## Exercise 3.8
**Statement:** Starting from $V^{(0)} \equiv 0$, bound $\|V^{(k)} - V_*\|_\infty$ for $\gamma=0.9$, $R_{\max}=1$. How many iterations to achieve error $\le 0.01$?

**Solution:**

**(a)** By the contraction property applied $k$ times:
$$\|V^{(k)} - V_*\|_\infty \le \gamma^k \|V^{(0)} - V_*\|_\infty \le \gamma^k \frac{R_{\max}}{1-\gamma}.$$

For $\gamma=0.9$, $R_{\max}=1$: $\|V^{(k)} - V_*\|_\infty \le 0.9^k \cdot 10$.

**(b)** We need $10 \cdot 0.9^k \le 0.01$, i.e., $0.9^k \le 0.001$.
$$k \ge \frac{\ln 0.001}{\ln 0.9} = \frac{-6.908}{-0.1054} \approx 65.5.$$

So $k = 66$ iterations suffice.

---

## Exercise 3.9
**Statement:** Give an MDP with infinitely many optimal policies but a unique optimal value function.

**Solution:**

**Two-state, two-action MDP:** States $\{s_1, s_2\}$ (terminal). In state $s_1$, two actions $a_1, a_2$ both give the same reward $r=5$ and transition to the terminal state $s_2$. No further rewards.

- $V_*(s_1) = 5$ (unique: both actions yield the same expected return)
- $V_*(s_2) = 0$ (terminal)

**Infinitely many optimal policies:** Any stochastic policy $\pi(a_1|s_1) = p$, $\pi(a_2|s_1) = 1-p$ for any $p \in [0,1]$ achieves $Q_\pi(s_1, a_1) = Q_\pi(s_1, a_2) = 5 = V_*(s_1)$. All are optimal. $\blacksquare$

---

## Exercise 3.10
**Statement:** 1×3 grid, no position observation (only wall-bump signal). (a) Why not Markov. (b) Augmented state. (c) Optimal policy.

**Solution:**

**(a)** The observation (bump or no bump) is the same whether the agent is in cell 1 or cell 2 after not bumping. From a single observation $o_t \in \{\text{bump}, \text{no bump}\}$, the agent cannot distinguish its position, so $P(S_{t+1}|O_t) \neq P(S_{t+1}|O_t, O_{t-1}, \ldots)$: the observation alone is not a sufficient statistic for the next state.

**(b)** An augmented state can be the full observation history $h_t = (o_0, a_0, o_1, a_1, \ldots, o_t)$. Since the grid is deterministic, after at most 3 steps the agent can exactly infer its position from the history (bump patterns uniquely identify location over time). This history satisfies the Markov property.

**(c)** Optimal policy in terms of history:
- Start at position unknown. Move **right**.
- If bump (hit right wall at cell 3=goal — not possible, goal is reached): done.
- If reach cell 3: goal achieved.
- If bump on right (cell 3 → goal): collect reward.
- More precisely: from cell 1 → right → cell 2 (no bump) → right → cell 3 (goal). From cell 2 → right → cell 3. 
- If starting at cell 1: go right twice to reach cell 3.
- Since initial position is unknown but always start at cell 1 (or we can track steps), the policy "always go right until goal" is optimal.
