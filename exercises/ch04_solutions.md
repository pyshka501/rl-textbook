# Chapter 4: Dynamic Programming — Solutions

## Exercise 4.1
**Statement:** (a) Show $\|v_k - v^*\|_\infty \le \gamma^k R_{\max}/(1-\gamma)$. (b) For $\gamma=0.99$, $R_{\max}=1$, iterations to reach error $\le 10^{-3}$.

**Solution:**

**(a)** Since $v_0 \equiv 0$ and $\|v^*\|_\infty \le R_{\max}/(1-\gamma)$:
$$\|v_0 - v^*\|_\infty = \|v^*\|_\infty \le \frac{R_{\max}}{1-\gamma}.$$

The contraction property gives $\|v_{k+1} - v^*\|_\infty = \|T^* v_k - T^* v^*\|_\infty \le \gamma\|v_k - v^*\|_\infty$.  
By induction: $\|v_k - v^*\|_\infty \le \gamma^k \|v_0 - v^*\|_\infty \le \gamma^k \frac{R_{\max}}{1-\gamma}$. $\blacksquare$

**(b)** With $\gamma=0.99$, $R_{\max}=1$: $\|v_k-v^*\|_\infty \le 0.99^k \cdot 100 \le 10^{-3}$.
$$0.99^k \le 10^{-5} \implies k \ge \frac{-5\ln 10}{\ln 0.99} = \frac{11.513}{0.01005} \approx 1145.$$

So approximately **1145 iterations** are needed.

---

## Exercise 4.2
**Statement:** Generalise policy improvement theorem to stochastic policies $\pi'$.

**Solution:**

**Claim:** If $\sum_a \pi'(a|s) q^\pi(s,a) \ge v^\pi(s)$ for all $s$, then $v^{\pi'} \ge v^\pi$ pointwise.

**Proof:**
$$v^\pi(s) \le \sum_a \pi'(a|s) q^\pi(s,a) = \sum_a \pi'(a|s)\sum_{s'} p(s'|s,a)[r(s,a,s') + \gamma v^\pi(s')]$$
$$= (T^{\pi'} v^\pi)(s).$$

So $v^\pi \le T^{\pi'} v^\pi$. Since $T^{\pi'}$ is a monotone operator:
$$v^\pi \le T^{\pi'} v^\pi \le (T^{\pi'})^2 v^\pi \le \cdots \le \lim_{k\to\infty} (T^{\pi'})^k v^\pi = v^{\pi'},$$
where the last equality uses the fact that $v^{\pi'}$ is the unique fixed point of $T^{\pi'}$ and $(T^{\pi'})^k v^\pi$ converges to it (contraction). $\blacksquare$

---

## Exercise 4.3
**Statement:** After value iteration with $\|V - v^*\|_\infty \le \varepsilon$, bound $v^*(s) - v^\pi(s)$.

**Solution:**

Let $\pi$ be the greedy policy with respect to $V$. Then for any state $s$:
$$v^\pi(s) = (T^\pi v^\pi)(s) = \sum_a \pi(a|s)\left[r(s,a) + \gamma \sum_{s'} p(s'|s,a)v^\pi(s')\right].$$

Since $\pi$ is greedy w.r.t. $V$, $(T^\pi V)(s) = (T^* V)(s) \ge v^*(s) - 2\varepsilon$ (because $\|V-v^*\|_\infty \le \varepsilon$ and the contraction is applied twice):

More precisely:
$$v^*(s) \le V(s) + \varepsilon \le (T^\pi V)(s) + \varepsilon \le (T^\pi v^\pi)(s) + \varepsilon + \gamma\varepsilon = v^\pi(s) + \varepsilon(1+\gamma).$$

The tighter bound via a two-application argument gives:
$$v^*(s) - v^\pi(s) \le \frac{2\gamma\varepsilon}{1-\gamma}. \quad \blacksquare$$

---

## Exercise 4.4
**Statement:** Write out Bellman equations for corner and edge-midpoint states of 4×4 gridworld.

**Solution:**

In the 4×4 gridworld with absorbing corners $s_1$ (top-left) and $s_{16}$ (bottom-right), reward $-1$ per step, random policy ($\pi(a|s)=0.25$), $\gamma=1$:

For a corner state (e.g., top-left, 2 valid neighbours → both edges):
- State $s_2$ (top row, non-corner): has 3 valid transitions:
$$V(s_2) = -1 + 0.25[V(s_1) + V(s_3) + V(s_2) + V(s_6)]/...$$

More explicitly, for a typical interior state with 4 neighbours $\{n_1, n_2, n_3, n_4\}$:
$$V(s) = -1 + 0.25 \sum_{i=1}^4 V(n_i)$$

For corner state $s_3$ (top row, position 3 with neighbours to left, right, down, and wall):
- Wall moves keep the agent in place, so:
$$V(s_3) = -1 + 0.25[V(s_2) + V(s_4) + V(s_3) + V(s_7)]$$
where the wall move (up) contributes $V(s_3)$ itself.

The terminal states $s_1 = s_{16} = 0$. The converged values (from policy evaluation) satisfy these equations by the Bellman consistency condition.

---

## Exercise 4.5
**Statement:** 5 states, 3 actions: how many deterministic policies? Worst-case iterations? Compare with exhaustive evaluation.

**Solution:**

**Number of deterministic policies:** $|\mathcal{A}|^{|\mathcal{S}|} = 3^5 = 243$.

**Policy iteration bound:** Each outer iteration takes $O(|\mathcal{S}|^2|\mathcal{A}|) = O(25 \cdot 3) = O(75)$ work. Since each policy is visited at most once and there are at most 243 policies:
$$\text{Total work} \le 243 \times 75 = 18{,}225 \text{ operations}.$$

**Exhaustive evaluation:** Evaluate all 243 policies, each requiring $O(|\mathcal{S}|^2 / (1-\gamma))$ work for policy evaluation ≈ $O(25)$ per evaluation iteration × many sweeps:
$$243 \times O(|\mathcal{S}|^2) = 243 \times 25 \approx 6{,}075 \text{ per evaluation iteration}.$$

Policy iteration is generally much more efficient because it converges quickly and each policy is better than the last — it typically terminates in far fewer than 243 iterations.

---

## Exercise 4.6
**Statement:** Explain why in-place synchronous sweeps can converge faster.

**Solution:**

In the **two-array synchronous** version, new values $v_{k+1}(s)$ are computed using only old values $v_k(s')$. Information from updated states cannot propagate within the same sweep.

In the **in-place single-array** version, when $v(s)$ is updated, subsequent updates in the same sweep can immediately use the new (improved) estimate. For example, in a 4×4 gridworld with left-to-right, top-to-bottom updates:

- State $s = (1,1)$ is updated first using its neighbours.
- When $s = (1,2)$ is updated next, it uses the already-updated value of $s=(1,1)$.
- The improved value "propagates" rightward across the entire row in a **single sweep**.

In the two-array version, this propagation requires a full additional sweep. Thus in-place updates can reduce the number of sweeps to convergence, particularly for chains of states where values need to propagate along a single direction.

---

## Exercise 4.7
**Statement:** Show (a) $m=\infty$ → standard policy iteration, (b) $m=1$ → value iteration, (c) non-decreasing convergence for all finite $m$.

**Solution:**

**(a) $m=\infty$:** Performing infinite sweeps of $T^\pi$ until convergence is exactly the definition of full policy evaluation, which is the evaluation step of standard policy iteration. $\blacksquare$

**(b) $m=1$:** After one sweep of $T^\pi$ (evaluating the current policy once), we improve to a greedy policy. But one sweep of $T^\pi$ followed by greedification is:
$$v_{k+1}(s) = \max_a \left[r(s,a) + \gamma \sum_{s'} p(s'|s,a) v_k(s')\right] = (T^* v_k)(s),$$
which is exactly the value iteration update. $\blacksquare$

**(c)** By the policy improvement theorem: after any $m$ evaluation steps $v_k^{(m)} \ge v_k$ pointwise (the value has improved), and greedification ensures $v_{k+1} \ge v_k^{(m)} \ge v_k$. So the sequence is non-decreasing. Since $v^*$ is an upper bound and the sequence is bounded and monotone, it converges. Because the greedy policy improves whenever $v \neq v^*$, the fixed point must be $v^*$. $\blacksquare$

---

## Exercise 4.8
**Statement:** Show $v^*$ is the component-wise minimum of all super-solutions $v \ge T^* v$. Formulate as LP.

**Solution:**

**$v^*$ is a super-solution:** Since $v^* = T^* v^*$, it satisfies $v^*(s) = (T^* v^*)(s) \ge (T^* v^*)(s)$ with equality — so it is a super-solution (and sub-solution).

**$v^*$ is the minimum super-solution:** Let $v \ge T^* v$. Then $v \ge T^* v \ge (T^*)^2 v \ge \ldots \ge \lim_{k\to\infty} (T^*)^k v = v^*$, where the last equality holds because value iteration converges to $v^*$ from any initial $v$. Wait — value iteration from above: since $v \ge T^* v$, the sequence is non-increasing and bounded below by $v^*$, converging to $v^*$. Hence $v \ge v^*$ for any super-solution. $\blacksquare$

**LP formulation:** Minimise $\sum_s \mu(s) v(s)$ (where $\mu(s)>0$ are given weights) subject to:
$$v(s) \ge r(s,a) + \gamma \sum_{s'} p(s'|s,a) v(s'), \quad \forall s \in \mathcal{S}, a \in \mathcal{A}.$$

The optimal solution is $v = v^*$.

---

## Exercise 4.9
**Statement:** Modify Jack's Car Rental with a free first move, $\$4$ parking fee, max 25 cars. Identify changes to transitions vs. reward.

**Solution:**

**State space:** $\{0,\ldots,25\}^2$ instead of $\{0,\ldots,20\}^2$. The transition probabilities (Poisson rental/return processes) change only in their upper truncation.

**Changes affecting transitions:**
- The maximum capacity is now 25 (not 20), so the Poisson truncation point changes. The dynamics $p(n'_i | n_i, a)$ for each location are otherwise the same structure but with higher ceiling.
- No change to the Poisson parameters for rentals/returns.

**Changes affecting only the reward:**
- **Free first car move:** The cost of moving $|a|$ cars is $\$2 \max(0, |a|-1)$ instead of $\$2|a|$.
- **Overnight parking fee:** Add $-\$4 \cdot \mathbb{1}[n_1 > 10] - \$4 \cdot \mathbb{1}[n_2 > 10]$ to the overnight reward.

These last two modifications affect only the expected reward $r(s,a)$, not the transition kernel $p(s'|s,a)$.

---

## Exercise 4.10
**Statement:** Why is on/off-policy distinction moot in model-based DP? What goes wrong in model-free off-policy evaluation?

**Solution:**

**Why moot in model-based DP:**  
Dynamic programming uses the model $p(s'|s,a)$ and $r(s,a)$ directly — it does not rely on samples. The policy evaluation sweep $v_{k+1}(s) = \sum_a \pi(a|s)\sum_{s'} p(s'|s,a)[r(s,a,s') + \gamma v_k(s')]$ queries the model for all $(s,a,s')$ triples regardless of which policy generated any data. There is no behaviour policy, no importance weighting — the model provides full coverage. $\blacksquare$

**Model-free off-policy evaluation:**  
In the model-free setting, updates use sampled transitions $(S_t, A_t, R_{t+1}, S_{t+1})$ generated by a behaviour policy $\mu \neq \pi$. The TD update for $\pi$ using samples from $\mu$ introduces bias because the distribution of $(S_t, A_t)$ under $\mu$ may not match the on-policy distribution under $\pi$.

**Required condition:** Off-policy TD converges if the behaviour policy $\mu$ satisfies **coverage**: $\mu(a|s) > 0$ whenever $\pi(a|s) > 0$ for all $(s,a)$. Without coverage, some transitions required for the evaluation of $\pi$ are never observed, making consistent estimation impossible.
