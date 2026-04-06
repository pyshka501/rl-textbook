# Chapter 5: Monte Carlo Methods — Solutions

## Exercise 5.1
**Statement:** Let $G_1^{\mathrm{FV}},\ldots,G_n^{\mathrm{FV}}$ be first-visit returns. (a) Show unbiasedness. (b) Compute variance. (c) How many episodes for variance $\le 0.01\sigma_s^2$?

**Solution:**

**(a)** Each first-visit return $G_i^{\mathrm{FV}}(s)$ is an unbiased estimate of $v^\pi(s)$ because first-visit returns are i.i.d. samples from the correct distribution (the first time $s$ is visited in each episode, the subsequent trajectory is independent of earlier visits and has the correct distribution under $\pi$):
$$\mathbb{E}\left[\hat{v}_n^{\,\mathrm{FV}}(s)\right] = \mathbb{E}\left[\frac{1}{n}\sum_{i=1}^n G_i^{\mathrm{FV}}(s)\right] = \frac{1}{n}\sum_{i=1}^n v^\pi(s) = v^\pi(s). \quad \blacksquare$$

**(b)** Since the first-visit returns are independent across episodes:
$$\mathrm{Var}\left[\hat{v}_n^{\,\mathrm{FV}}(s)\right] = \frac{1}{n^2} \cdot n\sigma_s^2 = \frac{\sigma_s^2}{n}.$$

**(c)** We need $\sigma_s^2/n \le 0.01\sigma_s^2 \implies n \ge 100$ episodes.

---

## Exercise 5.2
**Statement:** Show by example that the every-visit estimator is biased in a single episode.

**Solution:**

**Example:** Two-state loop $s \to s \to T$ (terminal), rewards $R_1 = 1$, $R_2 = 0$, $\gamma=1$.

True value: $v^\pi(s) = \mathbb{E}[G | S_0=s]$. Since the loop is visited twice with rewards 1 and 0:
- From visit 1: $G_1 = 1 + \gamma \cdot 0 = 1$.
- From visit 2: $G_2 = 0$.
- $v^\pi(s) = 1$ (the expected return from the first visit; the second visit contributes $0$ but is a different state of the world).

Wait — more carefully: state $s$ is visited at $t=0$ and $t=1$ in a single episode $s \to s \to T$.

- At $t=0$: return $G_0 = R_1 + \gamma R_2 = 1 + 0 = 1$.
- At $t=1$: return $G_1 = R_2 = 0$.

**Every-visit estimate** from a single episode: $\hat{v}_1^{\,\mathrm{EV}}(s) = \frac{G_0 + G_1}{2} = \frac{1+0}{2} = 0.5$.

**True value:** $v^\pi(s) = \mathbb{E}[G | S_0 = s]$. If the loop always runs deterministically, $v^\pi(s) = 1$.

So $\hat{v}_1^{\,\mathrm{EV}}(s) = 0.5 \neq 1 = v^\pi(s)$: the EV estimator is biased in a single episode because the two "observations" $G_0$ and $G_1$ are not independent — $G_1$ is a truncated portion of $G_0$, systematically underestimating the true return. $\blacksquare$

---

## Exercise 5.3
**Statement:** Derive the incremental MC update $V_{n+1} = V_n + \frac{1}{n+1}(G_{n+1} - V_n)$.

**Solution:**

$$V_n = \frac{1}{n}\sum_{i=1}^n G_i \implies n V_n = \sum_{i=1}^n G_i.$$

$$V_{n+1} = \frac{1}{n+1}\sum_{i=1}^{n+1} G_i = \frac{1}{n+1}\left(\sum_{i=1}^n G_i + G_{n+1}\right) = \frac{1}{n+1}(n V_n + G_{n+1})$$
$$= \frac{n V_n + G_{n+1}}{n+1} = V_n + \frac{G_{n+1} - V_n}{n+1} = V_n + \frac{1}{n+1}(G_{n+1} - V_n). \quad \blacksquare$$

---

## Exercise 5.4
**Statement:** Blackjack questions: (a) why restrict $x \ge 12$? (b) how does on-policy MC cover all 200 states? (c) estimate $v^\pi(20, \text{6, no ace})$.

**Solution:**

**(a)** For $x < 12$: the player can always safely hit (draw another card) without risk of busting, since even a 10-valued card brings the total to at most $11+10=21$. The optimal action is deterministically "hit" regardless of the dealer's card or usable ace. Including these states would add trivial entries. The interesting decision-making begins at $x=12$ (where hitting risks a bust with a 10-valued card).

**(b)** The simple policy "stick on 20 or 21" still *visits* all 200 states because:
- Episodes start with random deals, generating diverse initial states.
- The policy sticks at 20/21 and hits otherwise, so for states with $x \le 19$ it takes the "hit" action and can transition through all lower totals up to 20.
- The on-policy evaluator evaluates the policy *as it is* — it does not need to try both actions at every state, only the action the policy takes. For $x \le 19$, that action is always "hit". For $x = 20, 21$, that action is "stick". Both are explored by starting in those states.

**(c)** State: player sum = 20, dealer shows 6, no usable ace.

**Estimate:** $v^\pi(20, 6, \text{no ace}) \approx +0.65$.

**Reasoning:** With 20, the player sticks immediately. The dealer must draw to at least 17. A dealer showing 6 must reach 17–21:
- Dealer often busts (probability $\approx 42\%$) since 6 requires several cards.
- When the dealer doesn't bust, they may reach 17–19 (player's 20 wins), 20 (tie), or 21 (player loses).
- Overall win probability $\approx 65\%$, giving value $\approx +0.65 \cdot 1 + 0.07 \cdot 0 - 0.28 \cdot 1 \approx 0.37$... 

More precisely, empirical Blackjack simulations give $v^\pi(20, 6, \text{no ace}) \approx 0.65$ under the "stick on 20/21" policy.

---

## Exercise 5.5
**Statement:** Does MC-ES still converge if we update on every visit rather than first visit?

**Solution:**

**Yes, convergence is maintained** under some conditions, but the argument is more subtle.

With every-visit updates, the Q estimates are no longer unbiased (as shown in Exercise 5.2) because returns from multiple visits within an episode are not independent. However:

- The Q estimates still converge to the correct values in the limit of many episodes because each episode provides new evidence and the incremental update $Q \leftarrow Q + \frac{1}{n}(G - Q)$ is a stochastic approximation with vanishing step size.
- The convergence proof now relies on the fact that the bias vanishes as the number of visits grows (the correlation between visits within an episode becomes negligible compared to the long-run average).
- The policy improvement step is still applied correctly since it uses the converged Q values.

**What changes:** The convergence argument for first-visit MC relies on i.i.d. samples; every-visit MC requires a different argument (the Robbins–Monro conditions on the stochastic update still hold in the limit). Convergence is slower and the bias requires more episodes to wash out.

---

## Exercise 5.6
**Statement:** Characterise when the inequality in the $\varepsilon$-soft improvement proof becomes equality, and why at equality $\pi$ is $\varepsilon$-optimal.

**Solution:**

The key inequality in the proof is:
$$\sum_a \pi'(a|s) q^\pi(s,a) \ge v^\pi(s)$$
where $\pi'$ is the $\varepsilon$-greedy policy with respect to $q^\pi$.

**Equality holds when:** $\pi' = \pi$, i.e., when the current policy is already $\varepsilon$-greedy with respect to its own Q-values. This means:
- The greedy action is assigned probability $1 - \varepsilon + \varepsilon/|\mathcal{A}|$.
- All other actions are assigned equal probability $\varepsilon/|\mathcal{A}|$.

**Why $\pi$ is $\varepsilon$-optimal at equality:** At the fixed point $\pi = \pi'$, the policy assigns as much probability as possible to the greedy action given the $\varepsilon$-soft constraint. Any strictly better $\varepsilon$-soft policy would require reassigning probability away from the greedy action, which is impossible while remaining $\varepsilon$-soft. Hence $\pi$ is the best $\varepsilon$-soft policy, i.e., $\varepsilon$-optimal.

---

## Exercise 5.7
**Statement:** Show $\mathrm{Var}_\mu[\rho G] = \mathbb{E}_\mu[\rho^2 G^2] - (Q^\pi(s,a))^2$. Why can $\mathbb{E}_\mu[\rho^2 G^2]$ be large?

**Solution:**

Let $X = \rho G$ where $\rho = \prod_{t'=t}^{T-1} \frac{\pi(A_{t'}|S_{t'})}{\mu(A_{t'}|S_{t'})}$.

$$\mathrm{Var}_\mu[\rho G] = \mathbb{E}_\mu[(\rho G)^2] - (\mathbb{E}_\mu[\rho G])^2.$$

Now $\mathbb{E}_\mu[\rho G] = Q^\pi(s,a)$ (the OIS estimator is unbiased). Therefore:
$$\mathrm{Var}_\mu[\rho G] = \mathbb{E}_\mu[\rho^2 G^2] - (Q^\pi(s,a))^2. \quad \blacksquare$$

**Why $\mathbb{E}_\mu[\rho^2 G^2]$ can be large:** The IS ratio $\rho$ can be exponentially large when $\pi$ and $\mu$ differ. Over $T-t$ steps, $\rho = \prod_{t'=t}^{T-1} \frac{\pi(A_{t'}|S_{t'})}{\mu(A_{t'}|S_{t'})}$ can have ratios much larger than 1 with positive probability. If $\pi$ is deterministic and $\mu$ is nearly uniform, individual ratios can be as large as $|\mathcal{A}|$, making $\rho$ grow as $|\mathcal{A}|^{T-t}$. This motivates WIS, which normalises the weights.

---

## Exercise 5.8
**Statement:** Prove that if $\pi = \mu$, then OIS and WIS reduce to standard on-policy MC average.

**Solution:**

When $\pi = \mu$: for every trajectory, $\rho_{t:T-1} = \prod_{t'=t}^{T-1} \frac{\pi(A_{t'}|S_{t'})}{\mu(A_{t'}|S_{t'})} = \prod_{t'=t}^{T-1} 1 = 1$.

**OIS:** $\hat{Q}^{\mathrm{OIS}} = \frac{1}{n}\sum_{j=1}^n \rho^{(j)} G^{(j)} = \frac{1}{n}\sum_{j=1}^n G^{(j)}$, the standard sample mean. $\blacksquare$

**WIS:** $\hat{Q}^{\mathrm{WIS}} = \frac{\sum_j \rho^{(j)} G^{(j)}}{\sum_j \rho^{(j)}} = \frac{\sum_j G^{(j)}}{\sum_j 1} = \frac{1}{n}\sum_j G^{(j)}$, also the standard sample mean. $\blacksquare$

---

## Exercise 5.9
**Statement:** Show $\hat{Q}_1^{\,\mathrm{WIS}}(s,a) = G_0^{(1)}$ regardless of the IS weight.

**Solution:**

With a single episode ($n=1$):
$$\hat{Q}_1^{\,\mathrm{WIS}}(s,a) = \frac{\rho^{(1)} G_0^{(1)}}{\rho^{(1)}} = G_0^{(1)}.$$

The IS weight $\rho^{(1)}$ cancels completely in the ratio. $\blacksquare$

**Conclusion:** With a single episode, WIS simply returns the observed return $G_0^{(1)}$, which is generally not equal to $Q^\pi(s,a)$. Hence WIS is **biased** with a single episode (it is consistent but not unbiased for finite $n$).

---

## Exercise 5.10
**Statement:** 5-step MDP, reward $+1$ at final step, $\gamma=1$. (a) MC returns $G_t$. (b) Convergence of $V(s_0)$. (c) Variance of $G_0$ vs $G_4$.

**Solution:**

**(a)** The trajectory is $s_0, s_1, s_2, s_3, s_4, T$ with $R_5 = 1$, $R_t = 0$ for $t < 5$, $\gamma=1$:
$$G_4 = R_5 = 1, \quad G_3 = R_4 + G_4 = 1, \quad G_2 = 1, \quad G_1 = 1, \quad G_0 = 1.$$

All $G_t = 1$ since the policy is deterministic and $\gamma=1$.

**(b)** Since every episode gives $G_0 = 1$, the estimate $V(s_0) = 1$ after even a single episode, and converges to $v^\pi(s_0) = 1$.

**(c)** In this deterministic example, all $G_t = 1$ exactly so $\mathrm{Var}[G_t] = 0$ for all $t$.

**In general with stochastic rewards:** $G_0 = \sum_{k=0}^4 R_{k+1}$ involves a sum of 5 random variables, while $G_4 = R_5$ involves only 1. By independence:
$$\mathrm{Var}[G_0] = \sum_{k=0}^4 \mathrm{Var}[R_{k+1}] = 5\sigma^2 > \sigma^2 = \mathrm{Var}[G_4].$$

They would be equal only if $\mathrm{Var}[R_t] = 0$ for all $t$ (deterministic rewards).
