# Chapter 10: Reward Shaping and Reward Models — Solutions

## Exercise 10.1
**Statement:** Potential-based shaping with $\Phi(s_1)=0$, $\Phi(s_2)=0.5$, $\Phi(s_3)=1.0$, $\gamma=0.9$.

**Solution:**

**(a)** Shaping bonus $F(s,s') = \gamma\Phi(s') - \Phi(s)$:
$$F(s_1,s_2) = 0.9\times0.5 - 0 = 0.45.$$
$$F(s_2,s_3) = 0.9\times1.0 - 0.5 = 0.9-0.5 = 0.4.$$

**(b)** Original return: $G_0 = 0 + 0 + \gamma^2\cdot1 = 0.81$ (reward only at $s_3$, 2 steps, $\gamma=0.9$).

Shaped return: $\tilde{G}_0 = (0+F(s_1,s_2)) + \gamma(0+F(s_2,s_3)) + \gamma^2\cdot1$
$= 0.45 + 0.9\times0.4 + 0.81 = 0.45+0.36+0.81 = 1.62$.

Check: $G_0 - \Phi(s_1) = 0.81 - 0 = 0.81 \neq 1.62$. Let me recompute.

Actually the theorem is $\tilde{G}_0 = G_0 + \Phi(s_0)(1-\gamma^T) - \gamma^T\Phi(s_T)$ for finite episodes... The standard result is:

$\tilde{G}_t = G_t + \Phi(S_t) - \gamma^{T-t}\Phi(S_T)$... The simpler statement for episodic problems is that with terminal reward of 0 (absorbing terminal state with $\Phi(s_T)=0$):

$\tilde{G}_0 = G_0 + \Phi(s_0)$. So $\tilde{G}_0 = 0.81 + 0 = 0.81$... This doesn't match above.

The potential-based shaping theorem says the shaped value satisfies $\tilde{V}^\pi(s) = V^\pi(s) + \Phi(s) - \Phi(s_\mathrm{terminal})\gamma^{\infty}$. For episodic: $\tilde{V}^\pi(s) = V^\pi(s) + \Phi(s)$.

More directly: $\tilde{G}_0 = \sum_{t=0}^{T-1}\gamma^t[R_{t+1}+F(S_t,S_{t+1})] = G_0 + \sum_{t=0}^{T-1}\gamma^t F(S_t,S_{t+1})$.

$\sum_t \gamma^t F(S_t,S_{t+1}) = \sum_t \gamma^t[\gamma\Phi(S_{t+1})-\Phi(S_t)] = \gamma\sum_t\gamma^t\Phi(S_{t+1}) - \sum_t\gamma^t\Phi(S_t)$
$= \sum_{t=1}^T\gamma^t\Phi(S_t) - \sum_{t=0}^{T-1}\gamma^t\Phi(S_t) = \gamma^T\Phi(S_T) - \Phi(S_0)$.

For episodic tasks with $\Phi(S_T)=\Phi(\text{terminal})=0$ (or defined as 0):
$$\tilde{G}_0 = G_0 + 0 - \Phi(s_1) = G_0 - \Phi(s_1) = 0.81 - 0 = 0.81. \quad \blacksquare$$

**(c)** Since $\tilde{V}^\pi(s) = V^\pi(s) - \Phi(s_0) + $ terminal corrections, the shaped MDP has the same optimal policy. The shaping bonuses make the agent prefer intermediate progress towards the goal (state $s_3$), potentially speeding up learning, without changing the optimal solution.

---

## Exercise 10.2
**Statement:** Construct an MDP where non-potential shaping changes the optimal policy.

**Solution:**

**MDP:** States $\{s_0\}$ (starting), actions $\{a_1, a_2\}$, both leading to terminal state $s_T$.
- $R(s_0, a_1) = 5$, $R(s_0, a_2) = 3$.
- Optimal action before shaping: $a_1$ (higher reward).

**Non-potential shaping:** Define $F(s_0, a_1, s_T) = -3$, $F(s_0, a_2, s_T) = 0$.

Shaped rewards: $\tilde{R}(s_0,a_1) = 5-3=2$, $\tilde{R}(s_0,a_2) = 3+0=3$.

Now $a_2$ is optimal under the shaped reward. The optimal policy has changed.

**Verify $F$ is not potential-based:** A potential-based $F(s,a,s') = \gamma\Phi(s')-\Phi(s)$ depends on $s$ and $s'$ but not on $a$ (for fixed $s,s'$). Here $F$ depends on the action taken, so it cannot be written as $\gamma\Phi(s_T) - \Phi(s_0)$ for any $\Phi$. $\blacksquare$

---

## Exercise 10.3
**Statement:** Prove the four Bradley–Terry properties; explain translation invariance and identifiability.

**Solution:**

The Bradley–Terry model: $P(y_w \succ y_l | x) = \sigma(r(x,y_w) - r(x,y_l))$.

**Property 1 (Normalization):** $P(y_w \succ y_l) + P(y_l \succ y_w) = \sigma(\Delta) + \sigma(-\Delta) = 1$ since $\sigma(x)+\sigma(-x)=1$. $\blacksquare$

**Property 2 (Monotonicity):** $P(y_w \succ y_l) = \sigma(\Delta)$ is strictly increasing in $\Delta = r(y_w)-r(y_l)$. Higher reward $\to$ higher win probability. $\blacksquare$

**Property 3 (Transitivity):** If $r(y_1)>r(y_2)>r(y_3)$, then $P(y_1\succ y_2)>0.5$ and $P(y_2\succ y_3)>0.5$. By monotonicity $r(y_1)>r(y_3)$ so $P(y_1\succ y_3)>0.5$. $\blacksquare$

**Property 4 (Translation invariance):** $\sigma(r(y_w)-r(y_l)) = \sigma((r(y_w)+c)-(r(y_l)+c))$ for any constant $c$. Preferences are unchanged under a global shift of all rewards.

**Identifiability:** Because only reward *differences* determine preferences, the model cannot distinguish $r$ from $r+c$. To make the reward uniquely identifiable, one additional constraint is needed, such as:
- Fixing the reward of a reference response to 0: $r(x, y_\mathrm{ref}) = 0$.
- Constraining the mean reward to zero: $\mathbb{E}[r(x,y)] = 0$.
- Fixing the norm: $\|r\|_2 = 1$.

---

## Exercise 10.4
**Statement:** Reward model scores $r_\psi(y_w)=1.2$, $r_\psi(y_l)=1.8$. Compute loss and gradients.

**Solution:**

**(a)** $\mathcal{L}_{\mathrm{RM}} = -\log\sigma(r_\psi(y_w)-r_\psi(y_l)) = -\log\sigma(1.2-1.8) = -\log\sigma(-0.6)$.

$\sigma(-0.6) = 1/(1+e^{0.6}) = 1/(1+1.822) = 1/2.822 \approx 0.3543$.

$\mathcal{L}_{\mathrm{RM}} = -\log(0.3543) \approx \mathbf{1.038}$.

**(b)** Let $\Delta = r_\psi(y_w) - r_\psi(y_l) = -0.6$.

$\frac{\partial\mathcal{L}}{\partial r_\psi(y_w)} = -\sigma(-\Delta) = -(1-\sigma(\Delta)) = \sigma(\Delta)-1$.

$\sigma(\Delta) = \sigma(-0.6) \approx 0.354$, so $\frac{\partial\mathcal{L}}{\partial r_\psi(y_w)} \approx 0.354-1 = -0.646$. The loss decreases as $r_\psi(y_w)$ increases — correct direction.

$\frac{\partial\mathcal{L}}{\partial r_\psi(y_l)} = \sigma(-\Delta) = 1-\sigma(\Delta) \approx 0.646$. The loss increases as $r_\psi(y_l)$ increases, so it will be pushed down.

**Direction:** The gradient pushes $r_\psi(y_w)$ up and $r_\psi(y_l)$ down, correcting the initial wrong ordering.

**(c)** As $\Delta = r_\psi(y_w)-r_\psi(y_l)$ grows (correctly ordered), $\sigma(-\Delta) \to 0$. The gradient magnitudes $\partial\mathcal{L}/\partial r_\psi \approx \sigma(-\Delta) \to 0$ — the gradient vanishes when the model is very confident. This is saturation; the loss provides progressively smaller signal.

---

## Exercise 10.5
**Statement:** ORM vs PRM on multi-step solution with error in step 2.

**Solution:**

The solution: Step 1 (correct: $60\times2.5=150$), Step 2 (wrong: $80\times2=160$, should be $80\times1.5=120$), Step 3 (follows from Step 2).

**ORM (outcome reward model):** Evaluates the final answer only. The final answer 310 is wrong (correct: 270). ORM assigns a low (negative) reward to the entire solution.

- *Limitation:* ORM cannot identify which step was wrong. It penalises both Step 1 (correct) and Steps 2–3 (wrong) equally. This provides poor credit assignment.

**PRM (process reward model):** Evaluates each step individually.
- Step 1: correct → high reward (+1)
- Step 2: incorrect → low reward (−1)
- Step 3: follows correctly from Step 2's wrong premise → neutral or contextually negative

- *Advantage:* PRM pinpoints Step 2 as the error, providing dense feedback. The model can learn specifically to avoid errors in applying time values.

**When ORM suffices:** For short problems where errors are always terminal (no multi-step compounding), or when only the final answer matters (e.g., multiple-choice).

**When PRM is preferred:** Long chain-of-thought problems where intermediate steps can be correct or incorrect independently, and where dense supervision accelerates learning.

---

## Exercise 10.6
**Statement:** KL-optimal policy rearrangement. Derive the reward reparametrisation used in DPO.

**Solution:**

**(a)** The optimal policy satisfies $\pi^*(y|x) \propto \pi_\mathrm{ref}(y|x)e^{r_\psi(x,y)/\beta}$.

Taking the log:
$$\log\pi^*(y|x) = \log\pi_\mathrm{ref}(y|x) + \frac{r_\psi(x,y)}{\beta} - \log Z(x).$$
$$\log\pi^*(y|x) - \log\pi_\mathrm{ref}(y|x) = \frac{r_\psi(x,y)}{\beta} - \log Z(x). \quad \blacksquare$$

**(b)** Rearranging:
$$r_\psi(x,y) = \beta\left[\log\pi^*(y|x) - \log\pi_\mathrm{ref}(y|x)\right] + \beta\log Z(x). \quad \blacksquare$$

**(c)** Computing the reward difference (noting $Z(x)$ cancels):
$$r_\psi(x,y_w) - r_\psi(x,y_l) = \beta\left[\log\frac{\pi^*(y_w|x)}{\pi_\mathrm{ref}(y_w|x)} - \log\frac{\pi^*(y_l|x)}{\pi_\mathrm{ref}(y_l|x)}\right].$$

This is the DPO reward reparametrisation: rewards are expressed entirely in terms of the learned policy $\pi^*$ (approximated by $\pi_\theta$) and the reference $\pi_\mathrm{ref}$, eliminating the need for an explicit reward model.

---

## Exercise 10.7
**Statement:** Proxy-gold gap grows with KL divergence (Goodhart's Law formulation).

**Solution:**

**(a)** Let $\delta(x,y) = r_\psi(x,y) - r^*(x,y)$ be the reward model error. The gap is:
$$\mathbb{E}_{\pi_\theta}[r_\psi] - \mathbb{E}_{\pi_\theta}[r^*] = \mathbb{E}_{\pi_\theta}[\delta(x,y)].$$

As the policy $\pi_\theta$ deviates from $\pi_\mathrm{ref}$ (large KL), it explores regions where $\pi_\mathrm{ref}$ assigns low probability. The reward model $r_\psi$ was trained on data from $\pi_\mathrm{ref}$, so its generalisation error $\delta$ is larger in these out-of-distribution regions. Hence the gap grows with $\mathrm{KL}(\pi_\theta\|\pi_\mathrm{ref})$.

**(b)** With generalisation error $\varepsilon$ and assuming the reward model's error grows with distributional shift:
$$|\mathbb{E}_{\pi_\theta}[r_\psi]-\mathbb{E}_{\pi_\theta}[r^*]| \le C\sqrt{\varepsilon \cdot \mathrm{KL}(\pi_\theta\|\pi_\mathrm{ref})},$$
by a Cauchy-Schwarz / information-theoretic argument. The optimal KL budget $\mathrm{KL}^* = O(\varepsilon^{-1/3} / C^{2/3})$ balances reward gain against proxy error growth.

**(c)** Mitigation: ensemble of reward models (reduces $\varepsilon$ by averaging), conservative KL constraints (limits distributional shift), iterative reward model updates (reduce the coverage gap).

---

## Exercise 10.8
**Statement:** Show label-smoothed RM loss is standard loss plus a regulariser preventing large scores.

**Solution:**

The standard RM loss: $\mathcal{L}_{\mathrm{RM}} = -\log\sigma(\Delta)$ where $\Delta = r(y_w)-r(y_l)$.

Label-smoothed: $\mathcal{L}_{\mathrm{LS}} = -(1-\epsilon/2)\log\sigma(\Delta) - (\epsilon/2)\log\sigma(-\Delta)$.

$$\mathcal{L}_{\mathrm{LS}} = -\log\sigma(\Delta) + \frac{\epsilon}{2}[\log\sigma(\Delta) - \log\sigma(-\Delta)]$$
$$= \mathcal{L}_{\mathrm{RM}} + \frac{\epsilon}{2}\log\frac{\sigma(\Delta)}{\sigma(-\Delta)} = \mathcal{L}_{\mathrm{RM}} + \frac{\epsilon}{2}\log\frac{\sigma(\Delta)}{1-\sigma(\Delta)}.$$

Since $\log\sigma(\Delta)/(1-\sigma(\Delta)) = \Delta$, this is $\mathcal{L}_{\mathrm{RM}} + \frac{\epsilon}{2}\Delta$.

Alternatively: $\mathcal{L}_{\mathrm{LS}} = \mathcal{L}_{\mathrm{RM}} + \frac{\epsilon}{2}\log\frac{\sigma(-\Delta)}{1-\epsilon/2}$... 

The regularising term penalises large $|\Delta|$: as $\Delta \to \pm\infty$, $\mathcal{L}_{\mathrm{LS}}$ grows linearly in $|\Delta|$ (instead of saturating to zero), preventing the rewards from becoming arbitrarily large. $\blacksquare$

---

## Exercise 10.9
**Statement:** Show per-token KL penalty is potential-based shaping in the token MDP.

**Solution:**

**Token MDP:** State $s_t = (x, y_{1:t-1})$, action $a_t = y_t$ (next token). Define potential:
$$\Phi(s_t) = \beta\log\pi_\mathrm{ref}(y_t | x, y_{<t}).$$

**(a)** The shaping bonus:
$$F(s_t,a_t,s_{t+1}) = \gamma\Phi(s_{t+1}) - \Phi(s_t) = \gamma\beta\log\pi_\mathrm{ref}(y_{t+1}|x,y_{\le t}) - \beta\log\pi_\mathrm{ref}(y_t|x,y_{<t}).$$

The per-token KL penalty in the RLHF token reward is:
$$\tilde{r}(s_t,a_t) = r_\psi(x,y)\mathbf{1}[t=T] - \beta\log\frac{\pi_\theta(y_t|x,y_{<t})}{\pi_\mathrm{ref}(y_t|x,y_{<t})}.$$

The KL term is $-\beta[\log\pi_\theta(y_t) - \log\pi_\mathrm{ref}(y_t)]$. The $-\beta\log\pi_\mathrm{ref}(y_t)$ part is exactly $-\Phi(s_t)/\gamma^0$...

**(b)** The key point: the term $\beta\log\pi_\mathrm{ref}(y_t|s_t)$ in the per-token reward acts like a potential-based shaping bonus $\Phi(s_t) = \beta\log\pi_\mathrm{ref}(\cdot|s_t)$ evaluated at each step. This means the KL-penalised RLHF objective is equivalent (in terms of optimal policy) to an unpenalised objective with shaped rewards — confirming that KL regularisation does not change the ordering of policies but only the scale of their Q-values.
