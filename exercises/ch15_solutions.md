# Chapter 15: Reasoning, Tool Use, and Reinforcement Finetuning — Solutions

## Exercise 15.1
**Statement:** Show the RLVR policy gradient equals the REINFORCE gradient, using the log-derivative trick.

**Solution:**

The RLVR objective (with $\beta=0$):
$$J_{\mathrm{RLVR}}(\theta) = \mathbb{E}_{x\sim\mathcal{D}}\mathbb{E}_{y\sim\pi_\theta(\cdot|x)}\left[\mathbf{1}\{\mathrm{Verify}(y,a^*)\}\right].$$

Applying the log-derivative (REINFORCE) trick:
$$\nabla_\theta J_{\mathrm{RLVR}}(\theta) = \nabla_\theta\sum_y \pi_\theta(y|x)\mathbf{1}\{\mathrm{Verify}(y,a^*)\}$$
$$= \sum_y \nabla_\theta\pi_\theta(y|x)\cdot\mathbf{1}\{\mathrm{Verify}(y,a^*)\}$$
$$= \sum_y \pi_\theta(y|x)\nabla_\theta\log\pi_\theta(y|x)\cdot\mathbf{1}\{\mathrm{Verify}(y,a^*)\}$$
$$= \mathbb{E}_{y\sim\pi_\theta(\cdot|x)}\left[\mathbf{1}\{\mathrm{Verify}(y,a^*)\}\nabla_\theta\log\pi_\theta(y|x)\right]. \quad \blacksquare$$

**Connection to REINFORCE:** This is exactly the REINFORCE gradient (Chapter 8) with reward $R(y) = \mathbf{1}\{\mathrm{Verify}(y,a^*)\} \in \{0,1\}$ — a binary verifiable reward. The gradient pushes up the log-probability of correct completions and (implicitly, after normalisation) pushes down incorrect ones.

---

## Exercise 15.2
**Statement:** Prove majority vote bound using Hoeffding/Chernoff bounds.

**Solution:**

Let $X_1,\ldots,X_k \sim \mathrm{Bernoulli}(p)$ i.i.d. with $p > 1/2$. Let $S_k = \sum_i X_i$.

We want to bound $P(S_k < k/2)$ (majority fails).

**Hoeffding's inequality:** For bounded i.i.d. variables $X_i \in [0,1]$ with mean $p$:
$$P(S_k/k - p \le -t) \le \exp(-2kt^2).$$

Set $t = p - 1/2 > 0$:
$$P(S_k/k < 1/2) \le \exp(-2k(p-1/2)^2).$$

**Chernoff / KL bound (tighter):** By the Chernoff bound:
$$P(S_k < k/2) \le \exp(-k \cdot D_{\mathrm{KL}}(1/2 \| p)),$$
where $D_{\mathrm{KL}}(1/2\|p) = \frac{1}{2}\log\frac{1/2}{p} + \frac{1}{2}\log\frac{1/2}{1-p} = \log\frac{1}{2\sqrt{p(1-p)}}$.

Since $p > 1/2$: $\sqrt{p(1-p)} < 1/2$, so $D_{\mathrm{KL}}(1/2\|p) > 0$, and the probability decays exponentially in $k$. $\blacksquare$

**Interpretation:** As $k$ grows, the majority vote error probability decreases exponentially. The convergence rate depends on how much $p$ exceeds $1/2$: larger gaps → faster convergence.

---

## Exercise 15.3
**Statement:** GRPO advantages for $r=(1,1,0,0)$ vs $r=(1,0,0,0)$.

**Solution:**

**Case 1: $r=(1,1,0,0)$.**

$\mu = 0.5$, $\sigma = 0.5$:
$$\hat{A}_1=\hat{A}_2 = (1-0.5)/0.5 = 1.0, \quad \hat{A}_3=\hat{A}_4 = (0-0.5)/0.5 = -1.0.$$

**Case 2: $r=(1,0,0,0)$.**

$\mu = 0.25$, $\sigma = \sqrt{0.1875} \approx 0.433$:
$$\hat{A}_1 = (1-0.25)/0.433 \approx +1.732, \quad \hat{A}_2=\hat{A}_3=\hat{A}_4 \approx (0-0.25)/0.433 \approx -0.577.$$

**Comparison:**

In Case 2, the single correct completion receives advantage $+1.732$ vs $+1.0$ in Case 1. The gradient signal for the correct completion is **stronger** in Case 2. Intuitively: when only 1 out of 4 completions is correct, it stands out as much more exceptional (higher normalised advantage) than when 2 out of 4 are correct.

This is analogous to exploration in sparse reward settings: rare successes provide stronger learning signal.

---

## Exercise 15.4
**Statement:** (From ch15 exercises — typically covers inference-time compute scaling, best-of-N properties.)

**Solution:**

**Best-of-$N$ as test-time compute:** Given a model with per-sample accuracy $p$, generating $N$ samples and selecting the best (assuming a verifier):
$$P_N = 1-(1-p)^N \approx Np \text{ for small } p.$$

**Key properties:**
1. $P_N$ is monotonically increasing in $N$.
2. For large $N$: $P_N \to 1$ exponentially fast as $N$ grows.
3. The compute cost scales linearly in $N$ (generate $N$ responses).

**Comparison to training-time compute:** Doubling compute during training via dataset scaling (Chinchilla) improves $p$ by a polynomial factor; doubling $N$ (test time) improves $1-(1-p)^N$ by $(1-(1-p)^{2N}) - (1-(1-p)^N) = (1-p)^N(1-(1-p)^N)$ — exponential improvement for large $N$.

For very low base rates ($p \ll 1$), test-time scaling can be dramatically more efficient than training-time scaling.

---

## Exercise 15.5
**Statement:** (From ch15 — typically covers reward shaping for reasoning chains.)

**Solution:**

**Length vs quality in chain-of-thought rewards:**

Reward model $r_\psi$ assigns a single score to the full response. For long chains of thought:
- **Outcome reward:** $r = \mathbf{1}[\text{correct final answer}]$. Simple, but provides no signal for intermediate steps.
- **Process reward:** $r^{(t)} = \mathbf{1}[\text{step } t \text{ is correct}]$. Dense but requires step-level annotations.

**Potential-based shaping for reasoning:** Define $\Phi(s_t) = $ "correctness probability of current partial solution" estimated by a PRM. Then the shaped reward $F(s_t, a_t, s_{t+1}) = \Phi(s_{t+1}) - \Phi(s_t)$ provides dense learning signal without changing the optimal solution.

**Empirical finding:** Models trained with process rewards learn more reliable reasoning patterns and are less likely to shortcut to correct answers via incorrect chains.

---

## Exercise 15.6
**Statement:** (From ch15 — tool-calling MDP formulation.)

**Solution:**

**Tool-calling as an MDP:**
- **State:** $(x, y_{1:t-1}, o_{1:k-1})$ where $x$ is the user query, $y_{1:t-1}$ are generated tokens, $o_{1:k-1}$ are tool outputs received so far.
- **Action:** Next token $a_t = y_t$ (including special tokens like `<tool_call>`, `<end_call>`).
- **Transition:** If $y_t$ is a regular token, append to the current context. If $y_t$ completes a tool call JSON, execute the tool and append its output $o_k$ to the state.
- **Reward:** Non-zero only at the terminal step when a final answer is produced; evaluated by a verifier or human.

**Key challenge:** The tool execution is **non-differentiable** — we cannot backpropagate through the tool call. This makes policy gradient methods (REINFORCE, GRPO) natural, since they only require sampling trajectories and observing terminal rewards, not differentiating through execution.

---

## Exercise 15.7
**Statement:** (From ch15 — scaling laws for inference-time compute.)

**Solution:**

**Empirical scaling laws for self-consistency:**

Let $k$ = number of samples, $p$ = per-sample accuracy. Majority vote accuracy:
$$P_{\mathrm{maj}}(k, p) = 1 - F_{\mathrm{Binom}}(k/2; k, p),$$
where $F_\mathrm{Binom}$ is the binomial CDF.

For large $k$: $P_{\mathrm{maj}} \approx 1 - \exp(-2k(p-0.5)^2)$ by the Central Limit Theorem.

**Optimal allocation:** Given a fixed compute budget $C = k \times c_{\mathrm{gen}} + c_{\mathrm{train}}(N)$ (where $N$ is model size), the optimal split balances model quality (higher $p$ per sample) against inference-time sampling ($k$). For tasks where $p > 0.5$ at baseline, even modest $k=32$ provides near-perfect majority vote accuracy.

---

## Exercise 15.8
**Statement:** (From ch15 — RLVR training dynamics.)

**Solution:**

**RLVR training phases (empirically observed):**

1. **Exploration phase** (early training): the base model has moderate accuracy $p_0 \sim 0.3$. GRPO/RLVR provides gradient signal from the ~30% correct completions. The policy gradually shifts towards longer, more structured reasoning.

2. **Exploitation phase** (mid training): accuracy rises rapidly ($p \to 0.7$). The model learns standard solution templates. Reward signal is strong.

3. **Diminishing returns** (late training): accuracy approaches $p \sim 0.85$; remaining errors are on hard problems. Gradient signal is sparse (few wrong completions per group). Further improvement requires larger $G$ (more samples per prompt) or harder training problems.

**Role of chain-of-thought:** Without explicit CoT training, the model may learn to output correct answers without reasoning ("answer hacking"). Enforcing a reasoning format (requiring `<think>...</think>` tags before answers) with a format reward prevents this collapse and produces transferable reasoning skills.

---

## Exercise 15.9
**Statement:** (From ch15 — comparing inference-time strategies.)

**Solution:**

**Comparison of inference-time reasoning strategies:**

| Strategy | Compute | Accuracy | Requires Verifier? |
|----------|---------|----------|---------------------|
| Direct answer | 1× | Baseline $p$ | No |
| Chain-of-thought | 1–3× (longer) | $p + \Delta_{\mathrm{CoT}}$ | No |
| Self-consistency (k=16) | 16× | $1-(1-p)^{\mathrm{maj}}$ | No |
| Best-of-N (k=16) | 16× | $1-(1-p)^{16}$ | Yes (oracle) |
| Tree-of-thought (branching factor $b$, depth $d$) | $b^d\times$ | Higher than majority | Yes (partial) |

**Key insight:** Best-of-N dominates self-consistency (it always picks the correct answer if any sample is correct), but requires an external verifier. Self-consistency is practical for tasks where the most common answer is usually correct (e.g., arithmetic where wrong answers are diverse).

For RLVR-trained models: training with verifiable rewards *internalises* the verifier's role, making the model more accurate per sample and reducing the required $k$ for majority vote to reach high accuracy.
