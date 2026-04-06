# Chapter 11: Language Models and Post-Training — Solutions

## Exercise 11.1
**Statement:** LM assigns probabilities to "The cat sat". (a) Cross-entropy loss. (b) Perplexity. (c) Perplexity of uniform distribution.

**Solution:**

**(a)** Cross-entropy loss over $N=3$ tokens:
$$\mathcal{L} = -\frac{1}{3}\left[\log(0.05) + \log(0.02) + \log(0.08)\right]$$
$$= -\frac{1}{3}\left[-2.996 + (-3.912) + (-2.526)\right] = -\frac{-9.434}{3} = \mathbf{3.145} \text{ nats}.$$

In bits (divide by $\ln 2 \approx 0.693$): $3.145/0.693 \approx 4.54$ bits.

**(b)** Perplexity $= e^{\mathcal{L}} = e^{3.145} \approx \mathbf{23.2}$.

**(c)** For a uniform distribution over $|\mathcal{V}|=50{,}000$ tokens: $p(\text{each token}) = 1/50{,}000$.

$$\mathcal{L}_\mathrm{uniform} = -\log(1/50{,}000) = \log(50{,}000) \approx 10.82 \text{ nats}.$$
$$\text{Perplexity}_\mathrm{uniform} = 50{,}000.$$

A perplexity of 23.2 is much better than 50,000, showing the model has learned a reasonable distribution.

---

## Exercise 11.2
**Statement:** Token MDP state space and storage; why DQN is infeasible; alternative.

**Solution:**

**(a)** Upper bound on $|\mathcal{S}|$: a state is a prefix $(x, y_{1:t})$ of length up to $T=512$ tokens, each from a vocabulary of $|\mathcal{V}|=10^5$.

$$|\mathcal{S}| \le \sum_{t=0}^{512} |\mathcal{V}|^t \le 513 \times (10^5)^{512}.$$

This is astronomically large. Even for pairs $(s,a)$: the number of entries is $|\mathcal{S}| \times |\mathcal{V}|$.

**Memory:** $|\mathcal{S}|\times|\mathcal{V}|\times 4$ bytes is incomprehensibly large (many orders of magnitude beyond all storage on Earth).

**(b)** DQN requires storing and updating Q-values for all $(s,a)$ pairs. With state space $|\mathcal{S}| \approx (10^5)^{512}$ and action space $10^5$, a Q-table is completely infeasible. Even with function approximation, explicitly enumerating all states to select the max-action is intractable.

**(c)** Policy gradient methods (REINFORCE, PPO, GRPO) avoid enumerating Q-values — they directly optimise $\pi_\theta(a|s)$ parameterised by the LM weights, which naturally handles the exponentially large state and action spaces.

---

## Exercise 11.3
**Statement:** Causal mask, equivalence to Markov property, and encoder-only models.

**Solution:**

**(a)** Masked attention weight:
$$\alpha_{ij} = \mathrm{softmax}_j\left(\frac{q_i k_j^\top}{\sqrt{d}} + M_{ij}\right),$$
where $M_{ij} = 0$ if $j \le i$ and $M_{ij} = -\infty$ otherwise. Equivalently:
$$\alpha_{ij} \propto \exp\!\left(\frac{q_i k_j^\top}{\sqrt{d}}\right) \cdot \mathbf{1}[j \le i].$$

**(b)** The causal mask ensures that position $i$'s representation depends only on positions $j \le i$. Therefore, the output distribution $\pi_\theta(y_t | \text{context})$ depends only on $(x, y_{1:t-1})$, not future tokens $y_{t+1:T}$. This is exactly the Markov property of the token-level MDP: the action (next token) is chosen based only on the current state (past context). $\blacksquare$

**(c)** For a language model used for **encoding** (e.g., BERT), the causal mask is not necessary and would be harmful: we want each token to attend to all other tokens (bidirectional attention) to build rich representations. The causal mask is needed specifically for **autoregressive generation** to prevent the model from "cheating" by looking at future tokens during training.

---

## Exercise 11.4
**Statement:** Formalise exposure bias mismatch; its impact; mitigation strategies.

**Solution:**

**(a) Formalisation:** Let $p_\mathrm{teacher}(y_{1:t-1})$ be the teacher-forcing distribution: the context at step $t$ always uses ground-truth $y_{1:t-1}^*$. The model's inference distribution $p_\mathrm{model}(y_{1:t-1})$ uses the model's own previous outputs.

These coincide only when $\pi_\theta(y_k^* | y_{<k}^*, x) = 1$ for all $k < t$, i.e., the model assigns probability 1 to the correct token at every step. In general, $p_\mathrm{model}$ is a mixture over all possible prefixes weighted by their probabilities under $\pi_\theta$, which differs from $p_\mathrm{teacher}$.

**(b) Impact at inference:** The model generates tokens autoregressively from its own distribution. Errors compound: a wrong token at step $t$ creates a context that was never seen during training (since training always used gold context), causing the model to be in an out-of-distribution state and potentially making further errors.

**(c) Mitigation strategies:**
1. **DAgger / scheduled sampling**: gradually replace gold tokens with model-predicted tokens during training.
2. **REINFORCE / RL fine-tuning**: train with sequence-level rewards under the model's own distribution.
3. **Minimum Bayes Risk (MBR) decoding**: at inference, use multiple samples to find the output that minimises expected loss.

---

## Exercise 11.5
**Statement:** Chinchilla scaling: derive $N^*$ and $D^*$ for $C = 6\times10^{23}$ FLOPs.

**Solution:**

**Chinchilla scaling law:** $D = 20N$ (compute-optimal), and training cost $C \approx 6ND$.

Substituting $D=20N$: $C = 6N\cdot20N = 120N^2$.

$$N^* = \sqrt{C/120}, \quad D^* = 20N^*.$$

**Numerically** for $C = 6\times10^{23}$:
$$N^* = \sqrt{\frac{6\times10^{23}}{120}} = \sqrt{5\times10^{21}} \approx \sqrt{5}\times10^{10.5} \approx 2.24\times10^{10.5} \approx \mathbf{7\times10^{10}} \text{ parameters (70B)}.$$

$$D^* = 20N^* \approx 20\times7\times10^{10} = \mathbf{1.4\times10^{12}} \text{ tokens (1.4 trillion)}.$$

This is consistent with the Llama-2 70B model (~70B parameters) trained on ~2T tokens (slightly more than the Chinchilla-optimal ratio, which is a common practice).

---

## Exercise 11.6
**Statement:** Self-Instruct: (a) effect of seed quality; (b) ROUGE-L false positives; (c) diversity vs quality.

**Solution:**

**(a)** The quality of the seed set affects generated quality because the teacher model is prompted with in-context examples from the seed. High-quality seeds → the teacher generates instructions with similar style, specificity, and domain coverage. Low-quality seeds (ambiguous, short, wrong) → the teacher generates similarly low-quality instructions. Since the filtered dataset is used for SFT, its quality directly determines the SFT model's capabilities.

**(b)** Example pair of distinct instructions flagged as similar by ROUGE-L:
- "Summarise the following article in 3 sentences."
- "Provide a 3-sentence abstract for the article below."
Both contain "3 sentences" and similar n-grams, but they may target different skills (extractive vs. abstractive summarisation) or different output styles. ROUGE-L is a surface similarity metric and misses semantic distinctions.

**(c)** Diversity-quality trade-off:
- High threshold: keeps only clearly distinct instructions → diverse but may include low-quality ones.
- Low threshold (aggressive deduplication): keeps only the most unique ones → high quality but narrow coverage.
Best practice: combine ROUGE-L filtering with a quality classifier (trained on human-annotated quality scores).

---

## Exercise 11.7
**Statement:** Model collapse analysis.

**Solution:**

**(a)** With TV error bounded by $\epsilon$ per generation: $\mathrm{TV}(\hat{p}_g, p_0) \le g\epsilon$.

After $g$ generations, the total variation from the original distribution is at most $g\epsilon$. As $g \to \infty$, the distribution can drift arbitrarily far from $p_0$.

**(b)** The tail of the distribution is lost first. High-entropy or low-probability modes in $p_0$ require many samples to be faithfully estimated. The learned $\hat{p}_{g}$ will under-represent tails — the model concentrates on the most common outputs. Over generations, this leads to a narrowing distribution (less diverse, more repetitive outputs).

**(c) Mitigation:** Mix synthetic data with original human data at each generation, ensuring a constant fraction of $p_0$ is preserved. Formally, if at each step we use data $\hat{p}_g = (1-\alpha)\hat{p}_{g-1} + \alpha p_0$, the TV distance to $p_0$ is bounded by $\alpha \cdot g$ rather than growing unboundedly.

---

## Exercise 11.8
**Statement:** LLM-as-judge biases and mitigations.

**Solution:**

**(a) Verbosity bias:** LLM judges tend to prefer longer responses because they contain more detail and appear more thorough, even if the added content is redundant. **Mitigation:** Instruct the judge explicitly to evaluate quality independent of length ("Do not factor response length into your evaluation"), and include length-controlled examples in few-shot prompts showing that concise, correct responses are preferred over verbose, partially-correct ones.

**(b) Eliminating position bias:** Present each response pair twice, once in each order (A then B, and B then A). Collect two judgements per pair. Aggregate: if the judge consistently prefers response X regardless of position, score it as preferred. If the judgements are inconsistent (first-position preference), treat it as a tie or use a third tiebreaker. This symmetric protocol eliminates the position effect because any positional bias is balanced across the two orderings.

**(c)** If the reward model was trained on LLM-generated preferences, and the LLM judge and the policy share the same base model, the reward model has learned the judge's biases (verbosity, etc.). The policy will then exploit these biases (producing verbose, sycophantic responses) just as it would exploit a flawed human-annotated reward model. The same Goodhart's law dynamics apply.

---

## Exercise 11.9
**Statement:** Constitutional AI critique-revision loop, RLAIF, and relation to SFT.

**Solution:**

**(a) Critique-revision loop:**
1. **Critique:** Generate an initial response $y_0$ to prompt $x$. Feed $y_0$ and a constitution principle (e.g., "Is the response harmful? If so, explain why.") to the model. Obtain a critique $c$.
2. **Revision:** Feed $(x, y_0, c)$ to the model with instruction to revise. Obtain improved response $y_1$.

*Role of the constitution:* The principles guide what aspects to critique, encoding human values without requiring per-example human labels.

*Difference from standard SFT:* SFT trains on human-provided demonstrations; Constitutional AI generates its own training data through self-improvement, reducing human annotation burden.

**(b) RLAIF with perfect judge:** If the judge perfectly applies the constitution, its preference labels are consistent with the constitution's ground truth. In the limit, the preference dataset collected by RLAIF is equivalent to one collected by an infallible human annotator who agrees with the constitution. The resulting RM and RLHF policy would be identical to those obtained from perfect human annotation, making RLAIF equivalent to RLHF with ground-truth preferences.

---

## Exercise 11.10
**Statement:** Binary string MDP, reward $+1$ if equal zeros and ones. (a) Full MDP. (b) Optimal policy via DP. (c) Scale to $T=100$.

**Solution:**

**(a)** States: all binary prefixes of length 0–3. State at time $t$: $(t, n_1)$ where $n_1$ = number of 1s so far.

| $t$ | State $(t,n_1)$ | Actions | Next states | Reward |
|-----|-----------------|---------|-------------|--------|
| 0 | $(0,0)$ | 0,1 | $(1,0),(1,1)$ | 0 |
| 1 | $(1,0),(1,1)$ | 0,1 | — | 0 |
| 2 | $(2,0),(2,1),(2,2)$ | 0,1 | — | 0 |
| 3 | Terminal $(3,n_1)$ | — | — | +1 if $n_1=1$ or $n_1=2$, else −1 |

Wait: for $T=3$ (length 3 string), equal zeros and ones requires $n_1 = 1$ or $n_1 = 2$ out of 3 tokens — but 3 tokens cannot have equal numbers of 0s and 1s. The problem must mean $T$ even. For $T=3$, the reward condition "$n_1 = n_0$" requires $n_1=1.5$ which is impossible — perhaps the problem means equal counts with rounding, or $T$ should be even.

Assuming the condition is "at least as many 0s as 1s and at least as many 1s as 0s" (i.e., balance as close as possible): for $T=3$, reward +1 if $n_1 \in \{1,2\}$ (at most 1 off from balance), else $-1$.

**(b) Optimal policy** (by backward induction):
- At $t=2$: if current $n_1=0$ → must choose 0 or 1; choosing 1 gives state $(3,1)$ → +1, choosing 0 gives $(3,0)$ → −1. Optimal: choose 1.
- If $n_1=1$: choosing 0 → $(3,1)$ → +1; choosing 1 → $(3,2)$ → +1. Both optimal.
- If $n_1=2$: choosing 0 → $(3,2)$ → +1; choosing 1 → $(3,3)$ → −1. Optimal: choose 0.
- The optimal policy is to balance the string: choose the minority token at each step.

**(c) Scale to $T=100$:** The state is $(t, n_1)$ with $0 \le n_1 \le t$. There are $O(T^2)$ states, making DP tractable ($O(T^2)$ operations). The optimal policy is the "greedy balancing" policy: choose 1 if $n_1 < T/2 - (T-t)/2$ and 0 otherwise — always choose the action that keeps the count balanced.
