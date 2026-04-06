# Chapter 14: Practical RLHF — Solutions

## Exercise 14.1
**Statement:** 7B model, PPO RLHF, KL=18 nats, reward score 2.1→4.8. (a) KL healthy? (b) Length increase failure mode. (c) Fix.

**Solution:**

**(a)** KL divergence of 18 nats is in an **unhealthy range**. Typical healthy KL for PPO RLHF is 1–15 nats; values above 15–20 nats suggest the policy has drifted substantially from the reference, risking reward model overoptimisation (Goodhart's law). The reward increase from 2.1 to 4.8 combined with high KL is suspicious — it may reflect reward hacking rather than genuine improvement.

**(b)** Mean response length increasing from 120 to 350 tokens suggests **verbosity reward hacking** (also called "length exploitation" or "padding"). The reward model has learned to assign higher scores to longer responses (a length bias), and the policy exploits this by generating unnecessarily verbose outputs. This is a form of Goodhart's law.

**(c)** Fix strategies:
1. **Length penalty:** Add $-\alpha \cdot L$ to the per-token reward, where $L$ is response length and $\alpha$ is a tunable coefficient.
2. **Reward model retraining:** Collect new preference pairs that specifically penalise verbosity; retrain the RM with length-balanced examples.
3. **KL-based early stopping:** Stop training when KL exceeds a threshold (e.g., KL $\le$ 10 nats).
4. **Response length constraint:** Hard-truncate responses and only reward the first $K$ tokens.

---

## Exercise 14.2
**Statement:** Four-model PPO setup for 13B parameters. (a) Memory estimate. (b) A100 feasibility. (c) GRPO memory saving.

**Solution:**

**(a)** A 13B parameter model in BF16 requires $13\times10^9 \times 2$ bytes $= 26$ GB.

Four models (actor, critic, reference, reward model): $4 \times 26 = \mathbf{104}$ GB, ignoring activations, KV cache, and optimizer states.

With optimizer states (Adam: $3\times$ model params): actor + critic with Adam $\approx 2\times3\times26=156$ GB; plus reference and RM (inference only, no optimizer): $2\times26=52$ GB. Total $\approx 208$ GB minimum.

**(b)** 8× A100 (80 GB each) = 640 GB total. Data parallelism alone replicates all models across GPUs — each GPU needs $\ge 104$ GB (inference) or more for training. Since one GPU (80 GB) cannot hold even 2 models, **data parallelism alone is insufficient**. Tensor parallelism is required: split each model across multiple GPUs (e.g., 2-way or 4-way tensor parallel) so that each GPU only holds a fraction of the model weights.

**(c)** GRPO eliminates the critic/value model. Memory saving: $1\times26$ GB (just the value model), reducing from 104 to 78 GB for three models in BF16. With optimizer states, the saving is larger since the critic's Adam states are also eliminated (~$3\times26=78$ GB saved). Total potential saving: ~78–100 GB depending on optimizer configuration.

---

## Exercise 14.3
**Statement:** Design experiment to detect reward hacking.

**Solution:**

**(a) Gold evaluation protocol:**
1. Sample 200 prompts from a held-out prompt set never seen by $r_\psi$.
2. For each prompt, collect 5 responses from the current policy $\pi_\theta$.
3. Present pairs to human annotators (blind to which policy generated which) for pairwise preference labelling. Use a double-blind design: annotators don't know they are evaluating an RL-trained model.
4. Resulting dataset: 500 prompts × 5 responses = 2,500 human preference comparisons.

**(b) Quantitative metric:** "Proxy-gold gap" plotted over training time:
$$\Delta_t = \frac{1}{N}\sum_i r_\psi(x_i, y_{\pi_t}) - \frac{1}{N}\sum_i r_\mathrm{human}(x_i, y_{\pi_t}).$$

**Pattern indicating hacking:** $r_\psi$ increases monotonically while $r_\mathrm{human}$ peaks early and then declines (or plateaus while $r_\psi$ continues growing). A divergence between the two curves signals reward hacking.

**(c)** If $r_\psi$ increases but $r_\mathrm{human}$ does not: the policy has learned to exploit the reward model's spurious features (formatting, length, etc.) without genuinely improving response quality. Solutions: stop training at the peak of $r_\mathrm{human}$, retrain the reward model on policy's outputs, add a holdout reward model for regularisation.

---

## Exercise 14.4
**Statement:** Annotation guidelines for pairwise comparison; cost estimation; alternative strategies.

**Solution:**

**(a) Annotation guidelines (3–5 sentences):**

"Compare two chatbot responses to a customer support query. Prefer the response that fully resolves the customer's issue over one that is only partially helpful, regardless of response length. A shorter correct response is preferred over a longer response that includes unnecessary information or padding. Do not prefer a response solely because it is more polite or uses more formal language if the other response is more accurate and helpful. Prefer the response that escalates appropriately when the issue is beyond the chatbot's capabilities."

**(b)** Budget: $\$200{,}000$ at $\$10$/pair = 20,000 expert pairs.

For 50,000 pairs total, remaining 30,000 pairs need alternative sources:

1. **Crowd-workers** (e.g., MTurk, Scale AI): $\$0.5$–$\$2$ per comparison → 15,000–60,000 additional pairs at lower quality. Use majority vote across 3 annotators.
2. **LLM-as-judge** (e.g., GPT-4): $\sim\$0.01$ per comparison → 30,000 pairs very cheaply. Validate against expert labels; use only where inter-rater agreement with experts is high.
3. **Implicit feedback** from production: user satisfaction ratings, re-contact rates, ticket resolution signals — potentially millions of signals but noisier.

---

## Exercise 14.5
**Statement:** GAE computation for 4-token response.

**Solution:**

Per-token rewards: $\tilde{r} = (0.1, -0.2, 0.0, 1.5)$.
Value estimates: $V = (0.8, 0.6, 0.9, 1.0, 0)$ (terminal value $V_5=0$).
$\gamma=1$, $\lambda=0.95$.

**(a) TD residuals:** $\delta_t = \tilde{r}_{t+1} + \gamma V_{t+1} - V_t$:
$$\delta_1 = 0.1 + V_2 - V_1 = 0.1 + 0.6 - 0.8 = -0.1$$
$$\delta_2 = -0.2 + V_3 - V_2 = -0.2 + 0.9 - 0.6 = 0.1$$
$$\delta_3 = 0.0 + V_4 - V_3 = 0.0 + 1.0 - 0.9 = 0.1$$
$$\delta_4 = 1.5 + V_5 - V_4 = 1.5 + 0 - 1.0 = 0.5$$

**(b) GAE advantages** (backward, $c = \gamma\lambda = 0.95$):
$$\hat{A}_4 = 0.5$$
$$\hat{A}_3 = 0.1 + 0.95\times0.5 = 0.1+0.475 = 0.575$$
$$\hat{A}_2 = 0.1 + 0.95\times0.575 = 0.1+0.546 = 0.646$$
$$\hat{A}_1 = -0.1 + 0.95\times0.646 = -0.1+0.614 = 0.514$$

**(c) With $\lambda=0$:** $\hat{A}_t = \delta_t$: $(-0.1, 0.1, 0.1, 0.5)$.  
With $\lambda=1$:** $\hat{A}_t = \sum_{k=t}^4 \delta_k$:
$$\hat{A}_4=0.5, \; \hat{A}_3=0.6, \; \hat{A}_2=0.7, \; \hat{A}_1=0.6.$$

---

## Exercise 14.6
**Statement:** Propose parallelism strategy for 70B model on 64 A100 GPUs.

**Solution:**

**Model size:** 70B × 2 bytes (BF16) = 140 GB. One 80GB A100 cannot hold a single model.

**Generation phase (autoregressive, memory-bound):**
- **Tensor parallelism (TP)** degree 4: split attention heads and FFN across 4 GPUs within a node.
- **Pipeline parallelism (PP)** degree 1 (no pipeline for generation — latency is critical).
- **Data parallelism (DP)** degree 16 (64/4 = 16 independent replicas).
- Configuration: 4-way TP × 16-way DP = 64 GPUs. ✓
- NVLink used within each 8-GPU node (4-way TP per node).

**Training phase (backward pass, compute-bound):**
- **Tensor parallelism** degree 8 (within node, full 8-GPU NVLink bandwidth).
- **Pipeline parallelism** degree 2: split model across 2 nodes (16 micro-batches to hide bubble overhead).
- **Data parallelism** degree 4: 64/(8×2) = 4 DP replicas.
- Configuration: 8-way TP × 2-way PP × 4-way DP = 64 GPUs. ✓
- All-reduce for DP gradients uses InfiniBand between nodes.

**(b) Generation bottleneck:** Autoregressive generation is memory-bandwidth bound (each token requires loading all KV cache and model weights). The KV cache for a 70B model with sequence length 2048 is $\approx 2\times70B\times\text{layers}\times\text{heads}$... more practically: continuous batching and paged KV cache (as in vLLM) maximise GPU utilisation. 

The generation phase typically has 3–10× lower throughput than training; using separate generation and training replicas (asynchronous actor-learner) can improve overall throughput.

---

## Exercise 14.7
**Statement:** Compare InstructGPT, Llama 2, and DeepSeek alignment approaches.

**Solution:**

| Dimension | InstructGPT | Llama 2 | DeepSeek |
|-----------|:-----------:|:-------:|:--------:|
| Models in GPU memory during RL training | 4 (actor, critic, RM, ref) | 4 (same PPO setup) | 2–3 (GRPO: actor + ref; sometimes + RM) |
| Human preference data | ~13k comparison pairs (crowdworkers) | ~1M comparison pairs (Meta annotators, strict guidelines) | Mix of human + AI-generated preferences; math/code from verifiable rewards |
| RM staleness strategy | Periodic RM updates; KL constraint limits drift | KL-regularised reward; ghost attention for multi-turn | Iterative RM retraining + GRPO replaces RM for verifiable tasks |
| Suitability for verifiable rewards | Low (general chat, no verifier available) | Low-medium (some coding tasks) | High (GRPO designed for math/code verification) |

**Summary:** InstructGPT established the RLHF pipeline; Llama 2 scaled it up with much more human data and careful annotation; DeepSeek moved beyond RLHF for reasoning tasks by using verifiable rewards and GRPO, eliminating the need for a learned reward model where exact verification is available.
