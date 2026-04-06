# Chapter 12: Direct Preference Optimisation and Alignment Methods — Solutions

## Exercise 12.1
**Statement:** Full DPO derivation from KL-regularised objective through to DPO loss.

**Solution:**

**Step 1: KL-regularised objective.** Maximise over policies $\pi$:
$$\max_\pi \mathbb{E}_{x\sim\mathcal{D}, y\sim\pi}\left[r_\psi(x,y) - \beta\log\frac{\pi(y|x)}{\pi_\mathrm{ref}(y|x)}\right].$$

**Step 2: Optimal policy.** Setting the functional derivative to zero:
$$r_\psi(x,y) - \beta\left[\log\pi(y|x) - \log\pi_\mathrm{ref}(y|x)\right] - \beta = 0,$$
$$\pi^*(y|x) = \frac{\pi_\mathrm{ref}(y|x)e^{r_\psi(x,y)/\beta}}{Z(x)}, \quad Z(x) = \sum_y \pi_\mathrm{ref}(y|x)e^{r_\psi(x,y)/\beta}.$$

**Step 3: Reward reparametrisation.** Taking the log of the optimal policy:
$$\log\pi^*(y|x) = \log\pi_\mathrm{ref}(y|x) + \frac{r_\psi(x,y)}{\beta} - \log Z(x),$$
$$r_\psi(x,y) = \beta\log\frac{\pi^*(y|x)}{\pi_\mathrm{ref}(y|x)} + \beta\log Z(x).$$

**Step 4: Partition function cancels.** The reward difference:
$$r_\psi(x,y_w) - r_\psi(x,y_l) = \beta\log\frac{\pi^*(y_w|x)}{\pi_\mathrm{ref}(y_w|x)} - \beta\log\frac{\pi^*(y_l|x)}{\pi_\mathrm{ref}(y_l|x)}.$$
$Z(x)$ cancels. $\blacksquare$

**Step 5: DPO loss.** Substituting into the Bradley–Terry model $P(y_w\succ y_l|x) = \sigma(r(y_w)-r(y_l))$:
$$\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\mathrm{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_\mathrm{ref}(y_l|x)}\right)\right]. \quad \blacksquare$$

---

## Exercise 12.2
**Statement:** Numerical DPO gradient verification.

**Solution:**

Given: $\log\pi_\theta(y_w|x)=-2.0$, $\log\pi_\mathrm{ref}(y_w|x)=-2.4$, $\log\pi_\theta(y_l|x)=-1.5$, $\log\pi_\mathrm{ref}(y_l|x)=-1.3$, $\beta=0.1$.

**(a)** Implicit reward difference:
$$h_\theta = \beta\left[\left(\log\frac{\pi_\theta(y_w)}{\pi_\mathrm{ref}(y_w)}\right) - \left(\log\frac{\pi_\theta(y_l)}{\pi_\mathrm{ref}(y_l)}\right)\right]$$
$$= 0.1\left[(-2.0-(-2.4)) - (-1.5-(-1.3))\right] = 0.1\left[0.4 - (-0.2)\right] = 0.1\times0.6 = \mathbf{0.06}.$$

**(b)** DPO loss: $\mathcal{L}_{\mathrm{DPO}} = -\log\sigma(h_\theta) = -\log\sigma(0.06)$.

$\sigma(0.06) = 1/(1+e^{-0.06}) \approx 1/(1+0.9418) = 1/1.9418 \approx 0.515$.

$\mathcal{L}_{\mathrm{DPO}} \approx -\log(0.515) \approx \mathbf{0.664}$.

**(c)** Confidence weight $\sigma(-h_\theta) = \sigma(-0.06) \approx 1-0.515 = 0.485$.

This is close to 0.5, meaning the model is barely more confident than chance that $y_w$ is preferred. The gradient will be large (confidence weight $\approx 0.485 \approx 1/2$), suggesting the model needs significant updating. The confidence weight $\sigma(-h_\theta)$ is close to 1 when the model is very wrong (confidently preferring $y_l$), giving large gradient; close to 0 when already confident (large $h_\theta$), giving small gradient — analogous to logistic regression.

---

## Exercise 12.3
**Statement:** DPO has no finite minimiser for perfectly consistent preferences; IPO minimiser at $h_\theta^* = 1/(2\beta)$.

**Solution:**

**(a)** When $y_w \succ y_l$ with probability 1, the DPO loss is:
$$\mathcal{L}_{\mathrm{DPO}} = -\log\sigma(h_\theta).$$

As $h_\theta \to +\infty$: $\sigma(h_\theta) \to 1$, so $\mathcal{L}_{\mathrm{DPO}} \to 0$. The loss approaches but never reaches 0. Since the loss has no minimum (it is an infimum achieved only in the limit), there is no finite minimiser — the implicit reward difference $h_\theta$ diverges to $+\infty$, meaning the policy drifts unboundedly from the reference. $\blacksquare$

**(b)** The IPO loss: $\mathcal{L}_{\mathrm{IPO}} = (h_\theta - 1/(2\beta))^2$ (plus normalisation terms). Setting $\partial\mathcal{L}/\partial h_\theta = 0$:
$$2(h_\theta - 1/(2\beta)) = 0 \implies h_\theta^* = \frac{1}{2\beta}.$$

For $\beta=0.1$: $h_\theta^* = 1/(0.2) = 5$.

**Interpretation:** IPO has a unique, finite minimiser at $h_\theta^* = 1/(2\beta)$. Smaller $\beta$ (weaker regularisation) → larger optimal margin $h_\theta^*$; larger $\beta$ → smaller margin. Unlike DPO, IPO does not collapse to infinite confidence on the preferred response.

---

## Exercise 12.4
**Statement:** KTO and prospect theory — value function, loss aversion, reference point.

**Solution:**

**(a)** Kahneman-Tversky value function: concave for gains, convex for losses, with $v(0)=0$.

Loss aversion: $|v'(-\epsilon)| > |v'(\epsilon)|$ for small $\epsilon > 0$.

This means the marginal disutility of a loss exceeds the marginal utility of an equal gain. Sketch: the function is steeper to the left of the origin than the right.

```
  v(x)
  |        ___---
  |    ___-
  |___- 
--+-----------> x
  |
  |\
  | \___
  |     ---___
```
The slope at $x=0^-$ is steeper than at $x=0^+$.

**(b)** The reference point $z_\mathrm{ref}$ is the expected KL-penalised reward under a baseline (e.g., random policy or reference policy). It plays the role of "zero" in prospect theory: responses above $z_\mathrm{ref}$ are treated as gains and responses below as losses. Shifting $z_\mathrm{ref}$ up makes more responses feel like losses, increasing the effective learning signal for improvement.

**(c)** In KTO, winner and loser examples are processed separately with a value function:
- For winners: $v(r - z_\mathrm{ref})$ with $r > z_\mathrm{ref}$ (gains region, risk-averse).
- For losers: $v(r - z_\mathrm{ref})$ with $r < z_\mathrm{ref}$ (losses region, loss-averse, steeper gradient).

This asymmetry gives stronger gradient signal for bad responses than good ones of equal magnitude, encoding the human tendency to respond more strongly to losses.

---

## Exercise 12.5
**Statement:** Derive cDPO from DPO with label-flip probability $\epsilon$.

**Solution:**

Standard DPO loss for a perfectly labelled pair: $-\log\sigma(h_\theta)$.

With label-flip probability $\epsilon$: the observed "winner" is the true winner w.p. $1-\epsilon$ and the true loser w.p. $\epsilon$.

Expected DPO loss:
$$\mathcal{L}_{\mathrm{cDPO}} = -(1-\epsilon)\log\sigma(h_\theta) - \epsilon\log\sigma(-h_\theta).$$

This is the cDPO loss — a mixture of the two orderings weighted by label reliability.

For $\epsilon=0.5$ (uninformative annotator): $\mathcal{L}_{\mathrm{cDPO}} = -0.5\log\sigma(h_\theta) - 0.5\log\sigma(-h_\theta)$.

This equals $\log 2 + 0.5\log[\sigma(h_\theta)\sigma(-h_\theta)]^{-1}$, which is minimised at $h_\theta=0$: the policy defaults to the reference policy, correctly ignoring completely uninformative labels. $\blacksquare$

---

## Exercise 12.6
**Statement:** Distribution shift in DPO — coverage decreases, online RL advantage.

**Solution:**

**(a)** As $d = \mathrm{KL}(\pi_\theta\|\pi_\mathrm{ref})$ grows, by Pinsker's inequality $\|\pi_\theta - \pi_\mathrm{ref}\|_1 \le \sqrt{2d}$. The support of $\pi_\theta$ diverges from that of $\pi_\mathrm{ref}$: responses with high probability under $\pi_\theta$ but low probability under $\pi_\mathrm{ref}$ are not covered by the preference dataset (which was sampled from $\pi_\mathrm{ref}$). The fraction of uncovered responses grows with $d$.

**(b)** DPO is offline: it learns from a fixed dataset of preferences generated by $\pi_\mathrm{ref}$. As the model improves, it generates new, higher-quality responses that were never in the training set. DPO cannot assign rewards to these novel responses.

Online RL (e.g., PPO with RM) collects new preferences from $\pi_\theta$ at each training step, always using on-policy data. As the model improves, new rollouts are collected and their rewards estimated, maintaining coverage.

**(c)** Iterative DPO: run DPO for $k$ steps, collect new preference data from $\pi_\theta^{(k)}$, add to dataset, repeat. Each round adds on-policy coverage. This reduces distribution shift at the cost of requiring additional data collection.

---

## Exercise 12.7
**Statement:** Design alignment pipeline for a customer service chatbot.

**Solution:**

**(a) Alignment methods by objective:**

| Objective | Method | Rationale |
|-----------|--------|-----------|
| (a) Resolve issues correctly | GRPO with verifiable rewards (e.g., "issue resolved" signal from CRM system) | Binary verifiable reward; RL with exact feedback |
| (b) Respond politely to abusive messages | Constitutional AI (critique-revision) + DPO | Human-defined principles for politeness; paired preference data |
| (c) Escalate appropriately | RLHF with human labels on escalation appropriateness | Nuanced judgement requires human feedback |

**(b) Feedback collection:**

- **Explicit:** Post-conversation survey ("Was your issue resolved? 1–5 stars"). Reliable but low response rate.
- **Implicit:** CRM system tracks whether tickets are re-opened (issue not resolved), escalation frequency, and resolution time. High volume but noisy.
- **Combination:** Use implicit signals for training data scale; explicit for reward model calibration.

**(c) Conflicting objectives:** Resolving issues quickly (a) may conflict with politeness (b) — blunt, efficient responses may score high on (a) but low on (b). A scalarisation $r = w_a r_a + w_b r_b + w_c r_c$ with appropriate weights, or a constrained RL approach (maximise (a) subject to (b) $\ge$ threshold) handles this.

---

## Exercise 12.8
**Statement:** Memory comparison table; dataset choice; GRPO stability.

**Solution:**

**(a) PPO vs DPO memory:**
- **PPO**: requires actor, critic (value model), reference model, reward model in GPU memory = $4\times$ a single model.
- **DPO**: requires only the trained policy and the frozen reference model = $2\times$ a single model (no reward model, no value model needed).

**(b) Dataset: 50k interaction logs, labelled resolved/not resolved, no paired comparisons.**

Recommended method: **KTO (Kahneman-Tversky Optimisation)**. KTO is designed for unpaired binary feedback (good/bad outcomes) without needing preference pairs. The labels "resolved/not resolved" map directly to the KTO winner/loser distinction.

DPO requires paired comparisons (not available here). PPO needs a reward model (which requires training data — could be the current labels, but KTO is more direct). KTO directly optimises on the binary signal.

**(c) GRPO stability** (listed as "H" = high): GRPO normalises advantages within a group, making it robust to reward scale changes. The group-normalised advantages avoid the need for a value baseline, eliminating the critic's approximation error as a source of instability.

---

## Exercise 12.9
**Statement:** DPO on math reasoning — diagnosing improvement source; format vs reasoning.

**Solution:**

**(a) Experiment to distinguish learning formatting vs correct reasoning:**

Design a dataset of **format-equivalent** math problems where format is controlled:
- Take 100 problems where the DPO model is correct.
- Manually verify that the *reasoning chain* (not just final answer) is correct.
- Compare with 100 problems where the model gives the correct final answer but with wrong or missing intermediate steps.

Compute accuracy separately on "format-only" improvements (answer correct, chain wrong) vs "genuine reasoning" improvements (chain and answer correct). If most improvement is in the former category, the model learned formatting, not reasoning.

Additionally, test on problems that require the same reasoning patterns but with different superficial features (different numbers, different format requirements) to test genuine generalisation.

**(b) From 45% to 58%:** At 45% baseline accuracy, many problems have no correct solution in the training pairs. DPO cannot learn from pairs where $y_w$ is wrong. As accuracy increases, more correct solutions become available in $y_w$, compounding the improvement.

**(c) Prevent format exploitation:** Include in the preference dataset pairs where the correct answer is formatted differently from any "preferred" format, and mark these with preference based on correctness only. Include chain-of-thought verification as part of the preference signal (PRM-style labels).
