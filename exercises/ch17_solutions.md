# Chapter 17: Open Problems and Emerging Directions — Solutions

*Chapter 17 contains no formal exercises in the \begin{exercise}...\end{exercise} environment. Instead, it presents open research questions and discussion topics. Below are structured solutions to the key discussion questions raised throughout the chapter.*

---

## Discussion 17.1: Sample Efficiency and Data-Driven RL

**Question:** What are the fundamental limits of sample efficiency in RL, and how close are current methods to those limits?

**Solution:**

The sample complexity of RL for tabular MDPs is well-understood: $\tilde{O}(|\mathcal{S}||\mathcal{A}|/(1-\gamma)^3\varepsilon^2)$ samples are sufficient and necessary (Azar et al., 2017). For function approximation settings, tight bounds exist for linear MDPs.

**Gap with practice:** Current deep RL methods require orders of magnitude more samples than theoretical lower bounds suggest. DQN on Atari requires $\sim10^8$ frames; a human expert learns from $\sim10^4$.

**Promising directions:**
1. **Offline RL** (CQL, IQL, TD3+BC): exploit large datasets from previously collected experience.
2. **Model-based RL** (Dreamer, MuZero): build world models to generate synthetic experience.
3. **Transfer learning and meta-RL**: reuse knowledge across tasks to reduce per-task sample requirements.
4. **Imitation learning + RL**: initialise from demonstrations, then fine-tune with RL.

The key open question: can we close the 4–5 order-of-magnitude gap between human and RL agent sample efficiency on complex tasks?

---

## Discussion 17.2: Reward Specification and Alignment

**Question:** How can we specify complex, nuanced human values in a reward function, and how do we prevent reward hacking?

**Solution:**

**The specification problem:** Human values are complex, context-dependent, and often contradictory. Any fixed reward function will be incomplete; the agent will find ways to satisfy the letter but not the spirit of the reward (Goodhart's law).

**Current approaches:**
- RLHF: crowd-source human preferences; train reward model.
- Constitutional AI: use AI to critique and revise based on principles.
- Process reward models: evaluate reasoning steps, not just outcomes.
- Debate: have AIs argue for and against; human judges the debate.

**Open problems:**
1. **Reward model overfitting**: RM captures annotator biases rather than true preferences.
2. **Distribution shift**: RM trained on $\pi_\mathrm{ref}$ fails for $\pi_\theta$ far from reference.
3. **Value aggregation**: different humans have different values; whose values should the RM capture?
4. **Scalable oversight**: as AI becomes more capable than humans at many tasks, how do we maintain human oversight?

---

## Discussion 17.3: Offline RL and Distributional Shift

**Question:** What are the key challenges in offline RL, and how do current algorithms address them?

**Solution:**

**Core challenge: distributional shift.**
In offline RL, the learned policy may take actions that were rarely (or never) taken by the behaviour policy. The Q-function, trained on in-distribution data, may be erroneously optimistic for out-of-distribution actions.

**Conservative approaches:**
- **CQL (Conservative Q-Learning):** Adds a regulariser that explicitly penalises Q-values for unseen actions: $\min_Q \alpha(\mathbb{E}_{s\sim\mathcal{D},a\sim\mu}[Q(s,a)] - \mathbb{E}_{(s,a)\sim\mathcal{D}}[Q(s,a)]) + \mathcal{L}_{\mathrm{Bellman}}$.
- **IQL (Implicit Q-Learning):** Avoids evaluating Q-values at OOD actions entirely by using expectile regression to estimate $V(s) = \mathbb{E}_\tau[Q(s,a)]$.
- **TD3+BC:** Combines standard TD3 with a behavioural cloning regulariser to stay close to the data distribution.

**When offline RL works well:** When the dataset has good coverage (diverse behaviour policy), when the target task matches the training distribution, and when the dataset contains near-optimal trajectories (expert data).

**Open question:** Can offline RL methods reliably extrapolate beyond the training distribution, and if so, under what conditions?

---

## Discussion 17.4: World Models and Model-Based RL

**Question:** What are the benefits and limitations of world models for RL?

**Solution:**

**Benefits:**
1. **Sample efficiency**: plan in "imagination" without costly environment interactions (Dreamer achieves Atari SOTA with $\sim10\times$ fewer samples than model-free DQN).
2. **Zero-shot transfer**: a world model learned in one environment can support planning in related environments.
3. **Counterfactual reasoning**: "what would happen if I took action $a$ instead?"

**Limitations:**
1. **Compounding errors**: model errors compound over multi-step rollouts. For long-horizon planning, the model's distribution diverges from the real environment.
2. **Distribution shift**: the world model is trained on data from the current policy; it may be inaccurate for states visited by an improved policy.
3. **Hard environments**: stochastic or partially-observable environments are much harder to model accurately.

**Dreamer/MuZero approach:** Limit planning to short horizons (2–5 steps for Dreamer); use only abstract latent-space rollouts (not pixel-level predictions). This reduces the compounding error problem.

**Open question:** Can world models generalise to complex real-world environments (language, multi-agent, continuous action spaces) at the level needed for general-purpose planning?

---

## Discussion 17.5: RL for Language Model Reasoning

**Question:** What are the open problems at the intersection of RL and large language models?

**Solution:**

**Current state (2024–2025):** GRPO, RLVR, and process reward models have dramatically improved mathematical and coding reasoning in LLMs. Models like DeepSeek-R1 demonstrate that verifiable-reward RL can match or exceed human expert performance on competition mathematics.

**Key open problems:**

1. **Generalisation beyond training domains:** Models trained on math problems with RL do not reliably transfer their improved reasoning to other domains. Is "general reasoning" a monolithic skill, or is it domain-specific?

2. **Reward specification for open-ended tasks:** RLVR works well for tasks with clear correct answers. For creative writing, advice, or nuanced ethical questions, verifiable rewards don't exist. How do we apply RL to these domains?

3. **Long-horizon reasoning:** Current chains of thought are $O(100)$–$O(1000)$ tokens. Some hard problems require much longer reasoning. How do we train RL for very long chains without reward sparsity?

4. **Tool learning:** Current tool-calling is mostly supervised. Can RL discover novel tool combinations or learn to use tools in unexpected ways?

5. **Safety during RL:** The RL training phase can introduce harmful behaviours not present in the base model. How do we maintain safety constraints during GRPO/RLVR training?

---

## Discussion 17.6: Multi-Agent and Emergent Behaviours

**Question:** Can RL systems exhibit robust emergent cooperation and coordination at scale?

**Solution:**

**Evidence for emergent cooperation:** OpenAI's multi-agent hide-and-seek experiments demonstrated agents learning to use objects as tools, developing strategies not programmed by designers. AlphaGo/AlphaStar discovered novel game strategies through self-play.

**Challenges at scale:**
1. **Non-stationarity**: more agents → environment changes faster → harder to converge.
2. **Credit assignment**: with $N$ agents, attributing individual contributions to shared outcomes scales poorly.
3. **Communication overhead**: decentralised execution with limited communication is a hard constraint.
4. **Adversarial dynamics**: in mixed cooperative-competitive settings, agents may develop deceptive strategies.

**Open questions:**
1. Can population-based RL produce agents that generalise to new teammates/opponents zero-shot?
2. Can emergent communication protocols be aligned with human language to improve interpretability?
3. What is the computational complexity of finding Nash equilibria in large-scale multi-agent systems?

---

## Discussion 17.7: Theoretical Frontiers

**Question:** What are the main theoretical gaps between current understanding and practical deep RL?

**Solution:**

**1. Generalization theory for RL:**  
Statistical learning theory (PAC bounds) applies to supervised learning but not directly to RL. We lack tight sample complexity bounds for neural network function approximators in RL.

**2. Optimisation landscape:**  
Policy gradient methods are non-convex. We lack theoretical understanding of why they converge in practice, especially with the deadly triad present.

**3. Exploration theory:**  
UCB and Thompson sampling have near-optimal guarantees in tabular bandits. In deep RL with high-dimensional state spaces, exploration remains largely heuristic. Intrinsic motivation methods (curiosity, count-based) lack formal guarantees.

**4. Credit assignment:**  
Temporal credit assignment over long horizons is difficult both theoretically and practically. Eligibility traces help but don't fully solve the problem.

**5. Multi-objective RL:**  
Balancing multiple objectives (helpfulness, safety, conciseness) lacks a principled theory analogous to multi-objective optimisation.

**Near-term research directions:**
- PAC-Bayes bounds for neural Q-functions.
- Information-theoretic exploration bounds.
- Provably efficient offline RL for MDPs with linear or low-rank structure.
- Formal safety guarantees for constrained RL policies.
