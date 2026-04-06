# Chapter 16: Multi-Agent Systems and RL — Solutions

## Exercise 16.1
**Statement:** (From ch16 — normal-form games, Nash equilibria computation.)

**Solution:**

**Prisoner's Dilemma Nash Equilibrium:**

The stage-game payoff matrix (row player's payoff):

|           | C     | D     |
|-----------|-------|-------|
| **C**     | 3, 3  | 0, 5  |
| **D**     | 5, 0  | 1, 1  |

**Best response analysis:**
- If opponent plays C: row player prefers D (5 > 3). 
- If opponent plays D: row player prefers D (1 > 0).

D is strictly dominant — the unique Nash equilibrium is **(D, D)** with payoff (1, 1).

**Social optimum:** (C, C) with payoff (3, 3). The Nash equilibrium is socially inefficient — both players would prefer mutual cooperation but cannot sustain it without commitment in a one-shot game.

**For mixed strategies:** There are no other Nash equilibria since D strictly dominates C — any mixed strategy that assigns positive probability to C cannot be a best response.

---

## Exercise 16.2
**Statement:** (From ch16 — iterated game strategies, Folk Theorem.)

**Solution:**

**Folk Theorem for Iterated Games:**

In the infinitely repeated Prisoner's Dilemma with discount factor $\delta$, any individually rational payoff profile can be sustained as a Nash equilibrium if $\delta$ is sufficiently large.

**Cooperation as a Nash equilibrium under Tit-for-Tat:**

Suppose both players use Tit-for-Tat (TfT): cooperate first, then mimic opponent's last action.

If both use TfT and player 1 deviates to D in round $t$:
- Round $t$: player 1 gets 5 (vs 3 for cooperation).
- Rounds $t+1, t+2, \ldots$: TfT leads to mutual defection forever: payoff 1 per round.

**Deviation payoff:**
$$V_\mathrm{deviate} = 5 + \frac{\delta}{1-\delta}\cdot1.$$

**Cooperation payoff:**
$$V_\mathrm{cooperate} = \frac{3}{1-\delta}.$$

Cooperation is preferred when $V_\mathrm{cooperate} \ge V_\mathrm{deviate}$:
$$\frac{3}{1-\delta} \ge 5 + \frac{\delta}{1-\delta} \implies 3 \ge 5(1-\delta) + \delta \implies 3 \ge 5 - 4\delta \implies \delta \ge \frac{1}{2}.$$

For $\delta \ge 0.5$, mutual TfT is a Nash equilibrium sustaining cooperation. $\blacksquare$

---

## Exercise 16.3
**Statement:** (From ch16 — Q-learning in simultaneous-action games, convergence issues.)

**Solution:**

**Independent Q-learning in PD:**

Each agent maintains its own Q-table and updates independently using the opponent's previous action as the state.

**Convergence issues:** Independent Q-learning (IQL) is not guaranteed to converge in general-sum games because each agent's environment is non-stationary: as both agents learn, the opponent's policy changes, violating the stationarity assumption required for Q-learning convergence.

**Conditions for convergence:**
1. If one agent has a stationary policy (e.g., TfT), the other agent's Q-learning converges to the best response.
2. In zero-sum games: Nash equilibria can be found via minimax Q-learning.
3. In cooperative games with shared reward: all agents converge since they share a common objective.

**Empirical behaviour:** In the IPD, IQL often converges to mutual cooperation because both agents have symmetric incentives and the repeated-game dynamics reinforce cooperation (as shown in the notebook).

---

## Exercise 16.4
**Statement:** (From ch16 — MARL communication, emergent protocols.)

**Solution:**

**Emergent Communication in Multi-Agent Systems:**

Consider a cooperative reference game: sender observes target object; receiver must identify it from a set; both rewarded when receiver selects correctly.

**Protocol emergence:** Agents start with random symbols. Through RL (maximising shared reward), the sender learns to associate symbols with objects that the receiver can decode. The resulting "language" is emergent — not programmed — and is grounded in the task's communication needs.

**Key findings (Mordatch & Abbeel, 2018; Lazaridou et al.):**
1. Emergent languages are composable: new object combinations can be described by composing existing symbols.
2. The protocol is not interpretable to humans without a grounding mapping.
3. Protocol complexity scales with the number of objects and attributes the agents must distinguish.

**Limitations:** Emergent protocols tend to be brittle (sensitive to initialisation), non-compositional in practice, and fail to generalise to new agents (zero-shot generalisation is poor).

---

## Exercise 16.5
**Statement:** (From ch16 — self-play and policy evaluation.)

**Solution:**

**Self-Play Dynamics:**

Self-play trains an agent by having it compete against versions of itself. The key insight is that the Nash equilibrium is a fixed point of self-play — if both players use the Nash strategy, neither benefits from deviating.

**Convergence in zero-sum games:**
- For two-player zero-sum games, self-play converges (in the average-policy sense) to a Nash equilibrium by the minimax theorem.
- The average of all past policies converges even if the instantaneous policy cycles.

**Non-transitivity (Rock-Paper-Scissors):**
No single policy dominates all others. Self-play produces a policy that cycles: if training against itself produces a rock-heavy policy, this beats scissors-heavy, which beats paper-heavy, which beats the original. The average over the cycle approaches the Nash equilibrium (uniform mix).

**Population-based training:** Instead of a single self-play agent, maintain a population of agents. Train each against a mixture of population members, weighted by their relative strength (Elo ratings). This avoids cycles and converges more reliably.

---

## Exercise 16.6
**Statement:** (From ch16 — CTDE: centralised training, decentralised execution.)

**Solution:**

**Centralised Training, Decentralised Execution (CTDE):**

During training, agents share global state information (e.g., all agents' observations, actions) to compute coordinated updates. At execution, each agent acts using only its local observation.

**Why CTDE helps:**
1. **Credit assignment:** With shared reward, agents cannot distinguish their individual contributions from global outcomes. A centralised critic that sees all observations can compute counterfactual baselines: "what would the reward have been if agent $i$ had taken a different action while others were fixed?"
2. **Coordination:** Centralised training can detect coordination opportunities that decentralised training misses.

**QMIX:** Represents $Q_\mathrm{tot}(s, \mathbf{a})$ as a monotone combination of individual Q-values $Q_i(s_i, a_i)$. This ensures consistency: $\arg\max_\mathbf{a} Q_\mathrm{tot} = (\arg\max_{a_i} Q_i(s_i, a_i))_{i}$, enabling decentralised execution while using centralised training.

---

## Exercise 16.7
**Statement:** (From ch16 — Stackelberg games, leader-follower dynamics.)

**Solution:**

**Stackelberg Equilibrium:**

In a Stackelberg game, the leader commits to a strategy first; the follower best-responds. The Stackelberg equilibrium is $(x^*, y^*(x^*))$ where:
$$x^* = \arg\max_x U_L(x, y^*(x)), \quad y^*(x) = \arg\max_y U_F(x, y).$$

**Comparison with Nash:** The leader has a first-mover advantage in general:
$$U_L(x^*, y^*(x^*)) \ge U_L^{\mathrm{Nash}}.$$

This is because the leader can anticipate the follower's response and choose the action that maximises their own utility given this.

**RLHF as Stackelberg:** The alignment process can be viewed as a Stackelberg game:
- **Leader**: the human (specifies reward model/preferences)
- **Follower**: the AI (best-responds to the reward signal)

The human must anticipate how the AI will exploit the reward model (Goodhart's law) and design the reward to account for this strategic response.

---

## Exercise 16.8
**Statement:** (From ch16 — reward specification, mechanism design in MARL.)

**Solution:**

**Mechanism Design for Multi-Agent Cooperation:**

In mechanism design, a principal designs a reward structure to induce desired behaviour from self-interested agents.

**Problem:** In a cooperative task (e.g., team-based game), agents may free-ride — letting others do the work while collecting shared rewards.

**Solution approach — individual reward shaping:**
- Assign individual rewards based on each agent's counterfactual contribution: $r_i = R_\mathrm{team} - R_\mathrm{team}(a_{-i}, \emptyset_i)$, where $\emptyset_i$ denotes agent $i$ not acting.
- This "difference reward" measures how much worse the team would perform without agent $i$.

**VDN (Value Decomposition Networks):** Decomposes the global Q-value additively: $Q_\mathrm{tot} = \sum_i Q_i$. Each agent's Q-value approximates its individual contribution. Simple but restricts expressiveness.

**Connection to RL theory:** The mechanism design perspective frames MARL as a reward-engineering problem. The alignment tax (cooperative efficiency loss due to misaligned incentives) can be minimised through careful reward design, analogous to the VCG mechanism in economics.
