# Chapter 1: Introduction to Reinforcement Learning — Solutions

## Exercise 1.1
**Statement:** Identify three real-world problems naturally formulated as RL tasks, describing the agent, environment, action space, and reward signal for each.

**Solution:**

**1. Game-playing (e.g., Chess)**
- *Agent*: the chess engine
- *Environment*: the board state and opponent
- *Action space*: all legal moves from the current position
- *Reward*: +1 for win, −1 for loss, 0 for draw (terminal)

**2. Robotic manipulation (e.g., grasping objects)**
- *Agent*: the robot controller
- *Environment*: the physical world (object positions, robot joints)
- *Action space*: joint torques or end-effector velocities
- *Reward*: +1 when object is successfully grasped and lifted; −0.01 per time step as a living penalty

**3. Personalised content recommendation**
- *Agent*: the recommendation system
- *Environment*: the user and content catalogue
- *Action space*: choice of item to recommend next
- *Reward*: +1 for click/engagement, 0 for skip; long-term: user retention signal

---

## Exercise 1.2
**Statement:** Explain why supervised learning may fail for teaching a robot to walk, and why RL is more appropriate. What role does exploration–exploitation play?

**Solution:**

**Why SL fails:**  
Supervised learning requires i.i.d. labelled examples of the form (observation, correct action). For robot locomotion, obtaining such labels is extremely difficult because:
1. *Covariate shift*: at test time the robot visits states never seen in the expert demonstrations. When SL commits a small error, the robot drifts to a new state; the trained policy has no label for that state and may fail catastrophically.
2. *Compounding errors*: small per-step errors compound across a trajectory, diverging from the training distribution (the exposure-bias problem in imitation learning).
3. *Feedback loop*: a walking robot must continuously adapt to perturbations; a fixed mapping from observations to actions cannot capture the necessary reactive feedback.

**Why RL is appropriate:**  
RL allows the robot to improve by trial and error using the reward signal (e.g., forward velocity minus energy spent). The agent can visit novel states and learn to recover — exactly the feedback loop needed.

**Exploration–exploitation:**  
The robot must *explore* (try new gait patterns, perhaps fall) to discover effective locomotion strategies. If it always *exploits* the current best-known policy, it may get stuck in a local optimum (e.g., staying still has zero fall-penalty). A good schedule (e.g., early high exploration, later exploitation) is essential for finding good gaits.

---

## Exercise 1.3
**Statement:** Show that for $\gamma \in [0,1)$ and bounded rewards $|R_t| \le R_{\max}$, the discounted return satisfies $|G_t| \le R_{\max}/(1-\gamma)$.

**Solution:**

By definition:
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}.$$

Taking absolute values and applying the triangle inequality:
$$|G_t| \le \sum_{k=0}^{\infty} \gamma^k |R_{t+k+1}| \le \sum_{k=0}^{\infty} \gamma^k R_{\max} = R_{\max} \sum_{k=0}^{\infty} \gamma^k.$$

Since $\gamma \in [0,1)$ the geometric series converges:
$$\sum_{k=0}^{\infty} \gamma^k = \frac{1}{1-\gamma}.$$

Therefore $|G_t| \le \dfrac{R_{\max}}{1-\gamma}$. $\blacksquare$

---

## Exercise 1.4
**Statement:** For each task, determine the most natural ML paradigm and justify.

**Solution:**

**(a) Classifying emails as spam or not spam.**  
**Supervised learning.** We have labelled examples (email, spam/not-spam) and want to learn a mapping. The label is cheaply obtained (user feedback or manual labelling) and the i.i.d. assumption is reasonable.

**(b) Training a chatbot to be helpful and harmless using human preference data.**  
**Reinforcement learning (from human feedback, RLHF).** There is no single "correct" response — quality is defined by human preferences over pairs of responses. This is naturally modelled as a reward signal, and RL is used to optimise a policy against that reward. (One could also argue imitation learning for the SFT stage.)

**(c) Discovering customer segments from purchase histories.**  
**Unsupervised learning** (specifically clustering, e.g., k-means). There are no labels — the goal is to discover latent structure in the data.

**(d) Teaching a self-driving car to navigate using recordings of expert drivers.**  
**Imitation learning** (behavioural cloning / inverse RL). We have demonstrations from experts and want to mimic their behaviour. Pure SL (behavioural cloning) is simplest; inverse RL recovers the reward and then applies RL. The covariate-shift problem motivates methods like DAgger over plain SL.
