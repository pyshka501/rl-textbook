# Reinforcement Learning: From Bandits to LLM Alignment

<p align="center">
  <strong>A comprehensive textbook bridging classical RL theory and modern language model alignment</strong>
</p>

<p align="center">
  <a href="book/main.pdf">📖 Read the Book (PDF)</a> •
  <a href="#-notebooks-with-colab">💻 Notebooks</a> •
  <a href="exercises/">✏️ Exercises</a> •
  <a href="translations/">🌍 Translations</a>
</p>

---

## About

This open-source textbook covers reinforcement learning from first principles through to cutting-edge LLM alignment methods. It is designed for graduate students, ML engineers, and researchers who want a unified treatment of classical RL theory and modern post-training techniques.

**Inspired by:**
- *Reinforcement Learning: An Introduction* by Sutton & Barto — for rigorous pedagogy of classical RL
- *The RLHF Book* by Nathan Lambert — for practical insights on LLM alignment

## 💻 Notebooks with Colab

Every chapter has a companion Jupyter notebook you can run instantly — no setup required. Click the badge to launch in Google Colab:

### Part I: Foundations

| Ch | Title | Launch | Exercises |
|----|-------|--------|-----------|
| 1 | Introduction to Reinforcement Learning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch01_introduction.ipynb) | [Solutions](exercises/ch01_solutions.md) |
| 2 | Multi-Armed Bandits | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch02_bandits.ipynb) | [Solutions](exercises/ch02_solutions.md) |
| 3 | MDPs and the Bellman Equation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch03_mdp_bellman.ipynb) | [Solutions](exercises/ch03_solutions.md) |
| 4 | Dynamic Programming | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch04_dynamic_programming.ipynb) | [Solutions](exercises/ch04_solutions.md) |
| 5 | Monte Carlo Methods | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch05_monte_carlo.ipynb) | [Solutions](exercises/ch05_solutions.md) |
| 6 | Temporal-Difference Learning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch06_td_learning.ipynb) | [Solutions](exercises/ch06_solutions.md) |

### Part II: Deep Reinforcement Learning

| Ch | Title | Launch | Exercises |
|----|-------|--------|-----------|
| 7 | Value Function Approximation & Deep RL | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch07_deep_rl.ipynb) | [Solutions](exercises/ch07_solutions.md) |
| 8 | Policy Gradient Methods | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch08_policy_gradients.ipynb) | [Solutions](exercises/ch08_solutions.md) |
| 9 | Actor-Critic Methods and PPO | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch09_actor_critic_ppo.ipynb) | [Solutions](exercises/ch09_solutions.md) |

### Part III: RL for Language Models

| Ch | Title | Launch | Exercises |
|----|-------|--------|-----------|
| 10 | Reward Shaping and Reward Models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch10_reward_models.ipynb) | [Solutions](exercises/ch10_solutions.md) |
| 11 | Language Models and Post-Training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch11_lm_post_training.ipynb) | [Solutions](exercises/ch11_solutions.md) |
| 12 | DPO and Alignment Methods | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch12_dpo_alignment.ipynb) | [Solutions](exercises/ch12_solutions.md) |
| 13 | GRPO and Online RL | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch13_grpo.ipynb) | [Solutions](exercises/ch13_solutions.md) |
| 14 | Practical RLHF | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch14_practical_rlhf.ipynb) | [Solutions](exercises/ch14_solutions.md) |

### Part IV: Frontiers

| Ch | Title | Launch | Exercises |
|----|-------|--------|-----------|
| 15 | Reasoning, Tool Use, and Reinforcement Finetuning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch15_reasoning.ipynb) | [Solutions](exercises/ch15_solutions.md) |
| 16 | Multi-Agent Systems and RL | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch16_multi_agent.ipynb) | [Solutions](exercises/ch16_solutions.md) |
| 17 | Open Problems and Emerging Directions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/notebooks/ch17_open_problems.ipynb) | [Solutions](exercises/ch17_solutions.md) |

**Appendix:** [Mathematical Background](book/backmatter/appendix_math.tex)

> **Compute requirements:** Chapters 1–6 run on CPU. Chapters 7–17 benefit from a free T4 GPU on Colab/Kaggle. No paid compute is required for any notebook.

## Quick Start

### Read the book
Download [`book/main.pdf`](book/main.pdf) or build from source:

```bash
cd book
latexmk -pdf main.tex
```

**Requirements:** TeX Live 2023+ with `texlive-latex-extra`, `texlive-fonts-recommended`, `texlive-science`, `texlive-pictures`, `texlive-fonts-extra`.

### Run a notebook in 1 click
Click any **Open in Colab** badge above. Dependencies install automatically in the first cell.

### Run locally
```bash
pip install -r notebooks/requirements.txt
jupyter lab notebooks/
```

## Repository Structure

```
rl-textbook/
├── book/                   # LaTeX source for the textbook
│   ├── main.tex            # Master file
│   ├── chapters/           # Ch01–Ch17
│   ├── frontmatter/        # Preface, Notation
│   ├── backmatter/         # Appendix, Afterword
│   ├── figures/            # Images and diagrams
│   └── style_guide.md      # Writing conventions
├── notebooks/              # Jupyter notebooks (one per chapter)
│   ├── requirements.txt
│   └── ch01–ch17 .ipynb
├── exercises/              # Exercises with full solutions
│   └── ch01–ch17 _solutions.md
├── translations/           # Translations (ru, zh, es, de, fr, ja, ko)
├── LICENSE                 # CC BY-NC-ND 4.0
├── CONTRIBUTING.md         # Contribution guidelines
└── README.md               # This file
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- **Found a typo?** Open a PR
- **Want to translate?** See [translations/](translations/)
- **Have an idea?** Open an issue

## Citation

```bibtex
@book{rl-textbook-2026,
  title     = {Reinforcement Learning: From Bandits to LLM Alignment},
  year      = {2026},
  url       = {https://github.com/pyshka501/rl-textbook}
}
```

## License

This project is licensed under **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** — see [LICENSE](LICENSE).

You are free to read, share, and link to this work for non-commercial purposes with attribution. **Printing for sale, commercial publishing, and derivative works require prior written permission from the author.** Open an issue or contact the author to request permission.
