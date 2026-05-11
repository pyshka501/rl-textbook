# Практические ноутбуки (русская версия)

Интерактивные Jupyter-ноутбуки к каждой главе на русском языке. **Нажмите на бейдж, чтобы запустить в Google Colab — установка не требуется.**

## Запуск ноутбуков

### Google Colab (рекомендуется — 1 клик)
Нажмите бейдж **Open in Colab** рядом с любым ноутбуком ниже. Все зависимости устанавливаются автоматически в первой ячейке.

### Kaggle
Загрузите ноутбук на Kaggle, включите «Internet» в настройках и выберите GPU T4.

### Локально
```bash
pip install -r ../../../notebooks/requirements.txt
jupyter lab
```

## Указатель ноутбуков

| Гл | Ноутбук | Open in Colab | Темы | Время |
|----|----------|---------------|--------|---------|
| 1 | `ch01_introduction_ru.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/translations/ru/notebooks/ch01_introduction_ru.ipynb) | Цикл агент-среда, GridWorld | ~2 мин |
| 2 | `ch02_bandits_ru.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/translations/ru/notebooks/ch02_bandits_ru.ipynb) | $\varepsilon$-greedy, UCB, выборка Томпсона | ~3 мин |
| 3 | `ch03_mdp_bellman_ru.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/translations/ru/notebooks/ch03_mdp_bellman_ru.ipynb) | Определение MDP, оценивание Беллмана | ~2 мин |
| 4 | `ch04_dynamic_programming_ru.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/translations/ru/notebooks/ch04_dynamic_programming_ru.ipynb) | Итерация стратегии и ценности, GridWorld | ~3 мин |
| 5 | `ch05_monte_carlo_ru.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/translations/ru/notebooks/ch05_monte_carlo_ru.ipynb) | МК по первому посещению, Blackjack | ~5 мин |
| 6 | `ch06_td_learning_ru.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/translations/ru/notebooks/ch06_td_learning_ru.ipynb) | TD(0), SARSA, Q-Learning | ~5 мин |
| 7 | `ch07_deep_rl_ru.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/translations/ru/notebooks/ch07_deep_rl_ru.ipynb) | DQN, experience replay | ~8 мин |
| 8 | `ch08_policy_gradients_ru.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/translations/ru/notebooks/ch08_policy_gradients_ru.ipynb) | REINFORCE, базовые линии, RLOO | ~8 мин |
| 9 | `ch09_actor_critic_ppo_ru.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/translations/ru/notebooks/ch09_actor_critic_ppo_ru.ipynb) | A2C, PPO-Clip | ~10 мин |
| 10 | `ch10_reward_models_ru.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/translations/ru/notebooks/ch10_reward_models_ru.ipynb) | Шейпинг вознаграждения, Брэдли\,---\,Терри | ~5 мин |
| 11 | `ch11_lm_post_training_ru.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/translations/ru/notebooks/ch11_lm_post_training_ru.ipynb) | GPT-2, SFT, сэмплирование | ~10 мин |
| 12 | `ch12_dpo_alignment_ru.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/translations/ru/notebooks/ch12_dpo_alignment_ru.ipynb) | Функция потерь DPO, неявное вознаграждение | ~10 мин |
| 13 | `ch13_grpo_ru.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/translations/ru/notebooks/ch13_grpo_ru.ipynb) | Групповые относительные преимущества | ~8 мин |
| 14 | `ch14_practical_rlhf_ru.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/translations/ru/notebooks/ch14_practical_rlhf_ru.ipynb) | TRL, PPO для языковых моделей | ~15 мин |
| 15 | `ch15_reasoning_ru.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/translations/ru/notebooks/ch15_reasoning_ru.ipynb) | CoT, мажоритарное голосование, инструменты | ~5 мин |
| 16 | `ch16_multi_agent_ru.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/translations/ru/notebooks/ch16_multi_agent_ru.ipynb) | Дилемма заключённого, self-play | ~5 мин |
| 17 | `ch17_open_problems_ru.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyshka501/rl-textbook/blob/main/translations/ru/notebooks/ch17_open_problems_ru.ipynb) | Decision Transformer, offline RL | ~8 мин |

> **GPU**: главы 1–6 запускаются на CPU. Главы 7–17 выигрывают от бесплатного T4 GPU (Runtime → Change runtime type → T4 GPU в Colab).

## Заметки о переводе

- Markdown-ячейки переведены на русский язык; код, переменные и комментарии в коде оставлены на английском для согласованности с оригиналом и удобства запуска без правок.
- Технические термины переданы согласно глоссарию русского перевода учебника (`translations/ru/`).
- Для ссылок на оригинальную (английскую) версию, см. `notebooks/README.md` в корне репозитория.
