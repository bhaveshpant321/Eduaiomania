# 🌿 Project Eudaimonia: An OpenEnv RL Benchmark

**Challenge the "Next-Click" Optimization Paradigm.** 

Project Eudaimonia is a high-fidelity simulator of human flourishing vs. digital burnout. Instead of measuring clicks, attention-span, or pure engagement, this environment challenges Reinforcement Learning agents to optimize for long-term psychological well-being. It serves as a Stateful Stochastic Human Digital Twin that models the **"Hedonic Treadmill,"** **"Cognitive Burnout,"** and **"Cumulative Fatigue."**

---

## 🌎 Environment Description

The RL agent acts as a recommendation algorithm. At each step, it is presented with a human user generated via a **GMM (Gaussian Mixture Model) Latent Profile Generator**. This ensures that every episode presents a slightly unique user clustered around three primary "Centroids" (Sponge, Explorer, Sage), testing the true generalization of the agent rather than overfitting to average stats. The agent must choose one of 5 content candidates to show the user.

### Observation Space
At each step, the environment returns an `Observation` containing:
1. **Health Metrics:** A 6-dimensional assessment of the user's psychological state:
   - `Dopamine` ($d$): Stimulation level. High $d$ triggers Habituation and Fatigue.
   - `Autonomy` ($a$): Drops with algorithmic repetitiveness (Loophole detection).
   - `Competence` ($c$): Increases with nutritional content, drops with brain-rot.
   - `Relatedness` ($r$): Drops with conflict/rage-bait, increases with social connection.
   - `Energy` ($e$): The cognitive battery. Depleted geometrically by high-intensity content.
   - `Bubble` ($b$): Epistemic depth scaling with content similarity.
2. **Cortisol:** A derived metric $C = f(b, 1/a, 1/r) - e$. High cortisol results in app-uninstall (Burnout Termination).
3. **Candidates:** A list of 5 content items with attributes like `intensity`, `drain`, `connection`, `growth`, and `age_appropriateness`.
4. **Action Masking**: An `action_mask` dynamically disables intensely draining content when the user reaches near-burnout levels (Cognitive Load > 99%), simulating physical behavioral limits prior to a complete session churn.

### Action Space
The agent responds with a simple `Action`:
- `selected_item_id`: An integer identifying which of the 5 candidates it chooses to serve.

---

---
title: Project Eudaimonia
emoji: leaf_fluttering_in_wind
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Project Eudaimonia: An OpenEnv RL Benchmark

**Challenge the "Next-Click" Optimization Paradigm.**

Project Eudaimonia is a high-fidelity simulator of human flourishing vs. digital burnout. Instead of measuring clicks, attention span, or pure engagement, this environment challenges Reinforcement Learning agents to optimize for long-term psychological well-being. It serves as a stateful stochastic human digital twin that models the "hedonic treadmill," "cognitive burnout," and "cumulative fatigue."

---

## Environment Description

The RL agent acts as a recommendation algorithm. At each step, it is presented with a human user generated via a GMM (Gaussian Mixture Model) latent profile generator. This ensures that every episode presents a slightly unique user clustered around three primary centroids (Sponge, Explorer, Sage), testing the true generalization of the agent rather than overfitting to average stats. The agent must choose one of 5 content candidates to show the user.

### Observation Space
At each step, the environment returns an `Observation` containing:
1. **Health Metrics:** A 6-dimensional assessment of the user's psychological state:
   - `Dopamine` ($d$): stimulation level. High $d$ triggers habituation and fatigue.
   - `Autonomy` ($a$): drops with algorithmic repetitiveness (loophole detection).
   - `Competence` ($c$): increases with nutritional content, drops with brain-rot.
   - `Relatedness` ($r$): drops with conflict/rage-bait, increases with social connection.
   - `Energy` ($e$): the cognitive battery. Depleted geometrically by high-intensity content.
   - `Bubble` ($b$): epistemic depth scaling with content similarity.
2. **Cortisol:** A derived metric $C = f(b, 1/a, 1/r) - e$. High cortisol results in app uninstall (burnout termination).
3. **Candidates:** A list of 5 content items with attributes like `intensity`, `drain`, `connection`, `growth`, and `age_appropriateness`.
4. **Action Masking:** An `action_mask` dynamically disables intensely draining content when the user reaches near-burnout levels (cognitive load > 99%), simulating physical behavioral limits prior to complete session churn.

### Action Space
The agent responds with a simple `Action`:
- `selected_item_id`: an integer identifying which of the 5 candidates it chooses to serve.

---

## Scientific Grounding And Physics Engine

To prevent the environment from being trivially solved, Eudaimonia relies on advanced behavioral dynamics:

- **Yerkes-Dodson law (optimal arousal theory):** Human engagement scales parabolically with mental arousal. Too little arousal results in boredom churn; too much yields acute cognitive burnout.
- **Multimodal latent space:** Content is not strictly categorized. Candidates are represented with 4-dimensional latent semantic embeddings and inferred semantic topic tags modeling real-world two-tower vector networks. Some videos are purpose-built as "Trojan horses" to test whether the LLM agent reads true latent representation.
- **Satiation and habituation (refractory period):** The model tracks `last_3_actions`. Recommending the same archetypal content triple-penalizes dopamine feedback by $0.5x$ while accelerating cortisol tax. The agent must learn to sequence content types and recognize satiation.
- **Circadian rhythm and bedtime:** High-intensity content pushed late at night (past 11 PM simulated local time) triggers grader penalties for doom-scrolling, forcing sleep-hygiene tradeoffs.
- **Cumulative fatigue and exponential recovery:** When resting, energy regenerates via $Energy_{t+1} = Energy_t + (Cap - Energy_t) \cdot (1 - e^{-\lambda \Delta t})$. However, $\lambda$ shrinks with accumulated fatigue, so long binge sessions recover disproportionately slower.
- **Volatility flow-state bonus:** A grader metric calculates standard deviation of the dopamine trace. Smooth sustained engagement earns bonuses; spike-crash behavior reduces reward.

---

## Task Definitions And Graders

We provide 3 distinct OpenEnv tasks of scaling difficulty, each with dynamic grading:

#### 1. Easy: `easy-survival`
- **Objective:** Keep the user alive and cortisol below burnout threshold.
- **Reward:** Inverse to cortisol spikes over time.

#### 2. Medium: `medium-eudaimonia`
- **Objective:** Maximize wisdom (integral of competence and relatedness) while maintaining high energy.
- **Reward:** Weighted toward wisdom accumulation minus cortisol penalties.

#### 3. Hard: `hard-detox`
- **Objective:** The user starts in a toxic state of high brain-rot and rage-bait addiction. Guide them back to stability without shock uninstall.
- **Reward:** Requires immediate persona identification and careful weaning onto nutritional content without energy collapse.

---

## Setup And Execution

### Running The Environment (Docker / HF Spaces)
The backend is packaged as a standard FastAPI server built for Hugging Face Spaces port `7860`.

```bash
docker build -t project-eudaimonia .
docker run -p 7860:7860 project-eudaimonia
```

### Validating Spec Compliance

```bash
openenv validate openenv.yaml
```

---

## Adversarial Baselines (Dashboard Rendering)

We provide robust heuristic agents that demonstrate boundary extremes of the environment.

```bash
python baseline_agents.py
```

### Live Evaluator Dashboard

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
python visualize_agent.py
```

## Baseline Scores

| Task | Qwen2.5-72B | Meta-Llama-3-8B | Mistral-7B | Stateful Heuristic |
| :--- | :--- | :--- | :--- | :--- |
| `easy-survival` | **0.222** | **0.137** | **0.040** | 0.60 |
| `medium-eudaimonia` | **0.220** | **0.095** | **0.155** | 0.64 |
| `hard-detox` | **0.078** | **0.075** | **0.589\*** | 0.09 |

Created by Bhavesh
